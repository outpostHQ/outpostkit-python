import json
import os
from argparse import ArgumentParser
from asyncio import Semaphore
from tempfile import TemporaryDirectory

from fastapi import Request
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from transformers import pipeline as transformers_pipeline

from outpostkit.exceptions import OutpostError
from outpostkit.template_gen.templates.audio_classification import (
    request_parser as audio_classification_request_parser,
)
from outpostkit.template_gen.utils.precision import parse_dtype
from outpostkit.template_gen.utils.repo import clone_outpost_repo

task_type_handlers = {"audio-classification": audio_classification_request_parser}


def add_generic_template_args(parser: ArgumentParser) -> ArgumentParser:
    parser.add_argument(
        "--max-concurrent-inferences",
        type=int,
        default=1,
        help="maximum number of concurrent inferences to be performed",
    )
    parser.add_argument("--torch-dtype", type=str, default=None, help="torch dtype")
    parser.add_argument(
        "--framework",
        type=str,
        choices=["tf", "pt"],
        default=None,
        help="Model Framework",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="config path",
    )

    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="The model name used in the API. If not "
        "specified, the model name will be the same as "
        "the model directory",
    )
    parser.add_argument(
        "--load-source",
        type=str,
        choices=["huggingface", "outpost"],
        default="outpost",
        help="load model weights from",
    )
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--kwargs", type=json.loads, default=None, help="kwargs")
    parser.add_argument("--model-kwargs", type=json.loads, default=None, help="kwargs")
    parser.add_argument("--task-type", type=str, default=None, help="task type")
    parser.add_argument("--model-store", type=str, default="/model", help="task type")
    parser.add_argument("--device-map", type=str, default="auto", help="device-map")
    return parser


def create_template_class_from_args(parser: ArgumentParser):
    args, unknown = parser.parse_known_args()
    if unknown:
        print("ignoring unknown arguements: ", unknown)
    model_dir: str
    init_f = None
    if args.load_source == "huggingface":
        model_dir = args.model_name

        def init_funct(self):
            token = None
            if os.environ.get("HUGGING_FACE_HUB_TOKEN") is not None:
                token = os.environ["HUGGING_FACE_HUB_TOKEN"]
            common_args = dict(
                config=args.config,
                task=args.task_type,
                model=model_dir,
                revision=args.revision,
                token=token,
                torch_dtype=parse_dtype(args.torch_dtype),
                model_kwargs=args.model_kwargs,
                trust_remote_code=True,
                device_map=args.device_map,
                framework=args.framework,
                **(args.kwargs or {}),
            )
            pipeline = transformers_pipeline(
                **common_args,
            )
            self.pipeline = pipeline
            self.semaphore = Semaphore(args.max_concurrent_predictions)

        init_f = init_funct

    else:
        [entity, name] = args.model_name.split(",", 1)
        model_dir = args.model_store

        def init_funct(self):
            token = None
            if os.environ.get("OUTPOST_PULL_TOKEN") is not None:
                token = os.environ["OUTPOST_PULL_TOKEN"]
            common_args = dict(
                config=args.config,
                task=args.task_type,
                model=model_dir,
                token=token,
                torch_dtype=parse_dtype(args.torch_dtype),
                model_kwargs=args.model_kwargs,
                tokenizer=args.tokenizer_dir,
                trust_remote_code=True,
                device_map=args.device_map,
                **(args.kwargs or {}),
            )
            clone_outpost_repo(
                name=name,
                entity=entity,
                repo_type="model",
                commit=args.revision,
                destination=model_dir,
                outpost_pull_token=token,
            )

            pipeline = transformers_pipeline(
                device_map="auto",
                **common_args,
            )
            self.pipeline = pipeline
            self.semaphore = Semaphore(args.max_concurrent_predictions)

        init_f = init_funct

    if args.task_type not in task_type_handlers:
        raise OutpostError("Task type not supported")

    req_parser = task_type_handlers[args.task_type]

    async def prediction_handler(self, request: Request, **kwargs) -> JSONResponse:
        with TemporaryDirectory() as temp_dir:
            async with self.semaphore:
                input = await req_parser(request=request, temp_dir=temp_dir)
                output = self.pipeline(**input)
                return JSONResponse(content=jsonable_encoder(output))

    TemplateClass = type(
        "InferenceTemplate", (), {"__init__": init_f, "predict": prediction_handler}
    )
    return TemplateClass


def gen_inference_template():
    parser = ArgumentParser("Templated Task Type class generator")
    parser = add_generic_template_args(parser)
    return create_template_class_from_args(parser=parser)
