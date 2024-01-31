# ref: https://huggingface.co/docs/transformers/v4.36.1/en/main_classes/pipelines#transformers.DocumentQuestionAnsweringPipeline

from io import BytesIO
from typing import Any, Dict, Optional, TypedDict

from fastapi import HTTPException, Request
from generic.tasks.common import AbstractInferenceHandler
from generic.tasks.utils import (
    form_field_to_boolean,
    form_field_to_float,
    form_field_to_int,
    form_field_to_json,
    form_field_to_str,
)
from PIL import Image
from pydantic import BaseModel
from starlette.datastructures import UploadFile


class DocumentQuestionAnsweringJSONInput(BaseModel):
    image: str
    top_k: Optional[int] = None
    timeout: Optional[float] = None
    question: str
    doc_stride: Optional[int] = None
    max_answer_len: Optional[int] = None
    max_seq_len: Optional[int] = None
    max_question_len: Optional[int] = None
    handle_impossible_answer: Optional[bool] = None
    lang: Optional[str] = None
    tesseract_config: Optional[str] = None
    kwargs: Optional[Dict[str, Any]] = {}


class DocumentQuestionAnsweringInferenceInput(TypedDict):
    image: str | bytes
    kwargs: Optional[Dict[str, Any]]


class DocumentQuestionAnsweringInference(TypedDict):
    """
    score: The probability associated to the answer.
    start: The start word index of the answer (in the OCR’d version of the input or provided word_boxes).
    end: The end word index of the answer (in the OCR’d version of the input or provided word_boxes).
    answer: The answer to the question.
    words: The index of each word/box pair that is in the answer
    """

    score: float
    start: int
    end: int
    answer: str
    words: list[int]


class DocumentQuestionAnsweringInferenceHandler(AbstractInferenceHandler):
    async def parse_request(
        self, request: Request, temp_dir: str
    ) -> DocumentQuestionAnsweringInferenceInput:
        """
        ContentType: multipart/form-data, application/json
        Returns: {image:Union[str,bytes], topk:Optional[None], timeout:Optional[None], question:str, doc_stride:Optional[None], max_answer_len:Optional[None], max_seq_len:Optional[None], max_question_len:Optional[None], handle_impossible_answer:Optional[None], lang:Optional[None], tesseract_config:Optional[None]}
        """
        if request.headers["content-type"].startswith("multipart/form-data"):
            form = await request.form()
            raw_image = form.get("image")

            image: str | bytes
            if isinstance(raw_image, UploadFile):
                image_file = BytesIO(await raw_image.read())
                pil_image = Image.open(image_file)
                image = pil_image
            else:
                image = raw_image

            return DocumentQuestionAnsweringInferenceInput(
                image=image,
                kwargs=dict(
                    top_k=form_field_to_int("top_k", form.get("top_k")),
                    timeout=form_field_to_float("timeout", form.get("timeout")),
                    question=form_field_to_str("question", form.get("question")),
                    doc_stride=form_field_to_int("doc_stride", form.get("doc_stride")),
                    max_answer_len=form_field_to_int(
                        "max_answer_len", form.get("max_answer_len")
                    ),
                    max_seq_len=form_field_to_int(
                        "max_seq_len", form.get("max_seq_len")
                    ),
                    max_question_len=form_field_to_int(
                        "max_question_len", form.get("max_question_len")
                    ),
                    handle_impossible_answer=form_field_to_boolean(
                        "handle_impossible_answer", form.get("handle_impossible_answer")
                    ),
                    lang=form_field_to_str("lang", form.get("lang")),
                    tesseract_config=form_field_to_str(
                        "tesseract_config", form.get("tesseract_config")
                    ),
                    **(
                        form_field_to_json(
                            name="kwargs", value=form.get("kwargs"), on_none={}
                        )
                    ),
                ),
            )

        elif request.headers["content-type"].startswith("application/json"):
            data = await request.json()
            body = DocumentQuestionAnsweringJSONInput.model_validate(data)
            return DocumentQuestionAnsweringInferenceInput(
                image=body.image,
                kwargs=dict(
                    top_k=body.top_k,
                    timeout=body.timeout,
                    question=body.question,
                    doc_stride=body.doc_stride,
                    max_answer_len=body.max_answer_len,
                    max_seq_len=body.max_seq_len,
                    max_question_len=body.max_question_len,
                    handle_impossible_answer=body.handle_impossible_answer,
                    lang=body.lang,
                    tesseract_config=body.tesseract_config,
                    **body.kwargs,
                ),
            )
        else:
            raise HTTPException(status_code=400, detail="Invalid content type")

    # async def infer(
    #     self, data: DocumentQuestionAnsweringInferenceInput
    # ) -> Union[
    #     DocumentQuestionAnsweringInference, List[DocumentQuestionAnsweringInference]
    # ]:
    #     return self.pipeline(**data)

    # async def respond(
    #     self,
    #     output: Union[
    #         DocumentQuestionAnsweringInference, List[DocumentQuestionAnsweringInference]
    #     ],
    # ):
    #     json_compatible_output = jsonable_encoder(output)
    #     return JSONResponse(content=json_compatible_output)
