from io import BytesIO
from typing import Any, Dict, List, Optional, TypedDict, Union

from fastapi import HTTPException, Request
from generic.tasks.common import AbstractInferenceHandler
from generic.tasks.utils import (
    form_field_to_float,
    form_field_to_int,
    form_field_to_json,
)
from PIL import Image
from pydantic import BaseModel
from starlette.datastructures import UploadFile


class VisualQuestionAnsweringJSONInput(BaseModel):
    images: Union[str, List[str]]
    questions: Union[str, List[str]]
    top_k: Optional[int] = None
    timeout: Optional[float] = None
    kwargs: Optional[Dict[str, Any]] = {}


class VisualQuestionAnsweringInferenceInput(TypedDict):
    images: Union[str, List[str], Image.Image, List[Image.Image]]
    questions: Union[str, List[str]]
    kwargs: Dict[str, Any]


class VisualQuestionAnsweringInference(TypedDict):
    label: str
    score: float


class VisualQuestionAnsweringInferenceHandler(AbstractInferenceHandler):
    async def parse_request(
        self, request: Request, temp_dir: str
    ) -> VisualQuestionAnsweringInferenceInput:
        """
        ContentType: multipart/form-data, application/json
        Args:
            - images : images to be processed. either string(url) or actual image file(through multipart);
            - questions: questions to be answered
            - topk ?: The number of top labels that will be returned by the pipeline.
            - timeout ?: The maximum time in seconds to wait for fetching images from the web.
        Returns: {images:Union[str, List[str]], questions: Union[str, List[str]], kwargs: Dict[str, Any]}
        """
        if request.headers["content-type"].startswith("multipart/form-data"):
            form = await request.form()
            questions = form.get("questions")

            raw_images = form.getlist("images")
            images: Union[List[str], List[bytes]] = []
            if len(raw_images) == 0:
                raise HTTPException(status_code=400, detail="No images provided.")

            if all(isinstance(sub, raw_images[0]) for sub in raw_images):
                if isinstance(raw_images[0], UploadFile):
                    for raw_image in raw_images:
                        image_file = BytesIO(await raw_image.read())
                        pil_image = Image.open(image_file)
                        images.append(pil_image)
                else:
                    images = raw_images

            return VisualQuestionAnsweringInferenceInput(
                images=images,
                questions=questions,
                kwargs=dict(
                    top_k=form_field_to_int("top_k", form.get("top_k")),
                    timeout=form_field_to_float("timeout", form.get("timeout")),
                    **form_field_to_json(
                        name="kwargs", value=form.get("kwargs"), on_none={}
                    ),
                ),
            )

        elif request.headers["content-type"].startswith("application/json"):
            data = await request.json()
            body = VisualQuestionAnsweringJSONInput.model_validate(data)
            return VisualQuestionAnsweringInferenceInput(
                images=body.images,
                questions=body.questions,
                kwargs=dict(top_k=body.top_k, timeout=body.timeout, **body.kwargs),
            )

        else:
            raise HTTPException(status_code=400, detail="Invalid content type")

    # async def infer(
    #     self, data: VisualQuestionAnsweringInferenceInput
    # ) -> Union[
    #     VisualQuestionAnsweringInference, List[VisualQuestionAnsweringInference]
    # ]:
    #     return self.pipeline(**data)

    # async def respond(
    #     self,
    #     output: Union[
    #         VisualQuestionAnsweringInference, List[VisualQuestionAnsweringInference]
    #     ],
    # ):
    #     json_compatible_output = jsonable_encoder(output)
    #     return JSONResponse(content=json_compatible_output)
