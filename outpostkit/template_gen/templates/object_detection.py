from io import BytesIO
from typing import Any, Dict, List, Optional, TypedDict, Union

from fastapi import HTTPException, Request
from generic.tasks.common import AbstractInferenceHandler
from generic.tasks.utils import (
    form_field_to_float,
    form_field_to_json,
)
from PIL import Image
from pydantic import BaseModel
from starlette.datastructures import UploadFile


class ObjectDetectionJSONInput(BaseModel):
    images: Union[str, List[str]]
    timeout: Optional[float] = None
    threshold: Optional[float] = None
    kwargs: Optional[Dict[str, Any]] = {}


class ObjectDetectionInferenceInput(TypedDict):
    images: Union[str, List[str], Image.Image, List[Image.Image]]
    kwargs: Dict[str, Any]


ObjectDetectionPrediction = Dict[str, Any]


class ObjectDetectionInferenceHandler(AbstractInferenceHandler):
    async def parse_request(
        self, request: Request, temp_dir: str
    ) -> ObjectDetectionInferenceInput:
        """
        ContentType: multipart/form-data, application/json
        Args:
            - images : images to be processed. either string(url) or actual image file(through multipart);
            - timeout ?: Timeout for the inference process.
            - threshold ?: Threshold for the detection process.
        Returns: {images:Union[str,bytes], timeout:Optional[None], threshold:Optional[None]}
        """
        if request.headers["content-type"].startswith("multipart/form-data"):
            form = await request.form()
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

            return ObjectDetectionInferenceInput(
                images=images,
                kwargs=dict(
                    timeout=form_field_to_float("timeout", form.get("timeout")),
                    threshold=form_field_to_float("threshold", form.get("threshold")),
                    **form_field_to_json(
                        name="kwargs", value=form.get("kwargs"), on_none={}
                    ),
                ),
            )

        elif request.headers["content-type"].startswith("application/json"):
            data = await request.json()
            body = ObjectDetectionJSONInput.model_validate(data)
            return ObjectDetectionInferenceInput(
                images=body.images,
                kwargs=dict(
                    timeout=body.timeout, threshold=body.threshold, **body.kwargs
                ),
            )

        else:
            raise HTTPException(status_code=400, detail="Invalid content type")

    # async def infer(
    #     self, data: ObjectDetectionInferenceInput
    # ) -> Union[ObjectDetectionPrediction, List[ObjectDetectionPrediction]]:
    #     return self.pipeline(**data)

    # async def respond(
    #     self, output: Union[ObjectDetectionPrediction, List[ObjectDetectionPrediction]]
    # ):
    #     json_compatible_output = jsonable_encoder(output)
    #     return JSONResponse(content=json_compatible_output)
