# ref: https://huggingface.co/docs/transformers/v4.36.1/en/main_classes/pipelines#transformers.ImageToTextPipeline
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


class ImageToTextJSONInput(BaseModel):
    images: Union[str, List[str]]
    timeout: Optional[float] = None
    max_new_tokens: Optional[int] = None
    kwargs: Optional[Dict[str, Any]] = {}


class ImageToTextInferenceInput(TypedDict):
    images: Union[str, List[str], Image.Image, List[Image.Image]]
    kwargs: Dict[str, Any]


class ImageToTextPredictionDict(TypedDict):
    generated_text: str


ImageToTextPrediction = List[ImageToTextPredictionDict]


class ImageToTextInferenceHandler(AbstractInferenceHandler):
    async def parse_request(
        self, request: Request, temp_dir: str
    ) -> ImageToTextInferenceInput:
        """
        ContentType: multipart/form-data, application/json
        Args:
            - images : input images to be processed. either string(url) or actual image file(through multipart)
            - timeout (float, optional) — The maximum time in seconds to wait for fetching images from the web. If None, no timeout is set and the call may block forever.
            - max_new_tokens (int, optional) — The amount of maximum tokens to generate. By default it will use generate default.
            - generate_kwargs (Dict, optional) — Pass it to send all of these arguments directly to generate allowing full control of this function.
        Returns: List of dict
        Each result comes as a dictionary with the following key:
            - generated_text (str) — The generated text.
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

            return ImageToTextInferenceInput(
                images=images,
                kwargs=dict(
                    timeout=form_field_to_float("timeout", form.get("timeout")),
                    max_new_tokens=form_field_to_int(
                        "max_new_tokens", form.get("max_new_tokens")
                    ),
                    **form_field_to_json(
                        name="kwargs", value=form.get("kwargs"), on_none={}
                    ),
                ),
            )

        elif request.headers["content-type"].startswith("application/json"):
            data = await request.json()
            body = ImageToTextJSONInput.model_validate(data)
            return ImageToTextInferenceInput(
                images=body.images,
                kwargs=dict(
                    timeout=body.timeout,
                    max_new_tokens=body.max_new_tokens,
                    **body.kwargs,
                ),
            )

        else:
            raise HTTPException(status_code=400, detail="Invalid content type")

    # async def infer(
    #     self, data: ImageToTextInferenceInput
    # ) -> Union[ImageToTextPrediction, List[ImageToTextPrediction]]:
    #     return self.pipeline(**data)

    # async def respond(
    #     self, output: Union[ImageToTextPrediction, List[ImageToTextPrediction]]
    # ):
    #     json_compatible_output = jsonable_encoder(output)
    #     return JSONResponse(content=json_compatible_output)
