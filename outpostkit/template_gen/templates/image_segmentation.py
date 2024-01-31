# ref: https://huggingface.co/docs/transformers/v4.36.1/en/main_classes/pipelines#transformers.ImageSegmentationPipeline

from io import BytesIO
from typing import Any, Dict, List, Optional, TypedDict, Union

from fastapi import HTTPException, Request
from generic.tasks.common import AbstractInferenceHandler
from generic.tasks.utils import (
    form_field_to_float,
    form_field_to_json,
    form_field_to_str,
)
from PIL import Image
from pydantic import BaseModel
from starlette.datastructures import UploadFile


class ImageSegmentationJSONInput(BaseModel):
    images: Union[str, List[str]]
    subtask: Optional[str] = None
    timeout: Optional[float] = None
    threshold: Optional[float] = None
    mask_threshold: Optional[float] = None
    overlap_mask_area_threshold: Optional[float] = None
    kwargs: Optional[Dict[str, Any]] = {}


class ImageSegmentationInferenceInput(TypedDict):
    images: Union[str, List[str], Image.Image, List[Image.Image]]
    kwargs: Dict[str, Any]


ImageSegmentationPrediction = Dict[str, Any]


class ImageSegmentationInferenceHandler(AbstractInferenceHandler):
    async def parse_request(
        self, request: Request, temp_dir: str
    ) -> ImageSegmentationInferenceInput:
        """
        ContentType: multipart/form-data, application/json
        Args:
            - images : input images to be processed. either string(url) or actual image file(through multipart)
            - subtask ?: Subtask for image segmentation.
            - timeout ?: The maximum time in seconds to wait for fetching images from the web.
            - threshold ?: The probability necessary to make a prediction.
            - mask_threshold ?: Threshold for mask.
            - overlap_mask_area_threshold ?: Threshold for overlap mask area.
        Returns: {images:List[Union[str,bytes]], kwargs:Dict[str, Any]}
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
            else:
                raise HTTPException(
                    status_code=400,
                    detail="Invalid images field. please send either a list of urls or a list of image files.",
                )

            return ImageSegmentationInferenceInput(
                images=images,
                kwargs=dict(
                    timeout=form_field_to_float("timeout", form.get("timeout")),
                    subtask=form_field_to_str("subtask", form.get("subtask")),
                    threshold=form_field_to_float("threshold", form.get("threshold")),
                    mask_threshold=form_field_to_float(
                        "mask_threshold", form.get("mask_threshold")
                    ),
                    overlap_mask_area_threshold=form_field_to_float(
                        "overlap_mask_area_threshold",
                        form.get("overlap_mask_area_threshold"),
                    ),
                    **form_field_to_json(
                        name="kwargs", value=form.get("kwargs"), on_none={}
                    ),
                ),
            )

        elif request.headers["content-type"].startswith("application/json"):
            data = await request.json()
            body = ImageSegmentationJSONInput.model_validate(data)
            return ImageSegmentationInferenceInput(
                images=body.images,
                kwargs=dict(
                    subtask=body.subtask,
                    timeout=body.timeout,
                    threshold=body.threshold,
                    mask_threshold=body.mask_threshold,
                    overlap_mask_area_threshold=body.overlap_mask_area_threshold,
                    **body.kwargs,
                ),
            )

        else:
            raise HTTPException(status_code=400, detail="Invalid content type")

    # async def infer(
    #     self, data: ImageSegmentationInferenceInput
    # ) -> Union[ImageSegmentationPrediction, List[ImageSegmentationPrediction]]:
    #     return self.pipeline(**data)

    # async def respond(
    #     self,
    #     output: Union[ImageSegmentationPrediction, List[ImageSegmentationPrediction]],
    # ):
    #     json_compatible_output = jsonable_encoder(output)
    #     return JSONResponse(content=json_compatible_output)
