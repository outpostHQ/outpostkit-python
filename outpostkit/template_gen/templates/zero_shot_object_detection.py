from io import BytesIO
from typing import Any, Dict, List, Optional, TypedDict, Union

from fastapi import HTTPException, Request
from generic.tasks.common import AbstractInferenceHandler
from generic.tasks.utils import (
    form_field_to_float,
    form_field_to_int,
    form_field_to_json,
    form_field_to_str,
)
from PIL import Image
from pydantic import BaseModel
from starlette.datastructures import UploadFile


class ZeroShotObjectDetectionJSONInput(BaseModel):
    images: Union[str, List[str]]
    candidate_labels: Union[str, List[str], List[List[str]]]
    threshold: Optional[float] = None
    top_k: Optional[int] = None
    timeout: Optional[float] = None
    kwargs: Optional[Dict[str, Any]] = {}


class ZeroShotObjectDetectionInferenceInput(TypedDict):
    images: Union[str, List[str], Image.Image, List[Image.Image]]
    kwargs: Dict[str, Any]


class ZeroShotObjectDetectionInference(TypedDict):
    label: str
    score: float
    box: Dict[str, int]


class ZeroShotObjectDetectionInferenceHandler(AbstractInferenceHandler):
    async def parse_request(
        self, request: Request, temp_dir: str
    ) -> ZeroShotObjectDetectionInferenceInput:
        """
        ContentType: multipart/form-data, application/json
        Args:
            - images : The pipeline handles three types of images:
                A string containing a http link pointing to an image
                A string containing a local path to an image
                An image loaded in PIL directly
            - candidate_labels : What the model should recognize in the image.
            - threshold : The probability necessary to make a prediction.
            - top_k : The number of top predictions that will be returned by the pipeline. If the provided number is `None` or higher than the number of predictions available, it will default to the number of predictions.
            - timeout : The maximum time in seconds to wait for fetching images from the web. If None, no timeout is set and the call may block forever.
        Returns: {images:Union[str,bytes], kwargs:Dict[str, Any]}
        """
        if request.headers["content-type"].startswith("multipart/form-data"):
            form = await request.form()
            timeout = form_field_to_float("timeout", form.get("timeout"))
            threshold = form_field_to_float("threshold", form.get("threshold"))
            top_k = form_field_to_int("top_k", form.get("top_k"))
            candidate_labels = form_field_to_str(
                "candidate_labels", form.get("candidate_labels"), mandatory=True
            )

            try:
                candidate_labels = candidate_labels.split(",")
            except ValueError:
                raise HTTPException(
                    status_code=400, detail="Unable to parse candidate_labels"
                ) from ValueError

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

            return ZeroShotObjectDetectionInferenceInput(
                images=images,
                kwargs=dict(
                    candidate_labels=candidate_labels,
                    threshold=threshold,
                    top_k=top_k,
                    timeout=timeout,
                    **form_field_to_json(
                        name="kwargs", value=form.get("kwargs"), on_none={}
                    ),
                ),
            )

        elif request.headers["content-type"].startswith("application/json"):
            data = await request.json()
            body = ZeroShotObjectDetectionJSONInput.model_validate(data)
            return ZeroShotObjectDetectionInferenceInput(
                images=body.images,
                kwargs=dict(
                    candidate_labels=body.candidate_labels,
                    threshold=body.threshold,
                    top_k=body.top_k,
                    timeout=body.timeout,
                    **body.kwargs,
                ),
            )

        else:
            raise HTTPException(status_code=400, detail="Invalid content type")

    # async def infer(
    #     self, data: ZeroShotObjectDetectionInferenceInput
    # ) -> Union[
    #     ZeroShotObjectDetectionInference,
    #     List[ZeroShotObjectDetectionInference],
    # ]:
    #     return self.pipeline(**data)

    # async def respond(
    #     self,
    #     output: Union[
    #         ZeroShotObjectDetectionInference,
    #         List[ZeroShotObjectDetectionInference],
    #     ],
    # ):
    #     json_compatible_output = jsonable_encoder(output)
    #     return JSONResponse(content=json_compatible_output)
