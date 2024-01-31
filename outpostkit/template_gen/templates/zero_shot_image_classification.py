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


class ZeroShotImageClassificationJSONInput(BaseModel):
    images: Union[str, List[str]]
    candidate_labels: List[str]
    hypothesis_template: Optional[str] = None
    timeout: Optional[float] = None
    kwargs: Optional[Dict[str, Any]] = {}


class ZeroShotImageClassificationInferenceInput(TypedDict):
    images: Union[str, List[str], Image.Image, List[Image.Image]]
    kwargs: Dict[str, Any]


class ZeroShotImageClassificationInference(TypedDict):
    label: str
    score: float


class ZeroShotImageClassificationInferenceHandler(AbstractInferenceHandler):
    async def parse_request(
        self, request: Request, temp_dir: str
    ) -> ZeroShotImageClassificationInferenceInput:
        """
        ContentType: multipart/form-data, application/json
        Args:
            - images : The pipeline handles three types of images:
                A string containing a http link pointing to an image
                A string containing a local path to an image
                An image loaded in PIL directly
            - candidate_labels : The candidate labels for this image
            - hypothesis_template : The sentence used in cunjunction with candidate_labels to attempt the image classification by replacing the placeholder with the candidate_labels. Then likelihood is estimated by using logits_per_image
            - timeout : The maximum time in seconds to wait for fetching images from the web. If None, no timeout is set and the call may block forever.
        Returns: {images:Union[str,bytes], kwargs:Dict[str, Any]}
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

            return ZeroShotImageClassificationInferenceInput(
                images=images,
                kwargs=dict(
                    candidate_labels=form_field_to_str(
                        "candidate_labels", form.get("candidate_labels"), True
                    ).split(","),
                    hypothesis_template=form_field_to_str(
                        "hypothesis_template", form.get("hypothesis_template")
                    ),
                    timeout=form_field_to_float("timeout", form.get("timeout")),
                    **form_field_to_json(
                        name="kwargs", value=form.get("kwargs"), on_none={}
                    ),
                ),
            )

        elif request.headers["content-type"].startswith("application/json"):
            data = await request.json()
            body = ZeroShotImageClassificationJSONInput.model_validate(data)
            return ZeroShotImageClassificationInferenceInput(
                images=body.images,
                kwargs=dict(
                    candidate_labels=body.candidate_labels,
                    hypothesis_template=body.hypothesis_template,
                    timeout=body.timeout,
                    **body.kwargs,
                ),
            )

        else:
            raise HTTPException(status_code=400, detail="Invalid content type")

    # async def infer(
    #     self, data: ZeroShotImageClassificationInferenceInput
    # ) -> Union[
    #     ZeroShotImageClassificationInference, List[ZeroShotImageClassificationInference]
    # ]:
    #     return self.pipeline(**data)

    # async def respond(
    #     self,
    #     output: Union[
    #         ZeroShotImageClassificationInference,
    #         List[ZeroShotImageClassificationInference],
    #     ],
    # ):
    #     json_compatible_output = jsonable_encoder(output)
    #     return JSONResponse(content=json_compatible_output)
