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


class DepthEstimationJSONInput(BaseModel):
    images: Union[str, List[str]]
    top_k: Optional[int] = None
    timeout: Optional[float] = None
    kwargs: Optional[Dict[str, Any]] = {}


class DepthEstimationInferenceInput(TypedDict):
    images: Union[str, List[str], Image.Image, List[Image.Image]]
    kwargs: Dict[str, Any]


DepthEstimationPrediction = Dict


class DepthEstimationInferenceHandler(AbstractInferenceHandler):
    async def parse_request(
        self, request: Request, temp_dir: str
    ) -> DepthEstimationInferenceInput:
        """
        ContentType: multipart/form-data, application/json
        Args:
            - images : input images to be processed. either string(url) or actual image file(through multipart)
            - topk ?: The number of top labels that will be returned by the pipeline.
            - timeout ?: timeout for the request.
        Returns: {images:List[Union[str,bytes]], topk:Optional[None], timeout:Optional[None]}
        """
        if request.headers["content-type"].startswith("multipart/form-data"):
            form = await request.form()
            raw_images = form.getlist("images")
            top_k = form_field_to_int("top_k", form.get("top_k"))
            timeout = form_field_to_float("timeout", form.get("timeout"))
            kwargs = form_field_to_json(
                name="kwargs", value=form.get("kwargs"), on_none={}
            )
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
            return DepthEstimationInferenceInput(
                images=images, kwargs=dict(top_k=top_k, timeout=timeout, **kwargs)
            )

        elif request.headers["content-type"].startswith("application/json"):
            data = await request.json()
            body = DepthEstimationJSONInput.model_validate(data)
            if len(body.images) == 0:
                raise HTTPException(status_code=400, detail="No images provided.")
            return DepthEstimationInferenceInput(
                images=body.images,
                kwargs=dict(top_k=body.top_k, timeout=body.timeout, **body.kwargs),
            )

        else:
            raise HTTPException(status_code=400, detail="Invalid content type")

    # async def infer(
    #     self, data: DepthEstimationInferenceInput
    # ) -> Union[DepthEstimationPrediction, List[DepthEstimationPrediction]]:
    #     return self.pipeline(**data)

    # async def respond(
    #     self, output: Union[DepthEstimationPrediction, List[DepthEstimationPrediction]]
    # ):
    #     # if(isinstance(output,List)):
    #     # for op in output:
    #     print(output)
    #     json_compatible_output = jsonable_encoder(
    #         output, custom_encoder=custom_encoders
    #     )
    #     return JSONResponse(content=json_compatible_output)
