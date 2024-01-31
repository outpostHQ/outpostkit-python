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


class ImageToImageJSONInput(BaseModel):
    images: Union[str, List[str]]
    timeout: Optional[float] = None
    kwargs: Optional[Dict[str, Any]] = {}


class ImageToImageInferenceInput(TypedDict):
    images: Union[str, List[str], Image.Image, List[Image.Image]]
    kwargs: Dict[str, Any]


ImageToImagePrediction = Image.Image


class ImageToImageInferenceHandler(AbstractInferenceHandler):
    async def parse_request(
        self, request: Request, temp_dir: str
    ) -> ImageToImageInferenceInput:
        """
        ContentType: multipart/form-data, application/json
        Args:
            - images : input to be transformed. either string(url) or actual image file(through multipart);
            - timeout ?: The maximum time in seconds toxwait for fetching images from the web.
        Returns: {images:Union[List[str],List[bytes]], kwargs:Optional[None]}
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

            return ImageToImageInferenceInput(
                images=images,
                kwargs=dict(
                    timeout=form_field_to_float("timeout", form.get("timeout")),
                    **form_field_to_json(
                        name="kwargs", value=form.get("kwargs"), on_none={}
                    ),
                ),
            )

        elif request.headers["content-type"].startswith("application/json"):
            data = await request.json()
            body = ImageToImageJSONInput.model_validate(data)
            return ImageToImageInferenceInput(
                images=body.images, kwargs=dict(timeout=body.timeout, **body.kwargs)
            )

        else:
            raise HTTPException(status_code=400, detail="Invalid content type")

    # async def infer(
    #     self, data
    # ) -> Union[ImageToImagePrediction, List[ImageToImagePrediction]]:
    #     return self.pipeline(**data)

    # async def respond(
    #     self, output: Union[ImageToImagePrediction, List[ImageToImagePrediction]]
    # ):
    #     """
    #     Returns: Union[ImageToImagePrediction,List[ImageToImagePrediction]]
    #     """
    #     im_bytes: Union[str, map[str]]
    #     if isinstance(output, ImageToImagePrediction):
    #         im_bytes = [pil_to_b64(output)]
    #     else:
    #         im_bytes = map(pil_to_b64, output)
    #     json_compatible_output = jsonable_encoder({"images": im_bytes})
    #     return JSONResponse(content=json_compatible_output)
    # try:
    #     with TemporaryDirectory() as temp_dir:
    #         images: List[str] = []s
    #         for file in raw_images:
    #             if isinstance(file, UploadFile):
    #                 unique_filename = f"{int(time.time())}_{file.filename}"
    #                 temp_file = os.path.join(temp_dir, unique_filename)
    #                 with open(temp_file, "wb+") as fp:
    #                     fp.write(await file.read())
    #                 images.append(temp_file)
    #             elif isinstance(file, str):
    #                 images.append(file)

    #         return {"images": images}
    # except Exception as e:
    #     raise HTTPException(status_code=400, detail=str(e))
