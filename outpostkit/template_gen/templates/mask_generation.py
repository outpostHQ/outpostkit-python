# ref: https://huggingface.co/docs/transformers/main_classes/pipelines#transformers.pipeline
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


class MaskGenerationJSONInput(BaseModel):
    images: Union[str, List[str]]
    timeout: Optional[float] = None
    mask_threshold: Optional[float] = None
    pred_iou_thresh: Optional[float] = None
    stability_score_thresh: Optional[float] = None
    stability_score_offset: Optional[int] = None
    crops_nms_thresh: Optional[float] = None
    crops_n_layers: Optional[int] = None
    crop_overlap_ratio: Optional[float] = None
    crop_n_points_downscale_factor: Optional[int] = None
    kwargs: Optional[Dict[str, Any]] = {}


class MaskGenerationInferenceInput(TypedDict):
    images: Union[List[str], List[bytes]]
    kwargs: Dict[str, Any]


class MaskGenerationInference(TypedDict):
    mask: Image.Image
    score: Optional[float]


class MaskGenerationInferenceHandler(AbstractInferenceHandler):
    async def parse_request(
        self, request: Request, temp_dir: str
    ) -> MaskGenerationInferenceInput:
        """
        ContentType: multipart/form-data, application/json
        Args:
            - images : Image or list of images.
            - mask_threshold : Threshold to use when turning the predicted masks into binary values.
            - pred_iou_thresh : A filtering threshold in [0,1] applied on the model’s predicted mask quality.
            - stability_score_thresh : A filtering threshold in [0,1], using the stability of the mask under changes to the cutoff used to binarize the model’s mask predictions.
            - stability_score_offset : The amount to shift the cutoff when calculated the stability score.
            - crops_nms_thresh : The box IoU cutoff used by non-maximal suppression to filter duplicate masks.
            - crops_n_layers : If crops_n_layers>0, mask prediction will be run again on crops of the image. Sets the number of layers to run, where each layer has 2**i_layer number of image crops.
            - crop_overlap_ratio : Sets the degree to which crops overlap. In the first crop layer, crops will overlap by this fraction of the image length. Later layers with more crops scale down this overlap.
            - crop_n_points_downscale_factor : The number of points-per-side sampled in layer n is scaled down by crop_n_points_downscale_factor**n.
            - timeout : The maximum time in seconds to wait for fetching images from the web. If None, no timeout is set and the call may block forever.
        Returns: {images:Union[List[str],List[bytes]], kwargs:Dict[str, Any]}
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

            return MaskGenerationInferenceInput(
                images=images,
                kwargs=dict(
                    timeout=form_field_to_float("timeout", form.get("timeout")),
                    mask_threshold=form_field_to_float(
                        "mask_threshold", form.get("mask_threshold")
                    ),
                    pred_iou_thresh=form_field_to_float(
                        "pred_iou_thresh", form.get("pred_iou_thresh")
                    ),
                    stability_score_thresh=form_field_to_float(
                        "stability_score_thresh", form.get("stability_score_thresh")
                    ),
                    stability_score_offset=form_field_to_int(
                        "stability_score_offset", form.get("stability_score_offset")
                    ),
                    crops_nms_thresh=form_field_to_float(
                        "crops_nms_thresh", form.get("crops_nms_thresh")
                    ),
                    crops_n_layers=form_field_to_int(
                        "crops_n_layers", form.get("crops_n_layers")
                    ),
                    crop_overlap_ratio=form_field_to_float(
                        "crop_overlap_ratio", form.get("crop_overlap_ratio")
                    ),
                    crop_n_points_downscale_factor=form_field_to_int(
                        "crop_n_points_downscale_factor",
                        form.get("crop_n_points_downscale_factor"),
                    ),
                    **form_field_to_json(
                        name="kwargs", value=form.get("kwargs"), on_none={}
                    ),
                ),
            )
        elif request.headers["content-type"].startswith("application/json"):
            data = await request.json()
            body = MaskGenerationJSONInput.model_validate(data)
            return MaskGenerationInferenceInput(
                images=body.images,
                kwargs=dict(
                    timeout=body.timeout,
                    mask_threshold=body.mask_threshold,
                    pred_iou_thresh=body.pred_iou_thresh,
                    stability_score_thresh=body.stability_score_thresh,
                    stability_score_offset=body.stability_score_offset,
                    crops_nms_thresh=body.crops_nms_thresh,
                    crops_n_layers=body.crops_n_layers,
                    crop_overlap_ratio=body.crop_overlap_ratio,
                    crop_n_points_downscale_factor=body.crop_n_points_downscale_factor,
                    **body.kwargs,
                ),
            )

        else:
            raise HTTPException(status_code=400, detail="Invalid content type")

    # async def infer(
    #     self, data: MaskGenerationInferenceInput
    # ) -> List[MaskGenerationInference]:
    #     return self.pipeline(**data)

    # async def respond(self, output: List[MaskGenerationInference]):
    #     json_compatible_output = jsonable_encoder(output)
    #     return JSONResponse(content=json_compatible_output)
