import os
import time
from typing import Any, Dict, List, Optional, TypedDict, Union

from fastapi import HTTPException, Request
from generic.tasks.common import AbstractInferenceHandler
from generic.tasks.utils import (
    form_field_to_int,
    form_field_to_json,
)
from pydantic import BaseModel
from starlette.datastructures import UploadFile


class VideoClassificationJSONInput(BaseModel):
    videos: Union[str, List[str]]
    top_k: Optional[int] = None
    num_frames: Optional[int] = None
    frame_sampling_rate: Optional[int] = None
    kwargs: Optional[Dict[str, Any]] = {}


class VideoClassificationInferenceInput(TypedDict):
    videos: List[str]
    kwargs: Dict[str, Any]


class VideoClassificationInference(TypedDict):
    label: str
    score: float


class VideoClassificationInferenceHandler(AbstractInferenceHandler):
    async def parse_request(
        self, request: Request, temp_dir: str
    ) -> VideoClassificationInferenceInput:
        """
        ContentType: multipart/form-data, application/json
        Args:
            - videos : input to be classified. either string(url) or actual video file(through multipart);
            - top_k ?: The number of top labels that will be returned by the pipeline.
            - num_frames ?: The number of frames sampled from the video to run the classification on. If not provided, will default to the number of frames specified in the model configuration.
            - frame_sampling_rate ?: The sampling rate used to select frames from the video. If not provided, will default to 1, i.e. every frame will be used.
        Returns: {videos:List[str], kwargs:Dict[str, Any]}
        """
        if request.headers["content-type"].startswith("multipart/form-data"):
            form = await request.form()
            raw_videos = form.getlist("videos")

            try:
                videos = []
                for file in raw_videos:
                    if isinstance(file, UploadFile):
                        unique_filename = f"{int(time.time())}_{file.filename}"
                        temp_file = os.path.join(temp_dir, unique_filename)
                        with open(temp_file, "wb+") as fp:
                            fp.write(await file.read())
                        videos.append(temp_file)
                    elif isinstance(file, str):
                        videos.append(file)

                return VideoClassificationInferenceInput(
                    videos=videos,
                    kwargs=dict(
                        top_k=form_field_to_int("top_k", form.get("top_k")),
                        num_frames=form_field_to_int(
                            "num_frames", form.get("num_frames")
                        ),
                        frame_sampling_rate=form_field_to_int(
                            "frame_sampling_rate", form.get("frame_sampling_rate")
                        ),
                        **form_field_to_json(
                            name="kwargs", value=form.get("kwargs"), on_none={}
                        ),
                    ),
                )
            except Exception as e:
                raise HTTPException(status_code=400, detail=str(e))

        elif request.headers["content-type"].startswith("application/json"):
            data = await request.json()
            body = VideoClassificationJSONInput.model_validate(data)
            return VideoClassificationInferenceInput(
                videos=body.videos,
                kwargs=dict(
                    top_k=body.top_k,
                    num_frames=body.num_frames,
                    frame_sampling_rate=body.frame_sampling_rate,
                    **body.kwargs,
                ),
            )

        else:
            raise HTTPException(status_code=400, detail="Invalid content type")


#     async def infer(
#         self, data: VideoClassificationInferenceInput
#     ) -> Union[VideoClassificationInference, List[VideoClassificationInference]]:
#         return self.pipeline(**data)

#     async def respond(
#         self,
#         output: Union[VideoClassificationInference, List[VideoClassificationInference]],
#     ):
#         json_compatible_output = jsonable_encoder(output)
#         return JSONResponse(content=json_compatible_output)
