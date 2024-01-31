import os
import time
from typing import Any, Dict, List, Optional, TypedDict, Union

from fastapi import HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from generic.tasks.common import AbstractInferenceHandler
from generic.tasks.utils import (
    form_field_to_json,
    form_field_to_str,
)
from pydantic import BaseModel
from starlette.datastructures import UploadFile


class ZeroShotAudioClassificationJSONInput(BaseModel):
    audios: Union[str, List[str]]
    candidate_labels: Union[str, List[str]]
    hypothesis_template: Optional[str] = None
    kwargs: Optional[Dict[str, Any]] = {}


class ZeroShotAudioClassificationInferenceInput(TypedDict):
    audios: List[str]
    candidate_labels: List[str]
    hypothesis_template: Optional[str]
    kwargs: Dict[str, Any]


class ZeroShotAudioClassificationPrediction(TypedDict):
    labels: List[str]
    scores: List[float]


ZeroShotAudioClassificationInference = Union[
    ZeroShotAudioClassificationPrediction, List[ZeroShotAudioClassificationPrediction]
]


class ZeroShotAudioClassificationInferenceHandler(AbstractInferenceHandler):
    async def parse_request(
        self, request: Request, temp_dir: str
    ) -> ZeroShotAudioClassificationInferenceInput:
        """
        ContentType: multipart/form-data, application/json
        Args:
            - audios : The pipeline handles three types of inputs:
                - A string containing a http link pointing to an audio
                - A string containing a local path to an audio
                - An audio loaded in numpy
            - candidate_labels : The candidate labels for this audio
            - hypothesis_template : The sentence used in cunjunction with candidate_labels to attempt the audio classification by replacing the placeholder with the candidate_labels. Then likelihood is estimated by using logits_per_audio
        Returns: {audios:List[str],candidate_labels:List[str],hypothesis_template:Optional[str],kwargs:Dict[str,Any]}
        """
        if request.headers["content-type"].startswith("multipart/form-data"):
            form = await request.form()
            raw_audios = form.getlist("audios")
            candidate_labels = form_field_to_str(
                "candidate_labels", form.get("candidate_labels"), mandatory=True
            ).split(",")
            hypothesis_template = form_field_to_str(
                "hypothesis_template", form.get("hypothesis_template")
            )

            try:
                audios: List[str] = []
                for file in raw_audios:
                    if isinstance(file, UploadFile):
                        unique_filename = f"{int(time.time())}_{file.filename}"
                        temp_file = os.path.join(temp_dir, unique_filename)
                        with open(temp_file, "wb+") as fp:
                            fp.write(await file.read())
                        audios.append(temp_file)
                    elif isinstance(file, str):
                        audios.append(file)

                return ZeroShotAudioClassificationInferenceInput(
                    audios=audios,
                    candidate_labels=candidate_labels,
                    hypothesis_template=hypothesis_template,
                    kwargs=form_field_to_json(
                        name="kwargs", value=form.get("kwargs"), on_none={}
                    ),
                )
            except Exception as e:
                raise HTTPException(status_code=400, detail=str(e)) from Exception

        elif request.headers["content-type"].startswith("application/json"):
            data = await request.json()
            body = ZeroShotAudioClassificationJSONInput.model_validate(data)
            return ZeroShotAudioClassificationInferenceInput(
                audios=body.audios,
                candidate_labels=body.candidate_labels,
                hypothesis_template=body.hypothesis_template,
                kwargs=body.kwargs,
            )

        else:
            raise HTTPException(status_code=400, detail="Invalid content type")

    async def infer(
        self, data: ZeroShotAudioClassificationInferenceInput
    ) -> ZeroShotAudioClassificationInference:
        return self.pipeline(**data)

    async def respond(self, output: ZeroShotAudioClassificationInference):
        json_compatible_output = jsonable_encoder(output)
        return JSONResponse(content=json_compatible_output)
