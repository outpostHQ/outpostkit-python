from typing import Any, Dict, List, Optional, TypedDict, Union

from fastapi import HTTPException, Request
from generic.tasks.common import AbstractInferenceHandler
from pydantic import BaseModel


class ZeroShotClassificationJSONInput(BaseModel):
    sequences: Union[str, List[str]]
    candidate_labels: Union[str, List[str]]
    multi_label: Optional[bool] = None
    hypothesis_template: Optional[str] = None
    kwargs: Optional[Dict[str, Any]] = {}


class ZeroShotClassificationInferenceInput(TypedDict):
    sequences: Union[str, List[str]]
    candidate_labels: Union[str, List[str]]
    multi_label: Optional[bool]
    hypothesis_template: Optional[str]
    kwargs: Dict[str, Any]


class ZeroShotClassificationPrediction(TypedDict):
    sequence: str
    labels: List[str]
    scores: List[float]


ZeroShotClassificationInference = Union[
    ZeroShotClassificationPrediction, List[ZeroShotClassificationPrediction]
]


class ZeroShotClassificationInferenceHandler(AbstractInferenceHandler):
    async def parse_request(
        self, request: Request, temp_dir: str
    ) -> ZeroShotClassificationInferenceInput:
        if request.headers["content-type"].startswith("application/json"):
            data = await request.json()
            body = ZeroShotClassificationJSONInput(**data)
            return ZeroShotClassificationInferenceInput(
                sequences=body.sequences,
                candidate_labels=body.candidate_labels,
                multi_label=body.multi_label,
                hypothesis_template=body.hypothesis_template,
                kwargs=body.kwargs,
            )
        else:
            raise HTTPException(status_code=400, detail="Invalid content type")

    # async def infer(
    #     self, data: ZeroShotClassificationInferenceInput
    # ) -> ZeroShotClassificationInference:
    #     return self.pipeline(**data)

    # async def respond(
    #     self,
    #     output: ZeroShotClassificationInference,
    # ):
    #     json_compatible_output = jsonable_encoder(output)
    #     return JSONResponse(content=json_compatible_output)
