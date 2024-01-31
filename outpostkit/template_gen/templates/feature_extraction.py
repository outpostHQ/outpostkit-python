# ref: https://huggingface.co/docs/transformers/v4.36.1/en/main_classes/pipelines#transformers.FeatureExtractionPipeline

from typing import Any, Dict, List, Optional, TypedDict, Union

from fastapi import HTTPException, Request
from generic.tasks.common import AbstractInferenceHandler
from pydantic import BaseModel


class FeatureExtractionJSONInput(BaseModel):
    args: Union[str, List[str]]
    kwargs: Optional[Dict[str, Any]] = {}


class FeatureExtractionInferenceInput(TypedDict):
    args: Union[str, List[str]]
    kwargs: Dict[str, Any]


class FeatureExtractionInferenceHandler(AbstractInferenceHandler):
    async def parse_request(
        self, request: Request, temp_dir: str
    ) -> FeatureExtractionInferenceInput:
        """
        ContentType: application/json
        Args:
            - args : One or several texts (or one list of texts) to get the features of.
        Returns: {args:Union[str, List[str]], kwargs:Dict[str, Any]}
        """
        if request.headers["content-type"].startswith("application/json"):
            data = await request.json()
            body = FeatureExtractionJSONInput.model_validate(data)
            return FeatureExtractionInferenceInput(args=body.args, kwargs=body.kwargs)

        else:
            raise HTTPException(status_code=400, detail="Invalid content type")

    # async def infer(self, data) -> List[List[float]]:
    #     """
    #     Returns: List[List[float]] - The features computed by the model.
    #     """
    #     return self.pipeline(**data)

    # async def respond(self, output: List[List[float]]):
    #     json_compatible_output = jsonable_encoder(output)
    #     return JSONResponse(content=json_compatible_output)
