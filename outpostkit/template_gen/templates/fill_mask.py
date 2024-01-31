# ref: https://huggingface.co/docs/transformers/v4.36.1/en/main_classes/pipelines#transformers.FillMaskPipeline

from typing import Any, Dict, List, Optional, TypedDict, Union

from fastapi import HTTPException, Request
from generic.tasks.common import AbstractInferenceHandler
from pydantic import BaseModel


class FillMaskJSONInput(BaseModel):
    args: Union[str, List[str]]
    targets: Optional[Union[str, List[str]]] = None
    top_k: Optional[int] = None
    kwargs: Optional[Dict[str, Any]] = {}


class FillMaskInferenceInput(TypedDict):
    args: Union[str, List[str]]
    kwargs: Dict[str, Any]


class FillMaskInference(TypedDict):
    sequence: str
    score: float
    token: int
    token_str: str


class FillMaskInferenceHandler(AbstractInferenceHandler):
    async def parse_request(
        self, request: Request, temp_dir: str
    ) -> FillMaskInferenceInput:
        """
        ContentType: application/json
        Args:
            - args : One or several texts (or one list of prompts) with masked tokens.
            - targets ?: When passed, the model will limit the scores to the passed targets instead of looking up in the whole vocab.
            - top_k ?: When passed, overrides the number of predictions to return.
        Returns: {args:Union[str, List[str]], kwargs:Dict[str, Any]}
        """
        if request.headers["content-type"].startswith("application/json"):
            data = await request.json()
            body = FillMaskJSONInput.model_validate(data)
            return FillMaskInferenceInput(
                args=body.args,
                kwargs=dict(targets=body.targets, top_k=body.top_k, **body.kwargs),
            )

        else:
            raise HTTPException(status_code=400, detail="Invalid content type")

    # async def infer(self, data) -> List[FillMaskInference]:
    #     """
    #     Returns: List[Dict[{sequence:str, score:float, token:int, token_str:str}]]
    #     """
    #     return self.pipeline(**data)

    # async def respond(self, output: List[FillMaskInference]):
    #     json_compatible_output = jsonable_encoder(output)
    #     return JSONResponse(content=json_compatible_output)
