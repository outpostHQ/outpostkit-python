from typing import Any, Dict, List, Optional, TypedDict, Union

from fastapi import Request
from generic.tasks.common import AbstractInferenceHandler
from pydantic import BaseModel


class TokenClassificationJSONInput(BaseModel):
    inputs: Union[str, List[str]]
    kwargs: Optional[Dict[str, Any]] = {}


class TokenClassificationInferenceInput(TypedDict):
    inputs: Union[str, List[str]]
    kwargs: Dict[str, Any]


class TokenClassificationInference(TypedDict):
    word: str
    score: float
    entity: str
    index: Optional[int]
    start: Optional[int]
    end: Optional[int]


class TokenClassificationInferenceHandler(AbstractInferenceHandler):
    async def parse_request(
        self, request: Request, temp_dir: str
    ) -> TokenClassificationInferenceInput:
        """
        ContentType: application/json
        Args:
            - inputs (`str` or `List[str]`): One or several texts (or one list of texts) for token classification.
        Returns: {inputs:Union[str, List[str]], kwargs:Optional[Dict[str, Any]]}
        """
        data = await request.json()
        body = TokenClassificationJSONInput.model_validate(data)
        return TokenClassificationInferenceInput(inputs=body.inputs, kwargs=body.kwargs)

    # async def infer(
    #     self, data: TokenClassificationInferenceInput
    # ) -> Union[
    #     List[List[TokenClassificationInference]], List[TokenClassificationInference]
    # ]:
    #     return self.pipeline(**data)

    # async def respond(
    #     self,
    #     output: Union[
    #         List[List[TokenClassificationInference]], List[TokenClassificationInference]
    #     ],
    # ):
    #     json_compatible_output = jsonable_encoder(output)
    #     return JSONResponse(content=json_compatible_output)
