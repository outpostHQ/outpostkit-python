from typing import Any, Dict, List, Literal, Optional, TypedDict, Union

from fastapi import HTTPException, Request
from generic.tasks.common import AbstractInferenceHandler
from pydantic import BaseModel


class TextClassificationJSONInput(BaseModel):
    texts: Union[
        str,
        List[str],
        Dict[str, Union[str, List[str]]],
        List[Dict[str, Union[str, List[str]]]],
    ]
    top_k: Optional[int] = 1
    function_to_apply: Optional[Literal["sigmoid", "softmax", "none", "default"]] = (
        "default"
    )


class TextClassificationInferenceInput(TypedDict):
    texts: Union[
        str,
        List[str],
        Dict[str, Union[str, List[str]]],
        List[Dict[str, Union[str, List[str]]]],
    ]
    kwargs: Dict[str, Any]


class TextClassificationInference(TypedDict):
    label: str
    score: float


class TextClassificationInferenceHandler(AbstractInferenceHandler):
    async def parse_request(
        self, request: Request, temp_dir: str
    ) -> TextClassificationInferenceInput:
        """
        ContentType: application/json
        Args:
            - texts : One or several texts to classify. In order to use text pairs for your classification, you can send a dictionary containing {"text", "text_pair"} keys, or a list of those.
            - top_k ?: How many results to return.
            - function_to_apply ?: The function to apply to the model outputs in order to retrieve the scores.
        Returns: {texts:Union[str, List[str], Dict[str, Union[str, List[str]]], List[Dict[str, Union[str, List[str]]]]], kwargs:Dict[str, Any]}
        """
        if request.headers["content-type"].startswith("application/json"):
            data = await request.json()
            body = TextClassificationJSONInput.model_validate(data)
            return TextClassificationInferenceInput(
                texts=body.texts,
                kwargs=dict(top_k=body.top_k, function_to_apply=body.function_to_apply),
            )
        else:
            raise HTTPException(status_code=400, detail="Invalid content type")

    # async def infer(
    #     self, data: TextClassificationInferenceInput
    # ) -> Union[
    #     List[TextClassificationInference],
    #     List[List[TextClassificationInference]],
    # ]:
    #     return self.pipeline(**data)

    # async def respond(
    #     self,
    #     output: Union[
    #         List[TextClassificationInference],
    #         List[List[TextClassificationInference]],
    #     ],
    # ):
    #     json_compatible_output = jsonable_encoder(output)
    #     return JSONResponse(content=json_compatible_output)
