from typing import Any, Dict, List, Optional, TypedDict, Union

from fastapi import HTTPException, Request
from generic.tasks.common import AbstractInferenceHandler
from pydantic import BaseModel


class Text2TextJSONInput(BaseModel):
    args: Union[str, List[str]]
    return_tensors: Optional[bool] = None
    return_text: Optional[bool] = None
    clean_up_tokenization_spaces: Optional[bool] = None
    truncation: Optional[str] = None
    kwargs: Optional[Dict[str, Any]] = {}


class Text2TextInferenceInput(TypedDict):
    args: Union[str, List[str]]
    kwargs: Dict[str, Any]


class Text2TextInference(TypedDict):
    generated_text: Optional[str]
    generated_token_ids: Optional[Any]  # Union['torch.Tensor', 'tf.Tensor']


class Text2TextInferenceHandler(AbstractInferenceHandler):
    async def parse_request(
        self, request: Request, temp_dir: str
    ) -> Text2TextInferenceInput:
        """
        ContentType: application/json
        Args:
            - args : Input text for the encoder.
            - return_tensors ?: Whether or not to include the tensors of predictions (as token indices) in the outputs.
            - return_text ?: Whether or not to include the decoded texts in the outputs.
            - clean_up_tokenization_spaces ?: Whether or not to clean up the potential extra spaces in the text output.
            - truncation ?: The truncation strategy for the tokenization within the pipeline.
            - kwargs ?: Additional keyword arguments to pass along to the generate method of the model.
        Returns: {args:Union[str, List[str]], kwargs:Dict[str, Any]}
        """
        if request.headers["content-type"].startswith("application/json"):
            data = await request.json()
            body = Text2TextJSONInput.model_validate(data)
            return Text2TextInferenceInput(
                args=body.args,
                kwargs=dict(
                    return_tensors=body.return_tensors,
                    return_text=body.return_text,
                    clean_up_tokenization_spaces=body.clean_up_tokenization_spaces,
                    truncation=body.truncation,
                    **body.kwargs,
                ),
            )
        else:
            raise HTTPException(status_code=400, detail="Invalid content type")

    # async def infer(
    #     self, data: Text2TextInferenceInput
    # ) -> Union[List[List[Text2TextInference]], List[Text2TextInference]]:
    #     return self.pipeline(**data)

    # async def respond(
    #     self, output: Union[List[List[Text2TextInference]], List[Text2TextInference]]
    # ):
    #     json_compatible_output = jsonable_encoder(output)
    #     return JSONResponse(content=json_compatible_output)
