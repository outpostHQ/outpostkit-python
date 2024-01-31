from typing import Any, Dict, List, Optional, TypedDict, Union

from fastapi import HTTPException, Request
from generic.tasks.common import AbstractInferenceHandler
from pydantic import BaseModel


class SummarizationJSONInput(BaseModel):
    documents: Union[str, List[str]]
    return_text: Optional[bool] = None
    return_tensors: Optional[bool] = None
    clean_up_tokenization_spaces: Optional[bool] = None
    kwargs: Optional[Dict[str, Any]] = {}


class SummarizationInferenceInput(TypedDict):
    documents: Union[str, List[str]]
    return_text: Optional[bool]
    return_tensors: Optional[bool]
    clean_up_tokenization_spaces: Optional[bool]
    kwargs: Dict[str, str]


class SummarizationInferenceOutput(TypedDict):
    summary_text: Optional[str]
    summary_token_ids: Optional[Any]  # Union['torch.Tensor', 'tf.Tensor']


class SummarizationInferenceHandler(AbstractInferenceHandler):
    async def parse_request(
        self, request: Request, temp_dir: str
    ) -> SummarizationInferenceInput:
        """
        ContentType: application/json
        Args:
            - documents : One or several articles (or one list of articles) to summarize.
            - return_text ?: Whether or not to include the decoded texts in the outputs.
            - return_tensors ?: Whether or not to include the tensors of predictions (as token indices) in the outputs.
            - clean_up_tokenization_spaces ?: Whether or not to clean up the potential extra spaces in the text output.
            - kwargs ?: Additional keyword arguments to pass along to the generate method of the model.
        Returns: SummarizationInferenceInput
        """
        if request.headers["content-type"].startswith("application/json"):
            data = await request.json()
            body = SummarizationJSONInput.model_validate(data)
            return SummarizationInferenceInput(
                documents=body.documents,
                kwargs=dict(
                    return_text=body.return_text,
                    return_tensors=body.return_tensors,
                    clean_up_tokenization_spaces=body.clean_up_tokenization_spaces,
                    **body.kwargs,
                ),
            )
        else:
            raise HTTPException(status_code=400, detail="Invalid content type")

    # async def infer(
    #     self, data: SummarizationInferenceInput
    # ) -> Union[
    #     List[SummarizationInferenceOutput], List[List[SummarizationInferenceOutput]]
    # ]:
    #     return self.pipeline(**data)

    # async def respond(
    #     self,
    #     output: Union[
    #         List[SummarizationInferenceOutput], List[List[SummarizationInferenceOutput]]
    #     ],
    # ):
    #     json_compatible_output = jsonable_encoder(output)
    #     return JSONResponse(content=json_compatible_output)
