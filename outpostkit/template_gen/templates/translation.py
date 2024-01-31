from typing import Any, Dict, List, Optional, TypedDict, Union

from fastapi import HTTPException, Request
from generic.tasks.common import AbstractInferenceHandler
from pydantic import BaseModel


class TranslationJSONInput(BaseModel):
    texts: Union[str, List[str]]
    return_tensors: Optional[bool] = None
    return_text: Optional[bool] = None
    clean_up_tokenization_spaces: Optional[bool] = None
    src_lang: Optional[str] = None
    tgt_lang: Optional[str] = None
    kwargs: Optional[Dict[str, Any]] = {}


class TranslationInferenceInput(TypedDict):
    texts: Union[str, List[str]]
    kwargs: Dict[str, Any]


class TranslationInference(TypedDict):
    generated_text: Optional[str]
    generated_token_ids: Optional[Any]  # Union["torch.Tensor", "tf.Tensor"]


class TranslationInferenceHandler(AbstractInferenceHandler):
    async def parse_request(
        self, request: Request, temp_dir: str
    ) -> TranslationInferenceInput:
        """
        ContentType: application/json
        Args:
            - texts : Texts to be translated. either single string or list of strings
            - return_tensors : Whether or not to include the tensors of predictions (as token indices) in the outputs.
            - return_text : Whether or not to include the decoded texts in the outputs.
            - clean_up_tokenization_spaces : Whether or not to clean up the potential extra spaces in the text output.
            - src_lang : The language of the input. Might be required for multilingual models. Will not have any effect for single pair translation models
            - tgt_lang : The language of the desired output. Might be required for multilingual models. Will not have any effect for single pair translation models
            - generate_kwargs : Additional keyword arguments to pass along to the generate method of the model.
        Returns: {texts:Union[str, List[str]], kwargs:Dict[str, Any]}
        """
        if request.headers["content-type"].startswith("application/json"):
            data = await request.json()
            body = TranslationJSONInput.model_validate(data)
            return TranslationInferenceInput(
                texts=body.texts,
                kwargs=dict(
                    return_tensors=body.return_tensors,
                    return_text=body.return_text,
                    clean_up_tokenization_spaces=body.clean_up_tokenization_spaces,
                    src_lang=body.src_lang,
                    tgt_lang=body.tgt_lang,
                    **body.kwargs,
                ),
            )
        else:
            raise HTTPException(status_code=400, detail="Invalid content type")

    # async def infer(
    #     self, data: TranslationInferenceInput
    # ) -> Union[List[List[TranslationInference]], List[TranslationInference]]:
    #     return self.pipeline(**data)

    # async def respond(
    #     self,
    #     output: Union[List[List[TranslationInference]], List[TranslationInference]],
    # ):
    #     json_compatible_output = jsonable_encoder(output)
    #     return JSONResponse(content=json_compatible_output)
