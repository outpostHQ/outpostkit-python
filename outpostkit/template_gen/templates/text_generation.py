from typing import Any, Dict, List, Optional, TypedDict, Union

from fastapi import HTTPException, Request
from generic.tasks.common import AbstractInferenceHandler
from generic.tasks.utils import (
    form_field_to_int,
    form_field_to_json,
)
from pydantic import BaseModel


class TextGenerationJSONInput(BaseModel):
    prompts: Union[str, List[str]]
    return_tensors: Optional[bool] = None
    return_text: Optional[bool] = None
    return_full_text: Optional[bool] = None
    clean_up_tokenization_spaces: Optional[bool] = None
    prefix: Optional[str] = None
    handle_long_generation: Optional[str] = None
    kwargs: Optional[Dict[str, Any]] = {}


class TextGenerationInferenceInput(TypedDict):
    prompts: Union[str, List[str]]
    kwargs: Dict[str, Any]


class TextGenerationInference(TypedDict):
    generated_text: Optional[str]
    generated_token_ids: Optional[Any]  # Union['torch.Tensor', 'tf.Tensor']


class TextGenerationInferenceHandler(AbstractInferenceHandler):
    async def parse_request(
        self, request: Request, temp_dir: str
    ) -> TextGenerationInferenceInput:
        """
        ContentType: multipart/form-data, application/json
        Args:
            - prompts : One or several prompts (or one list of prompts) to complete.
            - return_tensors ?: Whether or not to return the tensors of predictions (as token indices) in the outputs. If set to True, the decoded text is not returned.
            - return_text ?: Whether or not to return the decoded texts in the outputs.
            - return_full_text ?: If set to False only added text is returned, otherwise the full text is returned. Only meaningful if return_text is set to True.
            - clean_up_tokenization_spaces ?: Whether or not to clean up the potential extra spaces in the text output.
            - prefix ?: Prefix added to prompt.
            - handle_long_generation ?: default strategy where nothing in particular happens
        Returns: {prompts:Union[str, List[str]], kwargs:Dict[str, Any]}
        """
        if request.headers["content-type"].startswith("multipart/form-data"):
            form = await request.form()
            raw_prompts = form.get("prompts")

            prompts: Union[str, List[str]]
            if isinstance(raw_prompts, str):
                prompts = raw_prompts
            else:
                prompts = [raw_prompt for raw_prompt in raw_prompts]

            return TextGenerationInferenceInput(
                prompts=prompts,
                kwargs=dict(
                    return_tensors=form_field_to_int(
                        "return_tensors", form.get("return_tensors")
                    ),
                    return_text=form_field_to_int(
                        "return_text", form.get("return_text")
                    ),
                    return_full_text=form_field_to_int(
                        "return_full_text", form.get("return_full_text")
                    ),
                    clean_up_tokenization_spaces=form_field_to_int(
                        "clean_up_tokenization_spaces",
                        form.get("clean_up_tokenization_spaces"),
                    ),
                    prefix=form.get("prefix"),
                    handle_long_generation=form.get("handle_long_generation"),
                    **form_field_to_json(
                        name="kwargs", value=form.get("kwargs"), on_none={}
                    ),
                ),
            )

        elif request.headers["content-type"].startswith("application/json"):
            data = await request.json()
            body = TextGenerationJSONInput.model_validate(data)
            return TextGenerationInferenceInput(
                prompts=body.prompts,
                kwargs=dict(
                    return_tensors=body.return_tensors,
                    return_text=body.return_text,
                    return_full_text=body.return_full_text,
                    clean_up_tokenization_spaces=body.clean_up_tokenization_spaces,
                    prefix=body.prefix,
                    handle_long_generation=body.handle_long_generation,
                    **body.kwargs,
                ),
            )

        else:
            raise HTTPException(status_code=400, detail="Invalid content type")

    # async def infer(
    #     self, data: TextGenerationInferenceInput
    # ) -> Union[List[TextGenerationInference], List[List[TextGenerationInference]]]:
    #     return self.pipeline(**data)

    # async def respond(
    #     self,
    #     output: Union[
    #         List[TextGenerationInference], List[List[TextGenerationInference]]
    #     ],
    # ):
    #     json_compatible_output = jsonable_encoder(output)
    #     return JSONResponse(content=json_compatible_output)
