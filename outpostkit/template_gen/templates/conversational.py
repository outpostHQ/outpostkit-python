from typing import Any, Dict, List, Optional, TypedDict, Union

from fastapi import HTTPException, Request
from pydantic import BaseModel
from transformers import Conversation


class ConversationalJSONInput(BaseModel):
    conversations: Union[str, List[Dict[str, str]]]
    clean_up_tokenization_spaces: Optional[bool] = None
    kwargs: Optional[Dict[str, Any]] = {}


class ConversationalInferenceInput(TypedDict):
    conversation: Conversation
    kwargs: Dict[str, Any]


class ConversationalInference(TypedDict):
    conversation: Conversation
    clean_up_tokenization_spaces: Optional[bool]


async def conversational_request_parser(
    request: Request, temp_dir: str
) -> ConversationalInferenceInput:
    if request.headers["content-type"].startswith("application/json"):
        data = await request.json()
        body = ConversationalJSONInput.model_validate(data)
        conversation = Conversation(body.conversations)
        return ConversationalInferenceInput(
            conversation=conversation,
            kwargs=dict(
                clean_up_tokenization_spaces=body.clean_up_tokenization_spaces,
                **body.kwargs,
            ),
        )
    else:
        raise HTTPException(status_code=400, detail="Invalid content type")
