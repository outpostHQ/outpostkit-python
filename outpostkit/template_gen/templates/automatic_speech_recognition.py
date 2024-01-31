from typing import Any, Dict, List, Optional, TypedDict, Union

from fastapi import HTTPException, Request
from pydantic import BaseModel
from starlette.datastructures import UploadFile

from outpostkit.template_gen.helpers.form import (
    form_field_to_boolean,
    form_field_to_int,
    form_field_to_json,
)


class AutomaticSpeechRecognitionJSONInput(BaseModel):
    inputs: str
    return_timestamps: Optional[Union[bool, str]]
    max_new_tokens: Optional[int]
    kwargs: Optional[Dict[str, Any]] = {}


class AutomaticSpeechRecognitionInferenceInput(TypedDict):
    inputs: str | bytes
    kwargs: Dict[str, Any]


class AutomaticSpeechRecognitionInference(TypedDict):
    text: str
    chunks: Optional[List[Dict[str, Union[str, tuple[float, float]]]]]


async def automatic_speech_recognition_request_parser(request: Request, **kwargs):
    if request.headers["content-type"].startswith("multipart/form-data"):
        form = await request.form()
        raw_inputs = form.get("inputs")

        inputs: str | bytes
        if isinstance(raw_inputs, UploadFile):
            inputs = await raw_inputs.read()
        elif isinstance(raw_inputs, str):
            inputs = raw_inputs
        else:
            raise HTTPException(status_code=400, detail="Invalid content type")
        return AutomaticSpeechRecognitionInferenceInput(
            inputs=inputs,
            kwargs=dict(
                return_timestamps=form_field_to_boolean(
                    "return_timestamps", form.get("return_timestamps")
                ),
                max_new_tokens=form_field_to_int(
                    "max_new_tokens", form.get("max_new_tokens")
                ),
                **(
                    form_field_to_json(
                        name="kwargs", value=form.get("kwargs"), on_none={}
                    )
                ),
            ),
        )

    elif request.headers["content-type"].startswith("application/json"):
        data = await request.json()
        body = AutomaticSpeechRecognitionJSONInput.model_validate(data)
        return body

    else:
        raise HTTPException(status_code=400, detail="Invalid content type")
