from typing import Any, Dict, Optional, TypedDict

from fastapi import HTTPException, Request
from pydantic import BaseModel
from starlette.datastructures import UploadFile

from outpostkit.inference.helpers.form import (
    form_field_to_int,
    form_field_to_json,
)


class AudioClassificationJSONInput(BaseModel):
    inputs: str
    top_k: Optional[int] = None
    kwargs: Optional[Dict[str, Any]] = {}


class AudioClassificationInferenceInput(TypedDict):
    inputs: str | bytes
    kwargs: Dict[str, Any]


class AudioClassificationInference(TypedDict):
    label: str
    score: float


async def request_parser(request: Request, **kwargs):
    """
    ContentType: multipart/form-data, application/json
    Args:
        - inputs : input to be classified. either string(url) or actual audio file(through multipart);
        - topk ?: The number of top labels that will be returned by the pipeline.
    Returns: {inputs:Union[str,bytes], topk:Optional[None]}
    """
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

        return AudioClassificationInferenceInput(
            inputs=inputs,
            kwargs=dict(
                top_k=form_field_to_int("top_k", form.get("top_k")),
                **form_field_to_json(
                    name="kwargs", value=form.get("kwargs"), on_none={}
                ),
            ),
        )

    elif request.headers["content-type"].startswith("application/json"):
        data = await request.json()
        body = AudioClassificationJSONInput.model_validate(data)
        return AudioClassificationInferenceInput(
            inputs=body.inputs, kwargs=dict(top_k=body.top_k, **body.kwargs)  # type: ignore
        )

    else:
        raise HTTPException(status_code=400, detail="Invalid content type")
