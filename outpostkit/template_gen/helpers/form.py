import json
from typing import Any, Optional, Union

from fastapi import HTTPException
from starlette.datastructures import UploadFile


def form_field_to_int(
    name: str, value: Optional[Union[UploadFile, str]], mandatory=False
):
    if isinstance(value, str):
        try:
            parse_value = int(value)
            return parse_value
        except ValueError as e:
            raise HTTPException(
                status_code=400, detail=f"Unable to parse {name} field to int."
            ) from e
    elif value is None:
        if mandatory:
            raise HTTPException(status_code=400, detail=f"{name} field is mandatory.")
        return None
    else:
        raise HTTPException(
            status_code=400, detail=f"Unable to parse {name} field to int."
        )


def form_field_to_float(
    name: str, value: Optional[Union[UploadFile, str]], mandatory=False
):
    if isinstance(value, str):
        try:
            parse_value = float(value)
            return parse_value
        except ValueError as e:
            raise HTTPException(
                status_code=400, detail=f"Unable to parse {name} field to float."
            ) from e
    elif value is None:
        if mandatory:
            raise HTTPException(status_code=400, detail=f"{name} field is mandatory.")
        return None
    else:
        raise HTTPException(
            status_code=400, detail=f"Unable to parse {name} field to float."
        )


def form_field_to_boolean(
    name: str, value: Optional[Union[UploadFile, str]], mandatory=False
):
    if isinstance(value, str):
        lower_value = value.lower()
        if lower_value in ["true", "false"]:
            return lower_value == "true"
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unable to parse {name} field to boolean, accepted values are 'true' and 'false'.",
            )
    elif value is None:
        if mandatory:
            raise HTTPException(status_code=400, detail=f"{name} field is mandatory.")
        return None
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unable to parse {name} field to boolean, accepted values are 'true' and 'false'.",
        )


def form_field_to_json(
    name: str,
    value: Optional[Union[UploadFile, str]],
    mandatory=False,
    on_none: Any = None,
):
    if isinstance(value, str):
        try:
            parse_value = json.loads(value)
            return parse_value
        except json.JSONDecodeError as e:
            raise HTTPException(
                status_code=400, detail=f"Unable to parse {name} field to json."
            ) from e
    elif value is None:
        if mandatory:
            raise HTTPException(status_code=400, detail=f"{name} field is mandatory.")
        return on_none
    else:
        raise HTTPException(
            status_code=400, detail=f"Unable to parse {name} field to json."
        )


def form_field_to_str(
    name: str, value: Optional[Union[UploadFile, str]], mandatory=False
):
    if isinstance(value, str):
        return value
    elif value is None:
        if mandatory:
            raise HTTPException(status_code=400, detail=f"{name} field is mandatory.")
        return None
    else:
        raise HTTPException(
            status_code=400, detail=f"Unable to parse {name} field to string."
        )
