import json
from base64 import b64encode
from io import BytesIO
from typing import Any, Optional, Union

import numpy as np
from fastapi import HTTPException, UploadFile
from fastapi.encoders import jsonable_encoder
from PIL.Image import Image
from scipy.io.wavfile import write
from torch import Tensor


def form_field_to_int(
    name: str, value: Optional[Union[UploadFile, str]], mandatory=False
):
    if isinstance(value, str):
        try:
            parse_value = int(value)
            return parse_value
        except ValueError:
            raise HTTPException(
                status_code=400, detail=f"Unable to parse {name} field to int."
            ) from None
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
        except ValueError:
            raise HTTPException(
                status_code=400, detail=f"Unable to parse {name} field to float."
            ) from ValueError
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
        except json.JSONDecodeError:
            raise HTTPException(
                status_code=400, detail=f"Unable to parse {name} field to json."
            ) from json.JSONDecodeError
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


def pil_to_b64(im: Image):
    buffered = BytesIO()
    im.save(buffered, format="JPEG")
    return "data:image/jpeg;base64," + b64encode(buffered.getvalue()).decode()


def audio_to_b64(wav, sr: int):
    bytes_wav = b""
    byte_io = BytesIO(bytes_wav)
    write(byte_io, sr, wav)
    wav_bytes = byte_io.read()
    return "data:audio/wav;base64," + b64encode(wav_bytes).decode()


def np_json_serialize(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return jsonable_encoder(obj)


custom_encoders = {
    Image: lambda x: pil_to_b64(x),
    Tensor: lambda x: np_json_serialize(x.numpy()),
    # "tf.Tensor": lambda x: x.numpy(),
}
