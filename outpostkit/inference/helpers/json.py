from base64 import b64encode
from io import BytesIO
from typing import TYPE_CHECKING

import numpy as np
from fastapi.encoders import jsonable_encoder

if TYPE_CHECKING:
    from PIL.Image import Image


def pil_to_b64(im: "Image"):
    buffered = BytesIO()
    im.save(buffered, format="JPEG")
    return "data:image/jpeg;base64," + b64encode(buffered.getvalue()).decode()


def audio_to_b64(wav, sr: int):
    from scipy.io.wavfile import write

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
