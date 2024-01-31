from typing import Any, Dict, List, Optional, TypedDict, Union

from fastapi import HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from generic.tasks.common import AbstractInferenceHandler
from generic.tasks.utils import audio_to_b64
from pydantic import BaseModel


class TextToAudioJSONInput(BaseModel):
    text_inputs: Union[str, List[str]]
    forward_params: Optional[Dict[str, Any]] = {}


class TextToAudioInferenceInput(TypedDict):
    text_inputs: Union[str, List[str]]
    forward_params: Dict[str, Any]


class TextToAudioInference(TypedDict):
    audio: List[float]
    sampling_rate: int


class TextToAudioInferenceHandler(AbstractInferenceHandler):
    async def parse_request(
        self, request: Request, temp_dir: str
    ) -> TextToAudioInferenceInput:
        """
        ContentType: application/json
        Args:
            - text_inputs : The text(s) to generate.
            - forward_params ?: Parameters passed to the model generation/forward method.
        Returns: {text_inputs:Union[str, List[str]], forward_params:Optional[Dict[str, Any]]}
        """
        if request.headers["content-type"].startswith("application/json"):
            data = await request.json()
            body = TextToAudioJSONInput.model_validate(data)
            return TextToAudioInferenceInput(
                text_inputs=body.text_inputs,
                forward_params=body.forward_params,
            )
        else:
            raise HTTPException(
                status_code=400, detail="Invalid content type. Accepts application/json"
            )

    # async def infer(
    #     self, data: TextToAudioInferenceInput
    # ) -> Union[List[TextToAudioInference], TextToAudioInference]:
    #     return self.pipeline(**data)

    async def respond(
        self, output: Union[List[TextToAudioInference], TextToAudioInference]
    ):
        print(type(output["audio"]))
        # sampling_rate = self.pipeline.config.audio_encoder.sampling_rate
        if isinstance(output, List):
            audio = map(lambda o: audio_to_b64(o["audio"], o["sampling_rate"]), output)
        else:
            audio = audio_to_b64(output["audio"], output["sampling_rate"])
        json_compatible_output = jsonable_encoder({"audio": audio})
        return JSONResponse(content=json_compatible_output)
