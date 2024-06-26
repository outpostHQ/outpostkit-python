import os
from typing import Optional

from outpostkit._types.endpoint import (
    EndpointAutogeneratedHFModelDetails,
    EndpointAutogeneratedTemplateConfig,
)
from outpostkit._utils.constants import OutpostSecret
from outpostkit.client import Client
from outpostkit.endpoints import Endpoints

API_TOKEN = os.getenv("OUTPOST_API_TOKEN")
HF_TOKEN: Optional[str] = None
ENTITY: str = "aj-ya"
template = EndpointAutogeneratedTemplateConfig(
    modelSource="huggingface",
    huggingfaceModel=EndpointAutogeneratedHFModelDetails(
        id="nomic-ai/nomic-embed-text-v1",
    ),
)

endpt = Endpoints(client=Client(api_token=API_TOKEN), entity=ENTITY).create(
    template=template,
    name="text-embedder-2",
    hardware_instance="1xnvidia-tesla-t4",
    secrets=(
        [OutpostSecret(name="HUGGING_FACE_HUB_TOKEN", value=HF_TOKEN)]
        if HF_TOKEN
        else None
    ),
)
endpt.deploy()

print(f"name: {endpt.name}")
print(f"home: https://outpost.run/{ENTITY}/inference-endpoints/{endpt.name}/overview")


# wait for endpoint to start.

# if endpt.get().status == "healthy":
#     predictor = endpt.create_predictor()
#     predictor.infer(json={"sentences": "hello."})
