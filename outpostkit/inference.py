import json
from typing import Any, Dict, List, Optional, TypedDict

from httpx import Response

from outpostkit.client import Client
from outpostkit.exceptions import OutpostError
from outpostkit.logger import outpost_logger
from outpostkit.resource import Namespace, Resource


class DomainInInference(TypedDict):
    protocol: str
    name: str
    apexDomain: str
    id: str


class InferenceHuggingfaceModel(TypedDict):
    id: str
    keyId: Optional[str]
    revision: Optional[str]


class InferenceToOutpostModel(TypedDict):
    fullName: str


class InferenceOutpostModel(TypedDict):
    model: InferenceToOutpostModel
    revision: Optional[str]


class Inference(Namespace):
    id: str
    fullName: str
    predictionPath: str
    healthcheckPath: str
    containerType: str
    endpoint: Optional[str]

    def __init__(self, client: Client,id: str,fullName: str,predictionPath: str,healthcheckPath: str,containerType: str, endpoint: Optional[str]=None) -> None:
        self.fullName = fullName
        self.predictionPath = predictionPath
        self.healthcheckPath= healthcheckPath
        self.containerType = containerType
        self.endpoint = endpoint
        self.id = id
        super().__init__(client)

    def get(self):
        """
        Get details about the inference endpoint
        """

        resp = self._client._request(path=f"/inferences/{self.fullName}",method="GET")
        resp.raise_for_status()

        return resp.json()

    async def async_get(self):
        """
        Get details about the inference endpoint
        """

        resp = await self._client._async_request(path=f"/inferences/{self.fullName}",method="GET")
        resp.raise_for_status()

        return resp.json()

    def infer(self, **kwargs)->Response:
        """Make predictions.

        Returns:
            The prediction.
        """
        if(self.endpoint is None):
            raise OutpostError("No endpoint configured")
        resp = self._client._request(path=f"{self.endpoint}{self.predictionPath}",**kwargs)
        resp.raise_for_status()

        return resp

    async def async_infer(self, **kwargs)->Response:
        """Make predictions.

        Returns:
            The prediction.
        """
        if(self.endpoint is None):
            raise OutpostError("No endpoint configured")
        resp = await self._client._async_request(path=f"{self.endpoint}{self.predictionPath}",**kwargs)

        return resp

    def update(self,fullName:str,data:Dict[str,Any]):
        """
        Update Inference
        """
        resp = self._client._request("PUT", f"/inferences/{fullName}",json=json.dumps(data))

        obj = resp.json()
        return obj

    async def async_update(self,fullName:str,data:Dict[str,Any]):
        """
        Update Inference Async
        """
        resp = await self._client._async_request("PUT", f"/inferences/{fullName}",json=json.dumps(data))

        obj = resp.json()
        return obj

    def update_name(self,fullName:str,name:str):
        """
        Update Inference
        """
        resp = self._client._request("PUT", f"/inferences/{fullName}/name",json=json.dumps(dict({"name":name})))

        obj = resp.json()
        return obj

    async def async_update_name(self,fullName:str,name:str):
        """
        Update Inference Async
        """
        resp = await self._client._async_request("PUT", f"/inferences/{fullName}/name",json=json.dumps(dict({"name":name})))

        obj = resp.json()
        return obj

    def delete(self,fullName:str):
        """
        Update Inference
        """
        resp = self._client._request("DELETE", f"/inferences/{fullName}")

        obj = resp.json()
        return obj

    async def async_delete(self,fullName:str):
        """
        Update Inference Async
        """
        resp = await self._client._async_request("DELETE", f"/inferences/{fullName}")

        obj = resp.json()
        return obj

class InferenceResource(Resource):

    """
    A Inference Service on Outpost.
    """

    fullName: str
    """The fullName used to identify the inference service."""

    name: str
    """Name of the inference service."""

    id: str
    """ID of the inference service."""

    ownerId: str
    """Owner of the inference service."""

    containerType: str
    """Container type of the inference service."""

    taskType: str
    """Task type of the inference service."""

    config: dict
    """Config of the inference service."""

    predictionPath: str
    """Relative path used for prediction and target for scaling."""

    healthcheckPath: str
    """Relative path used for healthcheck and readiness probes"""

    domains: List[DomainInInference]

    loadModelWeightsFrom: str

    huggingfaceModel: Optional[InferenceHuggingfaceModel]

    outpostModel: Optional[InferenceOutpostModel]

    def to_inference(self,client:Client,domain_index:int=0)->Inference:
        if(domain_index < len(self.domains)):
            domain = self.domains[0]
            endpoint = f"{domain.get('protocol')}://{domain.get('name')}"
            return Inference(client=client,id=self.id,fullName=self.fullName,predictionPath=self.predictionPath,healthcheckPath=self.healthcheckPath,endpoint=endpoint,containerType=self.containerType)
        else:
            outpost_logger.warning("Did not find required domain. Initializing without the endpoint.")
            return Inference(client=client,id=self.id,fullName=self.fullName,predictionPath=self.predictionPath,healthcheckPath=self.healthcheckPath,containerType=self.containerType)



class Inferences(Namespace):
    """
    A namespace for operations related to inferences of models.
    """
    def __init__(self, client: Client,entity:str) -> None:
        self.entity = entity
        super().__init__(client)

    def list(
        self,
    ) -> List[Inference]:
        """
        List inferences of models.

        Parameters:
            entity: The entity whos inferences you want to list.
        Returns:
            List[Inference]: A page of of model inferences.
        """
        resp = self._client._request("GET", f"/inferences/{self.entity}")

        obj = resp.json()
        obj["inferences"] = [_json_to_inference_resource(result) for result in obj["inferences"]]

        return obj

    async def async_list(
        self,
    ) -> List[Inference]:
        """
        List inferences of models.

        Parameters:
            entity: The entity whos inferences you want to list.
        Returns:
            List[Inference]: A page of of model inferences.WW
        """
        resp = await self._client._async_request("GET", f"/inferences/{self.entity}")

        obj = resp.json()
        obj["inferences"] = [_json_to_inference_resource(result) for result in obj["inferences"]]

        return obj
    def create(self,data:Dict[str,Any]):
        """
        Create Inference
        """
        resp = self._client._request("POST", f"/inferences/{self.entity}",json=json.dumps(data))

        obj = resp.json()
        return obj

    async def async_create(self,data:Dict[str,Any]):
        """
        Create Inference Async
        """
        resp = await self._client._async_request("POST", f"/inferences/{self.entity}",json=json.dumps(data))

        obj = resp.json()
        return obj


def _json_to_inference_resource(json: Dict[str, Any]) -> InferenceResource:
    return InferenceResource(**json)
