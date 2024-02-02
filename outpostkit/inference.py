from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional

from httpx import Response

from outpostkit.client import Client
from outpostkit.exceptions import OutpostError
from outpostkit.resource import Namespace
from outpostkit.user import UserShortDetails


@dataclass
class DomainInInference:
    protocol: str
    name: str
    apexDomain: str
    id: str


@dataclass
class InferenceHuggingfaceModel:
    id: str
    keyId: Optional[str]
    revision: Optional[str]


@dataclass
class InferenceToOutpostModel:
    fullName: str


@dataclass
class InferenceOutpostModel:
    model: InferenceToOutpostModel
    revision: Optional[str]


@dataclass
class InferenceDeployment:
    id: str
    status: str
    createdAt: str
    concludedAt: Optional[str]
    updatedAt: str
    timeTakenS: Optional[int]
    creator: Optional[UserShortDetails]


@dataclass
class ReplicaScalingConfig:
    id: str
    min: int
    max: int
    scaledownPeriod: int
    targetPendingRequests: int

@dataclass
class InferenceResource:
    """
    A Inference Service on Outpost.
    """

    fullName: str
    """The fullName used to identify the inference service."""

    name: str
    """Name of the inference service."""

    visibility: Literal['public','private','internal']
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

    loadModelWeightsFrom: Literal['huggingface','outpost','none']

    createdAt: str

    updatedAt: str

    status: str

    instanceType: str

    port: int

    internalDomains: List[Dict[str,Any]]

    huggingfaceModel: Optional[InferenceHuggingfaceModel] = None

    outpostModel: Optional[InferenceOutpostModel] = None

    # creatorId: Optional[str]=None

    # currentDeploymentId: Optional[str]=None

    currentDeployment: Optional[InferenceDeployment]=None

    # thirdPartyKeyId: Optional[str] =None

    # configSchema: Optional[str] =None

    replicaScalingConfig: Optional[ReplicaScalingConfig] =None

    def __init__(self, *args, **kwargs)->None:
        for field in self.__annotations__:
            setattr(self, field, kwargs.get(field))

class InferencePredictor(Namespace):

    def __init__(self, client: Client, endpoint: str, predictionPath:str,containerType:str, taskType:str) -> None:
        self.endpoint = endpoint
        self.containerType = containerType
        self.taskType = taskType
        self.predictionPath = predictionPath

        super().__init__(client)


    def infer(self, **kwargs) -> Response:
        """Make predictions.

        Returns:
            The prediction.
        """
        if self.endpoint is None:
            raise OutpostError("No endpoint configured")
        resp = self._client._request(
            path=f"{self.endpoint}{self.predictionPath}", **kwargs
        )
        resp.raise_for_status()

        return resp

    async def async_infer(self, **kwargs) -> Response:
        """Make predictions.

        Returns:
            The prediction.
        """
        if self.endpoint is None:
            raise OutpostError("No endpoint configured")
        resp = await self._client._async_request(
            path=f"{self.endpoint}{self.predictionPath}", **kwargs
        )

        return resp

@dataclass
class ListInferenceDeploymentsResponse:
    total: int
    deployments: List[InferenceDeployment]

    def __init__(self, total:int, deployments:List[Dict])->None:
        deps:List[InferenceDeployment]=[]
        self.total= total
        for dep in deployments:
            deps.append(InferenceDeployment(**dep))
        self.deployments = deps

@dataclass
class InferenceDeployResponse:
    id: int


class Inference(Namespace):
    def __init__(self, client: Client, entity:str, name: str) -> None:
        self.entity = entity
        self.name = name
        self.fullName = f"{entity}/{name}"
        super().__init__(client)

    def get(self)->InferenceResource:
        """
        Get details about the inference endpoint
        """

        resp = self._client._request(path=f"/inferences/{self.fullName}", method="GET")
        resp.raise_for_status()

        return InferenceResource(**resp.json())

    async def async_get(self)->InferenceResource:
        """
        Get details about the inference endpoint
        """

        resp = await self._client._async_request(path=f"/inferences/{self.fullName}", method="GET")
        resp.raise_for_status()

        return InferenceResource(**resp.json())

    def deploy(self,data:Optional[Dict[str,Any]]=None)->InferenceDeployResponse:
        """
        Get details about the inference endpoint
        """

        resp = self._client._request(path=f"/inferences/{self.fullName}/deployments", method="POST",json=data)
        resp.raise_for_status()

        return InferenceDeployResponse(**resp.json())

    async def async_deploy(self,data:Optional[Dict[str,Any]]=None)->InferenceDeployResponse:
        """
        Get details about the inference endpoint
        """

        resp = await self._client._async_request(path=f"/inferences/{self.fullName}/deployments", method="POST",json=data)
        resp.raise_for_status()

        return InferenceDeployResponse(**resp.json())


    def list_deploymets(self)->ListInferenceDeploymentsResponse:
        """
        Get details about the inference endpoint
        """

        resp = self._client._request(path=f"/inferences/{self.fullName}/deployments", method="GET")
        resp.raise_for_status()

        return ListInferenceDeploymentsResponse(**resp.json())

    async def async_list_deploymets(self)->ListInferenceDeploymentsResponse:
        """
        Get details about the inference endpoint
        """

        resp = await self._client._async_request(path=f"/inferences/{self.fullName}/deployments", method="GET")
        resp.raise_for_status()

        return ListInferenceDeploymentsResponse(**resp.json())



    def update(self, fullName: str, data: Dict[str, Any])->None:
        """
        Update Inference
        """
        resp = self._client._request(
            "PUT", f"/inferences/{fullName}", json=data
        )

        obj = resp.json()
        return obj

    async def async_update(self, fullName: str, data: Dict[str, Any])->None:
        """
        Update Inference Async
        """
        await self._client._async_request(
            "PUT", f"/inferences/{fullName}", json=data
        )

    def update_name(self, fullName: str, name: str)->None:
        """
        Update Inference
        """
        resp = self._client._request(
            "PUT", f"/inferences/{fullName}/name", json=dict({"name": name})
        )

        obj = resp.json()
        return obj

    async def async_update_name(self, fullName: str, name: str)->None:
        """
        Update Inference Async
        """
        resp = await self._client._async_request(
            "PUT", f"/inferences/{fullName}/name", json=dict({"name": name})
        )

        obj = resp.json()
        return obj

    def delete(self, fullName: str)->None:
        """
        Update Inference
        """
        resp = self._client._request("DELETE", f"/inferences/{fullName}")

        obj = resp.json()
        return obj

    async def async_delete(self, fullName: str)->None:
        """
        Update Inference Async
        """
        resp = await self._client._async_request("DELETE", f"/inferences/{fullName}")

        obj = resp.json()
        return obj

@dataclass
class InferenceListResponse:
    total: int
    inferences: List[InferenceResource]

    def __init__(self, total:int, inferences:List[Dict])->None:
        infs:List[InferenceResource]=[]
        self.total= total
        for inf in inferences:
            infs.append(InferenceResource(**inf))
        self.inferences = infs

@dataclass
class InferenceCreateResponse:
    id: int
    name: str

class Inferences(Namespace):
    """
    A namespace for operations related to inferences of models.
    """

    def __init__(self, client: Client, entity: str) -> None:
        self.entity = entity
        super().__init__(client)

    def list(
        self,
    ) -> InferenceListResponse:
        """
        List inferences of models.

        Parameters:
            entity: The entity whos inferences you want to list.
        Returns:
            List[Inference]: A page of of model inferences.
        """
        resp = self._client._request("GET", f"/inferences/{self.entity}")

        obj = InferenceListResponse(**resp.json())
        return obj

    async def async_list(
        self,
    ) -> InferenceListResponse:
        """
        List inferences of models.

        Parameters:
            entity: The entity whos inferences you want to list.
        Returns:
            List[Inference]: A page of of model inferences.WW
        """
        resp = await self._client._async_request("GET", f"/inferences/{self.entity}")

        obj = InferenceListResponse(**resp.json())
        return obj

    def create(self, data: Dict[str, Any])->InferenceCreateResponse:
        """
        Create Inference
        """
        resp = self._client._request(
            "POST", f"/inferences/{self.entity}", json=data
        )

        obj = resp.json()
        return obj

    async def async_create(self, data: Dict[str, Any])->InferenceCreateResponse:
        """
        Create Inference Async
        """
        resp = await self._client._async_request(
            "POST", f"/inferences/{self.entity}", json=data
        )

        obj = resp.json()
        return obj
