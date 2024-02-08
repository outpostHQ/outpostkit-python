from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from httpx import Response

from outpostkit._types.inference import InferenceDeployment, InferenceResource
from outpostkit.client import Client
from outpostkit.exceptions import OutpostError
from outpostkit.resource import Namespace


class Predictior(Namespace):
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

    def list_deploymets(self, **kwargs) -> ListInferenceDeploymentsResponse:
        """
        Get details about the inference endpoint
        """

        resp = self._client._request(
            path=f"/inferences/{self.fullName}/deployments", method="GET", **kwargs
        )
        resp.raise_for_status()

        return ListInferenceDeploymentsResponse(**resp.json())

    async def async_list_deploymets(self, **kwargs) -> ListInferenceDeploymentsResponse:
        """
        Get details about the inference endpoint
        """

        resp = await self._client._async_request(
            path=f"/inferences/{self.fullName}/deployments", method="GET", **kwargs
        )
        resp.raise_for_status()

        return ListInferenceDeploymentsResponse(**resp.json())



    def update(self, data: Dict[str, Any]= {})->None:
        """
        Update Inference
        """
        resp = self._client._request(
            "PUT", f"/inferences/{self.fullName}", json=data
        )

        obj = resp.json()
        return obj

    async def async_update(self,data: Dict[str, Any]= {})->None:
        """
        Update Inference Async
        """
        await self._client._async_request(
            "PUT", f"/inferences/{self.fullName}", json=data
        )

    def update_name(self, name: str)->None:
        """
        Update Inference
        """
        resp = self._client._request(
            "PUT", f"/inferences/{self.fullName}/name", json=dict({"name": name})
        )

        obj = resp.json()
        return obj

    async def async_update_name(self, name: str)->None:
        """
        Update Inference Async
        """
        resp = await self._client._async_request(
            "PUT", f"/inferences/{self.fullName}/name", json=dict({"name": name})
        )

        obj = resp.json()
        return obj

    def delete(self)->None:
        """
        Update Inference
        """
        resp = self._client._request("DELETE", f"/inferences/{self.fullName}")

        obj = resp.json()
        return obj

    def dep_status(self)-> Dict[str,Any]:
        """
        Current deployment status of the inference
        """
        resp = self._client._request(
            "GET", f"/inference/{self.fullName}/status",
        )

        obj = resp.json()
        return obj

    def wake(self)-> Dict[str,Any]:
        """
        Current deployment status of the inference
        """
        resp = self._client._request(
            "GET", f"/inference/{self.fullName}",
        )

        obj = resp.json()
        return obj


    async def async_delete(self)->None:
        """
        Update Inference Async
        """
        resp = await self._client._async_request("DELETE", f"/inferences/{self.fullName}")

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
        print(resp.json())
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

    def create(self, **kwargs) -> InferenceCreateResponse:
        """
        Create Inference
        """
        resp = self._client._request("POST", f"/inferences/{self.entity}", **kwargs)

        obj = InferenceCreateResponse(**resp.json())
        return obj

    async def async_create(self, **kwargs) -> InferenceCreateResponse:
        """
        Create Inference Async
        """
        resp = await self._client._async_request(
            "POST", f"/inferences/{self.entity}", **kwargs
        )

        obj = InferenceCreateResponse(**resp.json())
        return obj
