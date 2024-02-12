from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union, overload

from httpx import Response

from outpostkit._types.inference import EndpointResource, InferenceDeployment
from outpostkit.client import Client
from outpostkit.exceptions import OutpostError
from outpostkit.resource import Namespace


class Predictior(Namespace):
    def __init__(
        self,
        client: Client,
        endpoint: str,
        predictionPath: str,
        containerType: str,
        taskType: str,
        healthcheckPath: str,
    ) -> None:
        self.endpoint = endpoint
        self.containerType = containerType
        self.taskType = taskType
        self.predictionPath = predictionPath
        self.healthcheckPath = healthcheckPath

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

    def wake(self) -> Response:
        """
        Current deployment status of the inference
        """
        resp = self._client._request(
            "GET", path=f"{self.endpoint}{self.predictionPath}"
        )

        return resp

    def healthcheck(self) -> Response:
        """
        Current deployment status of the inference
        """
        resp = self._client._request(
            "GET",
            path=f"{self.endpoint}{self.healthcheckPath}",
        )

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
class ListEndpointDeploymentsResponse:
    total: int
    deployments: List[InferenceDeployment]

    def __init__(self, total: int, deployments: List[Dict]) -> None:
        deps: List[InferenceDeployment] = []
        self.total = total
        for dep in deployments:
            deps.append(InferenceDeployment(**dep))
        self.deployments = deps


@dataclass
class EndpointDeployResponse:
    id: int


class Endpoint(Namespace):
    def __init__(self, client: Client, entity: str, name: str) -> None:
        self.entity = entity
        self.name = name
        self.fullName = f"{entity}/{name}"
        super().__init__(client)

    def get(self) -> EndpointResource:
        """
        Get details about the inference endpoint
        """

        resp = self._client._request(path=f"/inferences/{self.fullName}", method="GET")
        resp.raise_for_status()

        return EndpointResource(**resp.json())

    async def async_get(self) -> EndpointResource:
        """
        Get details about the inference endpoint
        """

        resp = await self._client._async_request(
            path=f"/inferences/{self.fullName}", method="GET"
        )
        resp.raise_for_status()

        return EndpointResource(**resp.json())

    def deploy(self, data: Optional[Dict[str, Any]] = None) -> EndpointDeployResponse:
        """
        Get details about the inference endpoint
        """

        resp = self._client._request(
            path=f"/inferences/{self.fullName}/deployments", method="POST", json=data
        )
        resp.raise_for_status()

        return EndpointDeployResponse(**resp.json())

    async def async_deploy(
        self, data: Optional[Dict[str, Any]] = None
    ) -> EndpointDeployResponse:
        """
        Get details about the inference endpoint
        """

        resp = await self._client._async_request(
            path=f"/inferences/{self.fullName}/deployments", method="POST", json=data
        )
        resp.raise_for_status()

        return EndpointDeployResponse(**resp.json())

    def list_deployments(self, **kwargs) -> ListEndpointDeploymentsResponse:
        """
        Get details about the inference endpoint
        """

        resp = self._client._request(
            path=f"/inferences/{self.fullName}/deployments", method="GET", **kwargs
        )
        resp.raise_for_status()

        return ListEndpointDeploymentsResponse(**resp.json())

    async def async_list_deployments(self, **kwargs) -> ListEndpointDeploymentsResponse:
        """
        Get details about the inference endpoint
        """

        resp = await self._client._async_request(
            path=f"/inferences/{self.fullName}/deployments", method="GET", **kwargs
        )
        resp.raise_for_status()

        return ListEndpointDeploymentsResponse(**resp.json())

    def update(self, data: Dict[str, Any] = {}) -> None:
        """
        Update Inference
        """
        resp = self._client._request("PUT", f"/inferences/{self.fullName}", json=data)
        resp.raise_for_status()
        obj = resp.json()
        return obj

    async def async_update(self, data: Dict[str, Any] = {}) -> None:
        """
        Update Inference Async
        """
        await self._client._async_request(
            "PUT", f"/inferences/{self.fullName}", json=data
        )

    def update_name(self, name: str) -> None:
        """
        Update Inference
        """
        resp = self._client._request(
            "PUT", f"/inferences/{self.fullName}/name", json=dict({"name": name})
        )
        resp.raise_for_status()

        obj = resp.json()
        return obj

    async def async_update_name(self, name: str) -> None:
        """
        Update Inference Async
        """
        resp = await self._client._async_request(
            "PUT", f"/inferences/{self.fullName}/name", json=dict({"name": name})
        )
        resp.raise_for_status()

        obj = resp.json()
        return obj

    def delete(self) -> None:
        """
        Update Inference
        """
        resp = self._client._request("DELETE", f"/inferences/{self.fullName}")
        resp.raise_for_status()

        obj = resp.json()
        return obj

    async def async_delete(self) -> None:
        """
        Update Inference Async
        """
        resp = await self._client._async_request(
            "DELETE", f"/inferences/{self.fullName}"
        )
        resp.raise_for_status()

        obj = resp.json()
        return obj

    def dep_status(self) -> Dict[str, Any]:
        """
        Current deployment status of the inference
        """
        resp = self._client._request(
            "GET",
            f"/inferences/{self.fullName}/status",
        )
        resp.raise_for_status()

        obj = resp.json()
        return obj

    @overload
    def download_custom_template(self) -> bytes:
        ...

    @overload
    def download_custom_template(self, destination_path: str) -> None:
        ...

    def download_custom_template(
        self, destination_path: Optional[str] = None
    ) -> Union[bytes, None]:
        """
        Current deployment status of the inference
        """
        resp = self._client._request(
            "GET",
            f"/inferences/{self.fullName}/custom-template-file",
        )
        resp.raise_for_status()
        if destination_path:
            with open(destination_path, "wb") as file:
                file.write(resp.content)
        else:
            return resp.content


@dataclass
class EndpointListResponse:
    total: int
    inferences: List[EndpointResource]

    def __init__(self, total: int, inferences: List[Dict]) -> None:
        infs: List[EndpointResource] = []
        self.total = total
        for inf in inferences:
            infs.append(EndpointResource(**inf))
        self.inferences = infs


@dataclass
class EndpointCreateResponse:
    id: int
    name: str


class Endpoints(Namespace):
    """
    A namespace for operations related to inferences of models.
    """

    def __init__(self, client: Client, entity: str) -> None:
        self.entity = entity
        super().__init__(client)

    def list(
        self,
    ) -> EndpointListResponse:
        """
        List inferences of models.

        Parameters:
            entity: The entity whos inferences you want to list.
        Returns:
            List[Inference]: A page of of model inferences.
        """
        resp = self._client._request("GET", f"/inferences/{self.entity}")
        print(resp.json())
        obj = EndpointListResponse(**resp.json())
        return obj

    async def async_list(
        self,
    ) -> EndpointListResponse:
        """
        List inferences of models.

        Parameters:
            entity: The entity whos inferences you want to list.
        Returns:
            List[Inference]: A page of of model inferences.WW
        """
        resp = await self._client._async_request("GET", f"/inferences/{self.entity}")

        obj = EndpointListResponse(**resp.json())
        return obj

    def create(self, **kwargs) -> EndpointCreateResponse:
        """
        Create Inference
        """
        resp = self._client._request("POST", f"/inferences/{self.entity}", **kwargs)

        obj = EndpointCreateResponse(**resp.json())
        return obj

    async def async_create(self, **kwargs) -> EndpointCreateResponse:
        """
        Create Inference Async
        """
        resp = await self._client._async_request(
            "POST", f"/inferences/{self.entity}", **kwargs
        )

        obj = EndpointCreateResponse(**resp.json())
        return obj
