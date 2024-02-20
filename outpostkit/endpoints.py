from dataclasses import dataclass
from json import JSONDecodeError
from typing import Any, Dict, List, Literal, Optional, Tuple, Union, overload

import requests
from httpx import Response

from outpostkit._types.endpoint import EndpointDeployment, EndpointResource
from outpostkit.client import Client
from outpostkit.exceptions import OutpostError, PredictionHTTPException
from outpostkit.resource import Namespace


def _raise_for_status(resp: requests.Response) -> None:
    if 400 <= resp.status_code < 600:
        content_type, _, _ = resp.headers["content-type"].partition(";")
        # if content_type != "text/event-stream":
        #     raise ValueError(
        #         "Expected response Content-Type to be 'text/event-stream', "
        #         f"got {content_type!r}"
        #     )
        try:
            if content_type == "application/json":
                try:
                    data = resp.json()
                    if isinstance(data, dict):
                        raise PredictionHTTPException(
                            status_code=resp.status_code,
                            message="Prediction request failed.",
                            data=data,
                        ) from None
                    else:
                        raise PredictionHTTPException(
                            status_code=resp.status_code,
                            message="Prediction request failed.",
                            data=data,
                        ) from None
                except JSONDecodeError as e:
                    raise OutpostError("Failed to decode json body.") from e
            elif content_type == "text/plain":
                raise PredictionHTTPException(
                    status_code=resp.status_code, message=resp.text
                )
            elif content_type == "text/html":
                raise PredictionHTTPException(
                    status_code=resp.status_code, message=resp.text
                )
            else:
                raise PredictionHTTPException(
                    status_code=resp.status_code,
                    message=f"Request failed. Unhandled Content Type: {content_type}",
                )
        except Exception:
            raise


class Predictor(Namespace):
    def __init__(
        self,
        client: Client,
        endpoint: str,
        predictionPath: str,
        # containerType: str,
        # taskType: str,
        healthcheckPath: str,
    ) -> None:
        self.endpoint = endpoint
        self.predictionPath = predictionPath
        self.healthcheckPath = healthcheckPath

        super().__init__(client)

    def infer(self, **kwargs) -> requests.Response:
        """Make predictions.

        Returns:
            The prediction.
        """
        if self.endpoint is None:
            raise OutpostError("No endpoint configured")
        added_headers = kwargs.pop("headers", None)
        resp = requests.post(
            url=f"{self.endpoint}{self.predictionPath}",
            headers={
                "authorization": f"Bearer {str(self._client._api_token)}",
                **(added_headers if added_headers else {}),
            },
            **kwargs,
        )
        _raise_for_status(resp=resp)
        return resp

    def wake(self) -> requests.Response:
        """
        Current deployment status of the endpoint
        """
        resp = requests.get(
            url=f"{self.endpoint}{self.predictionPath}",
            headers={"authorization": str(self._client._api_token)},
        )
        return resp

    def healthcheck(self) -> requests.Response:
        """
        Current deployment status of the endpoint
        """
        # try:
        resp = requests.get(
            url=f"{self.endpoint}{self.healthcheckPath}",
        )
        return resp
        #     return resp
        #     return "healthy"
        # except Exception:
        #     return "unhealthy"

    async def async_infer(self, **kwargs) -> Response:
        """Make predictions.

        Returns:
            The prediction.
        """
        if self.endpoint is None:
            raise OutpostError("No endpoint configured")
        resp = await self._client._async_request(
            method="POST", path=f"{self.endpoint}{self.predictionPath}", **kwargs
        )

        return resp


@dataclass
class ListEndpointDeploymentsResponse:
    total: int
    deployments: List[EndpointDeployment]

    def __init__(self, total: int, deployments: List[Dict]) -> None:
        deps: List[EndpointDeployment] = []
        self.total = total
        for dep in deployments:
            deps.append(EndpointDeployment(**dep))
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
        Get details about the endpoint
        """

        resp = self._client._request(path=f"/endpoints/{self.fullName}", method="GET")
        resp.raise_for_status()

        return EndpointResource(**resp.json())

    def get_predictor(self) -> Predictor:
        """
        Get prediction client for the endpoint
        """

        resp = self._client._request(path=f"/endpoints/{self.fullName}", method="GET")
        resp.raise_for_status()

        endpt = EndpointResource(**resp.json())
        if endpt.primaryDomain is None:
            raise OutpostError("No primary domain set.")
        return Predictor(
            client=self._client,
            endpoint=f"{endpt.primaryDomain.protocol}://{endpt.primaryDomain.name}",
            predictionPath=endpt.predictionPath,
            # containerType=endpt.containerType,
            # taskType=endpt.taskType,
            healthcheckPath=endpt.healthcheckPath,
        )

    async def async_get(self) -> EndpointResource:
        """
        Get details about the endpoint
        """

        resp = await self._client._async_request(
            path=f"/endpoints/{self.fullName}", method="GET"
        )
        resp.raise_for_status()

        return EndpointResource(**resp.json())

    def deploy(self, data: Optional[Dict[str, Any]] = {}) -> EndpointDeployResponse:
        """
        Get details about the endpoint
        """

        resp = self._client._request(
            path=f"/endpoints/{self.fullName}/deployments", method="POST", json=data
        )
        return EndpointDeployResponse(**resp.json())

    async def async_deploy(
        self, data: Optional[Dict[str, Any]] = {}
    ) -> EndpointDeployResponse:
        """
        Get details about the endpoint
        """

        resp = await self._client._async_request(
            path=f"/endpoints/{self.fullName}/deployments", method="POST", json=data
        )


        return EndpointDeployResponse(**resp.json())

    def list_deployments(self, **kwargs) -> ListEndpointDeploymentsResponse:
        """
        Get details about the endpoint
        """

        resp = self._client._request(
            path=f"/endpoints/{self.fullName}/deployments", method="GET", **kwargs
        )


        return ListEndpointDeploymentsResponse(**resp.json())

    async def async_list_deployments(self, **kwargs) -> ListEndpointDeploymentsResponse:
        """
        Get details about the endpoint
        """

        resp = await self._client._async_request(
            path=f"/endpoints/{self.fullName}/deployments", method="GET", **kwargs
        )


        return ListEndpointDeploymentsResponse(**resp.json())

    def update(self, data: Dict[str, Any] = {}) -> Response:
        """
        Update endpoint
        """
        resp = self._client._request("PUT", f"/endpoints/{self.fullName}", json=data)

        return resp

    async def async_update(self, data: Dict[str, Any] = {}) -> Response:
        """
        Update endpoint
        """
        resp = await self._client._async_request(
            "PUT", f"/endpoints/{self.fullName}", json=data
        )

        return resp

    def update_name(self, name: str) -> Response:
        """
        Update endpoint
        """
        resp = self._client._request(
            "PUT", f"/endpoints/{self.fullName}/name", json=dict({"name": name})
        )

        return resp

    async def async_update_name(self, name: str) -> Response:
        """
        Update endpoint
        """
        resp = await self._client._async_request(
            "PUT", f"/endpoints/{self.fullName}/name", json=dict({"name": name})
        )

        return resp

    def delete(self) -> Response:
        """
        Update endpoint
        """
        resp = self._client._request("DELETE", f"/endpoints/{self.fullName}")

        return resp

    async def async_delete(self) -> Response:
        """
        Update endpoint
        """
        resp = await self._client._async_request(
            "DELETE", f"/endpoints/{self.fullName}"
        )

        return resp

    def dep_status(self) -> Dict[str, Any]:
        """
        Current deployment status of the endpoint
        """
        resp = self._client._request(
            "GET",
            f"/endpoints/{self.fullName}/status",
        )


        obj = resp.json()
        return obj

    def get_logs(
        self,
        deploymentId: str,
        type: Literal["dep", "runtime", "events"] = "runtime",
        sequence: Optional[int] = None,
    ) -> List[Tuple[str, str]]:
        resp = self._client._request(
            "GET",
            f"/endpoints/{self.fullName}/deployments/{deploymentId}/logs/{type}",
            params={sequence},
        )


        obj = [(str(log.time), str(log.message)) for log in resp.json()]
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
        Current deployment status of the endpoint.
        """
        resp = self._client._request(
            "GET",
            f"/endpoints/{self.fullName}/custom-template-file",
        )

        if destination_path:
            with open(destination_path, "wb") as file:
                file.write(resp.content)
        else:
            return resp.content


@dataclass
class EndpointListResponse:
    total: int
    endpoints: List[EndpointResource]

    def __init__(self, total: int, endpoints: List[Dict]) -> None:
        infs: List[EndpointResource] = []
        self.total = total
        for inf in endpoints:
            infs.append(EndpointResource(**inf))
        self.endpoints = infs


@dataclass
class EndpointCreateResponse:
    id: int
    name: str


class Endpoints(Namespace):
    """
    A namespace for operations related to endpoints.
    """

    def __init__(self, client: Client, entity: str) -> None:
        self.entity = entity
        super().__init__(client)

    def list(
        self,
    ) -> EndpointListResponse:
        """
        List endpoints.

        Parameters:
            entity: The entity whos endpoints you want to list.
        """
        resp = self._client._request("GET", f"/endpoints/{self.entity}")

        obj = EndpointListResponse(**resp.json())
        return obj

    async def async_list(
        self,
    ) -> EndpointListResponse:
        """
        List endpoints.

        Parameters:
            entity: The entity whos endpoints you want to list.
        """
        resp = await self._client._async_request("GET", f"/endpoints/{self.entity}")

        obj = EndpointListResponse(**resp.json())
        return obj

    def create(self, **kwargs) -> Endpoint:
        """
        Create endpoint
        """
        resp = self._client._request("POST", f"/endpoints/{self.entity}", **kwargs)

        obj = EndpointCreateResponse(**resp.json())
        return Endpoint(client=self._client, entity=self.entity, name=obj.name)

    async def async_create(self, **kwargs) -> Endpoint:
        """
        Create endpoint
        """
        resp = await self._client._async_request(
            "POST", f"/endpoints/{self.entity}", **kwargs
        )

        obj = EndpointCreateResponse(**resp.json())
        return Endpoint(client=self._client, entity=self.entity, name=obj.name)
