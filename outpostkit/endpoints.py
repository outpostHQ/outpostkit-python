import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional, Tuple, Union
from urllib.parse import urlparse

from httpx import Response

from outpostkit._types.endpoint import (
    EndpointAutogeneratedTemplateConfig,
    EndpointCustomTemplateConfig,
    EndpointDeployment,
    EndpointPrebuiltContainerDetails,
    EndpointReplicaStatus,
    EndpointResource,
    EndpointSecret,
    ReplicaScalingConfig,
)
from outpostkit._utils.constants import ServiceVisibility, scaffolding_file
from outpostkit.client import Client
from outpostkit.exceptions import OutpostError
from outpostkit.predictor import Predictor
from outpostkit.resource import Namespace


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
        Get essential details about the endpoint.
        """

        resp = self._client._request(path=f"/endpoints/{self.fullName}", method="GET")
        resp.raise_for_status()

        return EndpointResource(**resp.json())

    def list_deployments(
        self,
        sort_by: Optional[
            Literal["updatedAt", "createdAt", "concludedAt", "timeTakenS"]
        ] = None,
        sort_descriptor: Optional[Literal["desc", "asc"]] = None,
        status: Optional[str] = None,
        creator_id: Optional[str] = None,
        query: Optional[str] = None,
        limit: Optional[int] = None,
        skip: Optional[int] = None,
        cursor_id: Optional[str] = None,
    ) -> ListEndpointDeploymentsResponse:
        """
        List the endpoint's deployments.
        """

        resp = self._client._request(
            path=f"/endpoints/{self.fullName}/deployments",
            method="GET",
            params={
                "sb": sort_by,
                "sd": sort_descriptor,
                "l": limit,
                "status": status,
                "creatorId": creator_id,
                "q": query,
                "skip": skip,
                "cursorId": cursor_id,
            },
        )

        return ListEndpointDeploymentsResponse(**resp.json())

    def deploy(
        self,
        wakeup: bool = True,  # noqa: FBT001, FBT002
    ) -> EndpointDeployResponse:
        """
        Deploy the endpoint.
        """

        resp = self._client._request(
            path=f"/endpoints/{self.fullName}/deployments",
            method="POST",
            json={"wakeup": wakeup},
        )
        return EndpointDeployResponse(**resp.json())

    def create_predictor(self) -> Predictor:
        """
        Creates a client to interact with the endpoint to get predictions.
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
            healthcheckPath=endpt.healthcheckPath,
        )

    def update(
        self, task_type: Optional[str] = None, hardware_instance: Optional[str] = None
    ) -> Response:
        """
        Update endpoint's task type and/or hardware instance.
        """
        resp = self._client._request(
            "PUT",
            f"/endpoints/{self.fullName}",
            json={"taskType": task_type, "hardwareInstance": hardware_instance},
        )

        return resp

    def update_name(self, name: str) -> None:
        """
        Update the name of the endpoint.
        Note: This can make the old URLs pointing to the endpoint stale.
        """
        self._client._request(
            "PUT", f"/endpoints/{self.fullName}/name", json={"name": name}
        )
        self.name = name

    def delete(self) -> None:
        """
        Delete the endpoint.
        """
        self._client._request("DELETE", f"/endpoints/{self.fullName}")

    def get_replica_status(self) -> EndpointReplicaStatus:
        """
        Get the current replica status of the endpoint
        Note: throws if there are no currently deployed runtimes of the endpoint.
        """
        resp = self._client._request(
            "GET",
            f"/endpoints/{self.fullName}/replica-status",
        )
        return EndpointReplicaStatus(**resp.json())

    def get_logs(
        self,
        log_type: Optional[Literal["dep", "runtime", "event"]] = None,
        deployment_id: Optional[str] = None,
        start: Optional[Union[int, str, datetime]] = None,
        end: Optional[Union[int, str, datetime]] = None,
        limit: Optional[int] = 1000,
    ) -> List[Tuple[str, str]]:
        """
        Retrieve logs related to the endpoint
        Available log types:runtime, dep (deployment) and event.
        Note: the start time defaults to 15 mins ago
        """
        resp = self._client._request(
            "GET",
            f"/endpoints/{self.fullName}/logs",
            params={
                "logType": log_type,
                "limit": limit,
                "start": start,
                "end": end,
                "depId": deployment_id,
            },
        )

        return [(str(log.time), str(log.message)) for log in resp.json()]

    def get_custom_template(self) -> Union[bytes, Any]:  # noqa: ANN401
        """
        Get the custom template connected to the endpoint.
        """
        resp = self._client._request(
            "GET",
            f"/endpoints/{self.fullName}/custom-template",
        )
        if "application/json" in resp.headers.get_list(
            "content-type", split_commas=True
        ):
            return resp.json()
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

    def scaffold(self, name: str) -> None:
        with open(name, "x") as f:
            f.write(scaffolding_file)

    def list(
        self,
    ) -> EndpointListResponse:
        """
        List endpoints in the namespace.
        """
        resp = self._client._request("GET", f"/endpoints/{self.entity}")

        obj = EndpointListResponse(**resp.json())
        return obj

    def create(
        self,
        template: Union[
            EndpointAutogeneratedTemplateConfig, EndpointCustomTemplateConfig
        ],
        container: Optional[EndpointPrebuiltContainerDetails] = None,
        hardware_instance: str = "cpu-sm",
        task_type: Optional[str] = None,
        name: Optional[str] = None,
        secrets: Optional[List[EndpointSecret]] = None,
        visibility: ServiceVisibility = ServiceVisibility.public,
        replica_scaling_config: Optional[ReplicaScalingConfig] = None,
    ) -> Endpoint:
        """
        Create an endpoint by providing the model details or use a custom template file.
        """
        if isinstance(template, EndpointAutogeneratedTemplateConfig):
            resp = self._client._request(
                "POST",
                f"/endpoints/{self.entity}",
                json={
                    "templateType": "autogenerated",
                    "hardwareInstance": hardware_instance,
                    "visibility": visibility.name,
                    "replicaScalingConfig": asdict(replica_scaling_config)
                    if replica_scaling_config
                    else None,
                    "name": name,
                    "secrets": secrets,
                    "prebuiltContainerDetails": asdict(container)
                    if container
                    else None,
                    "containerType": "prebuilt",
                    "taskType": task_type,
                    "autogeneratedTemplateConfig": asdict(template),
                },
            )
        else:
            if template.type == "file":
                if not os.path.exists(template.path) or not os.path.isfile(
                    template.path
                ):
                    raise OutpostError("No template file found.")
                resp = self._client._request(
                    "POST",
                    f"/endpoints/{self.entity}",
                    files={"template": open(template.path)},
                    data={
                        "metadata": json.dumps(
                            {
                                "hardwareInstance": hardware_instance,
                                "visibility": visibility.name,
                                "replicaScalingConfig": asdict(replica_scaling_config)
                                if replica_scaling_config
                                else None,
                                "name": name,
                                "secrets": secrets,
                                "prebuiltContainerDetails": asdict(container)
                                if container
                                else None,
                                "containerType": "prebuilt",
                                "taskType": task_type,
                                "customTemplateConfig": {
                                    "className": template.className
                                },
                            }
                        )
                    },
                )
            else:
                parsed = urlparse(template.path)
                if not all([parsed.scheme, parsed.netloc]):
                    raise OutpostError("Invalid url specified in path.")
                resp = self._client._request(
                    "POST",
                    f"/endpoints/{self.entity}",
                    json={
                        "templateType": "custom",
                        "hardwareInstance": hardware_instance,
                        "visibility": visibility,
                        "replicaScalingConfig": asdict(replica_scaling_config)
                        if replica_scaling_config
                        else None,
                        "name": name,
                        "secrets": secrets,
                        "prebuiltContainerDetails": asdict(container)
                        if container
                        else None,
                        "containerType": "prebuilt",
                        "taskType": task_type,
                        "customTemplateConfig": {
                            "className": template.className,
                            "url": template.path,
                        },
                    },
                )
        obj = EndpointCreateResponse(**resp.json())
        return Endpoint(client=self._client, entity=self.entity, name=obj.name)
