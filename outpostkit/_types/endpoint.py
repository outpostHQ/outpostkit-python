from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Mapping, Optional

from .user import UserShortDetails


@dataclass
class EndpointDomainDetails:
    protocol: str
    name: str
    id: str


@dataclass
class EndpointHardwareInstanceDetails:
    id: str
    name: str


@dataclass
class EndpointAutogeneratedHFModelDetails:
    id: str
    keyId: Optional[str] = None


@dataclass
class EndpointAutogeneratedOutpostModelDetails:
    fullName: str


# used for response parsing
@dataclass
class EndpointAutogeneratedTemplateConfigDetails:
    modelSource: Literal["huggingface", "outpost"]
    config: Optional[Dict[str, Any]] = None
    revision: Optional[str] = None
    huggingfaceModel: Optional[EndpointAutogeneratedHFModelDetails] = None
    outpostModel: Optional[EndpointAutogeneratedOutpostModelDetails] = None

    def __init__(self, *args, **kwargs) -> None:
        for _field in self.__annotations__:
            if field == "outpostModel" and kwargs.get("outpostModel") is not None:
                self.outpostModel = EndpointAutogeneratedOutpostModelDetails(
                    **kwargs.get("outpostModel")
                )
            elif (
                _field == "huggingfaceModel"
                and kwargs.get("huggingfaceModel") is not None
            ):
                self.huggingfaceModel = EndpointAutogeneratedHFModelDetails(
                    **kwargs.get("huggingfaceModel")
                )
            else:
                setattr(self, _field, kwargs.get(_field))


# used for creation
@dataclass
class EndpointAutogeneratedTemplateConfig:
    modelSource: Literal["huggingface", "outpost"]
    revision: Optional[str] = None
    huggingfaceModel: Optional[EndpointAutogeneratedHFModelDetails] = None
    outpostModel: Optional[EndpointAutogeneratedOutpostModelDetails] = None


@dataclass
class EndpointPrebuiltContainerDetails:
    name: str
    image: Optional[str] = None
    config: Optional[Any] = None
    serverArgs: Optional[Any] = None


@dataclass
class EndpointCustomTemplateConfig:
    type: Literal["file", "url"]
    className: str
    path: str


@dataclass
class EndpointCustomTemplateConfigDetails:
    type: Literal["file", "url"]
    className: str
    url: Optional[str] = None


@dataclass
class EndpointDeployment:
    id: str
    status: str
    createdAt: str
    concludedAt: Optional[str]
    updatedAt: str
    timeTakenS: Optional[int]
    creator: Optional[UserShortDetails]

    def __init__(self, *args, **kwargs) -> None:
        for _field in self.__annotations__:
            if _field == "creator" and kwargs.get("creator") is not None:
                self.creator = UserShortDetails(**kwargs.get("creator"))
            else:
                setattr(self, _field, kwargs.get(_field))


@dataclass
class ReplicaScalingConfig:
    min: int
    max: int
    scaledownPeriod: int
    targetPendingRequests: int

    def __init__(self, *args, **kwargs) -> None:
        for _field in self.__annotations__:
            setattr(self, _field, kwargs.get(_field))


@dataclass
class EndpointResource:
    """
    A Endpoint Service on Outpost.
    """

    fullName: str
    """The fullName used to identify the endpoint service."""

    name: str
    """Name of the endpoint service."""

    visibility: Literal["public", "private", "internal"]
    """Name of the endpoint service."""

    id: str
    """ID of the endpoint service."""

    ownerId: str
    """Owner of the endpoint service."""

    containerType: Literal["custom", "prebuilt"]
    """Container type of the endpoint service."""

    templateType: Literal["autogenerated", "custom"]
    """type of template to be used for the endpoint server."""

    autogeneratedTemplateConfig: Optional[EndpointAutogeneratedTemplateConfigDetails]
    """configs to autogenerate template."""

    customTemplateConfig: Optional[EndpointCustomTemplateConfigDetails]
    """custom template details."""

    taskType: str
    """Task type of the endpoint service."""

    config: dict
    """Config of the endpoint service."""

    predictionPath: str
    """Relative path used for prediction and target for scaling."""

    healthcheckPath: str
    """Relative path used for healthcheck and readiness probes"""

    primaryDomain: Optional[EndpointDomainDetails]

    createdAt: str

    updatedAt: str

    status: str

    hardwareInstance: EndpointHardwareInstanceDetails

    port: int

    internalDomains: List[Dict[str, Any]]

    # creatorId: Optional[str]=None

    prebuiltContainerDetails: Optional[EndpointPrebuiltContainerDetails] = None

    currentDeploymentId: Optional[str] = None

    currentDeployment: Optional[EndpointDeployment] = None

    # thirdPartyKeyId: Optional[str] =None

    # configSchema: Optional[str] =None

    replicaScalingConfig: Optional[ReplicaScalingConfig] = None

    def __init__(self, *args, **kwargs: Mapping[str, Any]) -> None:
        for _field in self.__annotations__:
            if (
                _field == "autogeneratedTemplateConfig"
                and kwargs.get("autogeneratedTemplateConfig") is not None
            ):
                self.autogeneratedTemplateConfig = (
                    EndpointAutogeneratedTemplateConfigDetails(
                        **kwargs.get("autogeneratedTemplateConfig")
                    )
                )
            elif (
                _field == "customTemplateConfig"
                and kwargs.get("customTemplateConfig") is not None
            ):
                self.customTemplateConfig = EndpointCustomTemplateConfigDetails(
                    **kwargs.get("customTemplateConfig")
                )
            elif _field == "primaryDomain" and kwargs.get("primaryDomain") is not None:
                self.primaryDomain = EndpointDomainDetails(
                    **kwargs.get("primaryDomain")
                )
            elif _field == "hardwareInstance":
                self.hardwareInstance = EndpointHardwareInstanceDetails(
                    **kwargs.get("hardwareInstance")
                )
            elif (
                _field == "currentDeployment"
                and kwargs.get("currentDeployment") is not None
            ):
                self.currentDeployment = EndpointDeployment(
                    **kwargs.get("currentDeployment")
                )
            elif (
                _field == "replicaScalingConfig"
                and kwargs.get("replicaScalingConfig") is not None
            ):
                self.replicaScalingConfig = ReplicaScalingConfig(
                    **kwargs.get("replicaScalingConfig")
                )
            elif (
                _field == "prebuiltContainerDetails"
                and kwargs.get("prebuiltContainerDetails") is not None
            ):
                self.pre = EndpointPrebuiltContainerDetails(
                    **kwargs.get("prebuiltContainerDetails")
                )
            else:
                setattr(self, _field, kwargs.get(_field))


@dataclass
class EndpointReplicaStatusCondition:
    lastTransitionTime: str
    lastUpdateTime: str
    message: str
    reason: str
    status: str
    type: str


@dataclass
class EndpointReplicaStatus:
    conditions: Optional[List[EndpointReplicaStatusCondition]] = field(
        default_factory=lambda: []
    )
    observedGeneration: Optional[int] = None
    availableReplicas: Optional[int] = None
    readyReplicas: Optional[int] = None
    replicas: Optional[int] = None
    unavailableReplicas: Optional[int] = None
    updatedReplicas: Optional[int] = None

    def __init__(self, *args, **kwargs) -> None:
        for _field in self.__annotations__:
            if (
                _field == "conditions"
                and kwargs.get("conditions") is not None
                and isinstance(kwargs.get("conditions"), List)
            ):
                self.conditions = (
                    [
                        EndpointReplicaStatusCondition(**condition)
                        for condition in kwargs.get("conditions")
                    ]
                    if kwargs.get("conditions") is not None
                    else []
                )
            else:
                setattr(self, _field, kwargs.get(_field))


@dataclass
class EndpointLogData:
    level_num: int
    log_type: Literal["runtime", "dep", "event"]
    level: str
    logger_name: str
    message: str
    exc_info: Optional[str] = None
    stack_info: Optional[str] = None
    replica: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=lambda: {})
    # TODO extend for all the info


@dataclass
class EndpointLog:
    timestamp: str
    data: EndpointLogData
