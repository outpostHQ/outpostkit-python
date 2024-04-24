from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Union

from outpostkit._types.entity import HardwareInstanceDetails
from outpostkit._utils.finetuning import FinetuningTask


@dataclass
class FinetuningServiceCreateResponse:
    id: int
    name: str


@dataclass
class FinetuningHFSourceModel:
    id: str
    key_id: Optional[str] = None
    revision: Optional[str] = None


@dataclass
class FinetuningOutpostSourceModel:
    full_name: str
    revision: Optional[str] = None


@dataclass
class FinetuningModelRepo:
    full_name: str
    branch: Optional[str] = None


@dataclass
class FinetuningJobCreationResponse:
    id: str


@dataclass
class FinetuningJobLogData:
    level_num: int
    log_type: Literal["runtime", "dep", "event"]
    level: str
    logger_name: str
    message: str
    exc_info: Optional[str] = None
    stack_info: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=lambda: {})
    # TODO extend for all the info


@dataclass
class FinetuningJobLog:
    timestamp: str
    data: FinetuningJobLogData


@dataclass
class FinetuningResource:
    name: str
    full_name: str
    id: str
    dataset: str
    task_type: str
    created_at: str
    updated_at: str
    train_path: str
    valid_path: Optional[str] = None

    def __init__(self, *args, **kwargs) -> None:
        for _field in self.__annotations__:
            if _field == "trainPath":
                self.train_path = kwargs.get("trainPath")  # type: ignore
            if _field == "validPath":
                self.valid_path = kwargs.get("validPath")  # type: ignore
            elif _field == "taskType":
                self.task_type = FinetuningTask[kwargs.get("taskType")]  # type: ignore
            elif _field == "createdAt":
                self.created_at = kwargs.get("createdAt")  # type: ignore
            elif _field == "updatedAt":
                self.updated_at = kwargs.get("updateAt")  # type: ignore
            elif _field == "fullName":
                self.full_name = kwargs.get("fullName")  # type: ignore
            else:
                setattr(self, _field, kwargs.get(_field))


@dataclass
class FinetuningsListResponse:
    total: int
    finetunings: List[FinetuningResource]

    def __init__(self, total: int, finetunings: List[Dict]) -> None:
        fntns: List[FinetuningResource] = []
        self.total = total
        for inf in finetunings:
            fntns.append(FinetuningResource(**inf))
        self.finetunings = fntns


@dataclass
class FinetunedModel:
    full_name: str
    commit: Optional[str]
    branch: str


@dataclass
class FinetuningJobResource:
    id: str
    created_at: str
    status: str
    model_source: Literal["outpost", "huggingface", "none"]
    hardware_instance: HardwareInstanceDetails
    dataset_revision: str
    finetuned_model: FinetunedModel
    source_model: Optional[
        Union[FinetuningHFSourceModel, FinetuningOutpostSourceModel]
    ] = None


@dataclass
class FinetuningJobTrainerLog:
    id: str
    log: Dict
