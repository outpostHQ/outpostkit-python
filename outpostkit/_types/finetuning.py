from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Optional


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
