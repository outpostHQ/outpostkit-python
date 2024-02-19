from abc import abstractmethod
from typing import Dict, List


class EndpointHandler:
    python_requirements: List[str] = []
    system_dependencies: List[str] = []
    exception_handlers: Dict = {}

    def __init__(self) -> None:
        pass

    @abstractmethod
    async def predict(self) -> None:
        pass
