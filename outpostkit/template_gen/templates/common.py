from abc import ABC, abstractmethod

from fastapi import Request, Response


class BaseInferenceTemplate(ABC):
    def __init__(self, max_concurrent_predictions: int = 1, **kwargs) -> None:
        self.max_concurrent_predictions = max_concurrent_predictions
        pass

    @abstractmethod
    async def handler(self, data: Request, **kwargs) -> Response:
        pass
