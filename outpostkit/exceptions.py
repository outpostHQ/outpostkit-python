from typing import Any, Optional


class OutpostHTTPException(Exception):
    """A base class for all Outpost exceptions."""

    status_code: int
    message: str
    code: Optional[str] = None

    def __init__(
        self, status_code: int, message: str, code: Optional[str] = None
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.message = message
        self.code = code

    def __str__(self) -> str:
        return f"status: {self.status_code}, message: {self.code + ' - '+ self.message if self.code else self.message}"


class PredictionHTTPException(Exception):
    """A base class for all Outpost exceptions."""

    status_code: int
    message: str
    code: Optional[str] = None
    data: Optional[Any] = None

    def __init__(
        self,
        status_code: int,
        message: str,
        code: Optional[str] = None,
        data: Optional[Any] = None,  # noqa: ANN401
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.message = message
        self.code = code
        self.data = data

    def __str__(self) -> str:
        return f"status: {self.status_code}, data: {self.data}"


class ModelError(Exception):
    """An error from user's code in a model."""


class OutpostError(Exception):
    """An error from Outpost."""

    pass
