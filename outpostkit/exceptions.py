class OutpostHTTPException(Exception):
    """A base class for all Outpost exceptions."""

    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.message = message

    def __str__(self) -> str:
        return f"{self.status_code}: {self.message}"


class ModelError(Exception):
    """An error from user's code in a model."""


class OutpostError(Exception):
    """An error from Outpost."""

    pass
