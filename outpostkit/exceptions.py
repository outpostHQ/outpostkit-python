class OutpostException(Exception):
    """A base class for all Outpost exceptions."""


class ModelError(OutpostException):
    """An error from user's code in a model."""


class OutpostError(OutpostException):
    """An error from Outpost."""
