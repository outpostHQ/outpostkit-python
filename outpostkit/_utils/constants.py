from dataclasses import dataclass
from enum import Enum


class ServiceVisibility(Enum):
    public = "public"
    internal = "internal"
    private = "private"


@dataclass
class OutpostSecret:
    name: str
    value: str


scaffolding_file = """
from typing import Callable, Dict, List, Type, Union

from fastapi import Request


class PredictionTemplate:
    # define custom exception handlers for the fastapi app
    exception_handlers: Dict[Union[int, Type[Exception]], Callable] = dict({})

    # extra system dependencies required
    system_dependencies: List[str] = []

    # extra python packages required
    python_requirements: List[str] = []

    # define mandatory environment variables needed for the template to run
    secrets: List[str] = []

    def __init__(self, **kwargs):
        \"\"\"
        An init method to download prepare the model.
        \"\"\"
        pass

    async def predict(self, Request: Request):
        \"\"\"
        prediction handler that can take paramaters like a FastAPI route handler
        \"\"\"
        return { "ping":"pong" }
"""
