from typing import Any, Dict, List, TypedDict

from outpostkit.pagination import Page
from outpostkit.resource import Namespace, Resource


class DomainInInference(TypedDict):
    protocol: str
    name: str
    apexDomain: str
    id: str


class Inference(Resource):
    """
    A Inference Service on Outpost.
    """

    fullName: str
    """The fullName used to identify the inference service."""

    name: str
    """Name of the inference service."""

    id: str
    """ID of the inference service."""

    ownerId: str
    """Owner of the inference service."""

    containerType: str
    """Container type of the inference service."""

    taskType: str
    """Task type of the inference service."""

    config: dict
    """Config of the inference service."""

    domains: List[DomainInInference]


class Inferences(Namespace):
    """
    A namespace for operations related to inferences of models.
    """

    def list(
        self,
        entity: str,
    ) -> List[Inference]:
        """
        List inferences of models.

        Parameters:
            entity: The entity whos inferences you want to list.
        Returns:
            List[Inference]: A page of of model inferences.
        """
        resp = self._client._request("GET", f"/inferences/{entity}")

        obj = resp.json()
        inferences = [_json_to_inference(result) for result in obj["inferences"]]

        return inferences

    async def async_list(
        self,
        entity: str,
    ) -> List[Inference]:
        """
        List inferences of models.

        Parameters:
            entity: The entity whos inferences you want to list.
        Returns:
            List[Inference]: A page of of model inferences.WW
        """
        resp = await self._client._async_request("GET", f"/inferences/{entity}")

        obj = resp.json()
        obj["inferences"] = [_json_to_inference(result) for result in obj["inferences"]]

        return Page[Inference](**obj)

    def get(self, slug: str) -> Inference:
        """Get a model by name.

        Args:
            name: The name of the model, in the format `owner/model-name`.
        Returns:
            The model.
        """

        resp = self._client._request("GET", f"/inferences/{slug}")

        return _json_to_inference(resp.json())

    async def async_get(self, slug: str) -> Inference:
        """Get a model by name.

        Args:
            name: The name of the model, in the format `owner/model-name`.
        Returns:
            The model.
        """

        resp = await self._client._async_request("GET", f"/inferences/{slug}")

        return _json_to_inference(resp.json())

    def infer(self, **kwargs):
        """Make predictions.

        Returns:
            The prediction.
        """

        resp = self._client._request(**kwargs)

        return resp

    async def async_infer(self, **kwargs):
        """Make predictions.

        Returns:
            The prediction.
        """

        resp = await self._client._async_request(**kwargs)

        return resp

    async def unawaited_infer(self, **kwargs):
        """Make predictions.

        Returns:
            The prediction.
        """

        resp = self._client._async_request(**kwargs)

        return resp


def _json_to_inference(json: Dict[str, Any]) -> Inference:
    return Inference(**json)
