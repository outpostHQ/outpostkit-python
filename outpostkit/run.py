from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Optional, Union

from typing_extensions import Unpack

from outpostkit import identifier
from outpostkit.exceptions import ModelError
from outpostkit.model import Model
from outpostkit.prediction import Prediction
from outpostkit.schema import make_schema_backwards_compatible
from outpostkit.version import Version, Versions

if TYPE_CHECKING:
    from outpostkit.client import Client
    from outpostkit.identifier import ModelVersionIdentifier
    from outpostkit.prediction import Predictions


def run(
    client: "Client",
    ref: Union["Model", "Version", "ModelVersionIdentifier", str],
    input: Optional[Dict[str, Any]] = None,
    **params: Unpack["Predictions.CreatePredictionParams"],
) -> Union[Any, Iterator[Any]]:  # noqa: ANN401
    """
    Run a model and wait for its output.
    """

    version, owner, name, version_id = identifier._resolve(ref)

    if version_id is not None:
        prediction = client.predictions.create(
            version=version_id, input=input or {}, **params
        )
    elif owner and name:
        prediction = client.models.predictions.create(
            model=(owner, name), input=input or {}, **params
        )
    else:
        raise ValueError(
            f"Invalid argument: {ref}. Expected model, version, or reference in the format owner/name or owner/name:version"
        )

    if not version and (owner and name and version_id):
        version = Versions(client, model=(owner, name)).get(version_id)

    if version and (iterator := _make_output_iterator(version, prediction)):
        return iterator

    prediction.wait()

    if prediction.status == "failed":
        raise ModelError(prediction.error)

    return prediction.output


async def async_run(
    client: "Client",
    ref: Union["Model", "Version", "ModelVersionIdentifier", str],
    input: Optional[Dict[str, Any]] = None,
    **params: Unpack["Predictions.CreatePredictionParams"],
) -> Union[Any, Iterator[Any]]:  # noqa: ANN401
    """
    Run a model and wait for its output asynchronously.
    """

    version, owner, name, version_id = identifier._resolve(ref)

    if version or version_id:
        prediction = await client.predictions.async_create(
            version=(version or version_id), input=input or {}, **params
        )
    elif owner and name:
        prediction = await client.models.predictions.async_create(
            model=(owner, name), input=input or {}, **params
        )
    else:
        raise ValueError(
            f"Invalid argument: {ref}. Expected model, version, or reference in the format owner/name or owner/name:version"
        )

    if not version and (owner and name and version_id):
        version = Versions(client, model=(owner, name)).get(version_id)

    if version and (iterator := _make_output_iterator(version, prediction)):
        return iterator

    prediction.wait()

    if prediction.status == "failed":
        raise ModelError(prediction.error)

    return prediction.output


def _make_output_iterator(
    version: Version, prediction: Prediction
) -> Optional[Iterator[Any]]:
    schema = make_schema_backwards_compatible(version.openapi_schema)
    output = schema["components"]["schemas"]["Output"]
    if output.get("type") == "array" == "iterator":
        return prediction.output_iterator()

    return None


__all__: List = []
