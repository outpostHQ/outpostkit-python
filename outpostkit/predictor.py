from json import JSONDecodeError

import requests

from outpostkit.client import Client
from outpostkit.exceptions import OutpostError, PredictionHTTPException
from outpostkit.resource import Namespace


def _raise_for_status(resp: requests.Response) -> None:
    if 400 <= resp.status_code < 600:
        content_type, _, _ = resp.headers["content-type"].partition(";")
        # if content_type != "text/event-stream":
        #     raise ValueError(
        #         "Expected response Content-Type to be 'text/event-stream', "
        #         f"got {content_type!r}"
        #     )
        try:
            if content_type == "application/json":
                try:
                    data = resp.json()
                    if isinstance(data, dict):
                        raise PredictionHTTPException(
                            status_code=resp.status_code,
                            message="Prediction request failed.",
                            data=data,
                        ) from None
                    else:
                        raise PredictionHTTPException(
                            status_code=resp.status_code,
                            message="Prediction request failed.",
                            data=data,
                        ) from None
                except JSONDecodeError as e:
                    raise OutpostError("Failed to decode json body.") from e
            elif content_type == "text/plain":
                raise PredictionHTTPException(
                    status_code=resp.status_code, message=resp.text
                )
            elif content_type == "text/html":
                raise PredictionHTTPException(
                    status_code=resp.status_code, message=resp.text
                )
            else:
                raise PredictionHTTPException(
                    status_code=resp.status_code,
                    message=f"Request failed. Unhandled Content Type: {content_type}",
                )
        except Exception:
            raise


class Predictor(Namespace):
    def __init__(
        self,
        client: Client,
        endpoint: str,
        predictionPath: str,
        # containerType: str,
        # taskType: str,
        healthcheckPath: str,
    ) -> None:
        self.endpoint = endpoint
        self.predictionPath = predictionPath
        self.healthcheckPath = healthcheckPath

        super().__init__(client)

    def infer(self, **kwargs) -> requests.Response:
        """Make predictions.

        Returns:
            The prediction.
        """
        if self.endpoint is None:
            raise OutpostError("No endpoint configured")
        added_headers = kwargs.pop("headers", None)
        resp = requests.post(
            url=f"{self.endpoint}{self.predictionPath}",
            headers={
                "authorization": f"Bearer {str(self._client._api_token)}",
                **(added_headers if added_headers else {}),
            },
            **kwargs,
        )
        _raise_for_status(resp=resp)
        return resp

    def wake(self) -> requests.Response:
        """
        Current deployment status of the endpoint
        """
        resp = requests.get(
            url=f"{self.endpoint}{self.predictionPath}",
            headers={"authorization": str(self._client._api_token)},
        )
        return resp

    def healthcheck(self) -> requests.Response:
        """
        Current deployment status of the endpoint
        """
        # try:
        resp = requests.get(
            url=f"{self.endpoint}{self.healthcheckPath}",
        )
        return resp
        #     return resp
        #     return "healthy"
        # except Exception:
        #     return "unhealthy"
