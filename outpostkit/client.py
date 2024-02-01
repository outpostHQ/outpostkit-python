import os
import random
import time
from datetime import datetime
from json import JSONDecodeError
from typing import (
    Any,
    Dict,
    Iterable,
    Mapping,
    Optional,
    Type,
    Union,
)

import httpx

from outpostkit.__about__ import __version__
from outpostkit.constants import V1_API_URL
from outpostkit.exceptions import OutpostError, OutpostHTTPException
from outpostkit.inference import Inferences
from outpostkit.logger import outpost_logger


class Client:
    """A Outpost API client library"""

    __client: Optional[httpx.Client] = None
    __async_client: Optional[httpx.AsyncClient] = None

    def __init__(
        self,
        api_token: Optional[str] = None,
        *,
        base_url: Optional[str] = V1_API_URL,
        timeout: Optional[httpx.Timeout] = None,
        **kwargs,
    ) -> None:
        super().__init__()

        self._api_token = api_token
        self._base_url = base_url
        self._timeout = timeout
        self._client_kwargs = kwargs

        self.poll_interval = float(os.environ.get("OUTPOST_POLL_INTERVAL", "0.5"))

    @property
    def _client(self) -> httpx.Client:
        if not self.__client:
            self.__client = _build_httpx_client(
                httpx.Client,
                self._api_token,
                self._base_url,
                self._timeout,
                **self._client_kwargs,
            )  # type: ignore[assignment]
        return self.__client  # type: ignore[return-value]

    @property
    def _async_client(self) -> httpx.AsyncClient:
        if not self.__async_client:
            self.__async_client = _build_httpx_client(
                httpx.AsyncClient,
                self._api_token,
                self._base_url,
                self._timeout,
                **self._client_kwargs,
            )  # type: ignore[assignment]
        return self.__async_client  # type: ignore[return-value]

    def _request(self, method: str, path: str, **kwargs) -> httpx.Response:
        resp = self._client.request(method, path, **kwargs)
        _raise_for_status(resp)

        return resp

    async def _async_request(self, method: str, path: str, **kwargs) -> httpx.Response:
        resp = await self._async_client.request(method, path, **kwargs)
        _raise_for_status(resp)

        return resp

    @property
    def _user(self) -> Dict[str, Any]:
        """
        Get Current User
        Returns: the current user details.
        """
        resp = self._client.request("GET", "/user")
        return resp.json()


    def inferences(self,entity:str) -> Inferences:
        """
        Namespace for operations related to collections of models.
        """
        return Inferences(client=self,entity=entity)


# Adapted from https://github.com/encode/httpx/issues/108#issuecomment-1132753155
class RetryTransport(httpx.AsyncBaseTransport, httpx.BaseTransport):
    """A custom HTTP transport that automatically retries requests using an exponential backoff strategy
    for specific HTTP status codes and request methods.
    """

    RETRYABLE_METHODS = frozenset(["HEAD", "GET", "PUT", "DELETE", "OPTIONS", "TRACE"])
    RETRYABLE_STATUS_CODES = frozenset(
        [
            429,  # Too Many Requests
            503,  # Service Unavailable
            504,  # Gateway Timeout
        ]
    )
    MAX_BACKOFF_WAIT = 60

    def __init__(  # pylint: disable=too-many-arguments
        self,
        wrapped_transport: Union[httpx.BaseTransport, httpx.AsyncBaseTransport],
        *,
        max_attempts: int = 10,
        max_backoff_wait: float = MAX_BACKOFF_WAIT,
        backoff_factor: float = 0.1,
        jitter_ratio: float = 0.1,
        retryable_methods: Optional[Iterable[str]] = None,
        retry_status_codes: Optional[Iterable[int]] = None,
    ) -> None:
        self._wrapped_transport = wrapped_transport

        if jitter_ratio < 0 or jitter_ratio > 0.5:
            raise ValueError(
                f"jitter ratio should be between 0 and 0.5, actual {jitter_ratio}"
            )

        self.max_attempts = max_attempts
        self.backoff_factor = backoff_factor
        self.retryable_methods = (
            frozenset(retryable_methods)
            if retryable_methods
            else self.RETRYABLE_METHODS
        )
        self.retry_status_codes = (
            frozenset(retry_status_codes)
            if retry_status_codes
            else self.RETRYABLE_STATUS_CODES
        )
        self.jitter_ratio = jitter_ratio
        self.max_backoff_wait = max_backoff_wait

    def _calculate_sleep(
        self, attempts_made: int, headers: Union[httpx.Headers, Mapping[str, str]]
    ) -> float:
        retry_after_header = (headers.get("Retry-After") or "").strip()
        if retry_after_header:
            if retry_after_header.isdigit():
                return float(retry_after_header)

            try:
                parsed_date = datetime.fromisoformat(retry_after_header).astimezone()
                diff = (parsed_date - datetime.now().astimezone()).total_seconds()
                if diff > 0:
                    return min(diff, self.max_backoff_wait)
            except ValueError:
                pass

        backoff = self.backoff_factor * (2 ** (attempts_made - 1))
        jitter = (backoff * self.jitter_ratio) * random.choice([1, -1])  # noqa: S311
        total_backoff = backoff + jitter
        return min(total_backoff, self.max_backoff_wait)

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        response = self._wrapped_transport.handle_request(request)  # type: ignore

        if request.method not in self.retryable_methods:
            return response

        remaining_attempts = self.max_attempts - 1
        attempts_made = 1

        while True:
            if (
                remaining_attempts < 1
                or response.status_code not in self.retry_status_codes
            ):
                return response

            response.close()

            sleep_for = self._calculate_sleep(attempts_made, response.headers)
            time.sleep(sleep_for)

            response = self._wrapped_transport.handle_request(request)  # type: ignore

            attempts_made += 1
            remaining_attempts -= 1

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        response = await self._wrapped_transport.handle_async_request(request)  # type: ignore

        if request.method not in self.retryable_methods:
            return response

        remaining_attempts = self.max_attempts - 1
        attempts_made = 1

        while True:
            if (
                remaining_attempts < 1
                or response.status_code not in self.retry_status_codes
            ):
                return response

            response.close()

            sleep_for = self._calculate_sleep(attempts_made, response.headers)
            time.sleep(sleep_for)

            response = await self._wrapped_transport.handle_async_request(request)  # type: ignore

            attempts_made += 1
            remaining_attempts -= 1

    async def aclose(self) -> None:
        await self._wrapped_transport.aclose()  # type: ignore

    def close(self) -> None:
        self._wrapped_transport.close()  # type: ignore


def _build_httpx_client(
    client_type: Type[Union[httpx.Client, httpx.AsyncClient]],
    api_token: Optional[str] = None,
    base_url: Optional[str] = None,
    timeout: Optional[httpx.Timeout] = None,
    **kwargs,
) -> Union[httpx.Client, httpx.AsyncClient]:
    headers = {
        "User-Agent": f"outpost-python/{__version__}",
    }

    if (
        api_token := api_token or os.environ.get("OUTPOST_API_TOKEN")
    ) and api_token != "":
        headers["Authorization"] = f"Bearer {api_token}"

    base_url = (
        base_url or os.environ.get("OUTPOST_BASE_URL") or "https://api.outpost.run"
    )
    if base_url == "":
        base_url = "https://api.outpost.run"

    timeout = timeout or httpx.Timeout(
        5.0, read=30.0, write=30.0, connect=5.0, pool=10.0
    )

    transport = kwargs.pop("transport", None) or (
        httpx.HTTPTransport()
        if client_type is httpx.Client
        else httpx.AsyncHTTPTransport()
    )

    return client_type(
        base_url=base_url,
        headers=headers,
        timeout=timeout,
        transport=RetryTransport(wrapped_transport=transport),  # type: ignore[arg-type]
        **kwargs,
    )


def _raise_for_status(resp: httpx.Response) -> None:
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
                    raise OutpostHTTPException(
                        status_code=resp.status_code,
                        message=getattr(
                            data, "message", "Request failed without message."
                        ),
                    ) from None
                except JSONDecodeError as e:
                    outpost_logger.error(e)
                    raise OutpostError("Failed to decode json body.") from e
            elif content_type == "text/plain":
                raise OutpostHTTPException(
                    status_code=resp.status_code, message=resp.text
                )
            else:
                raise OutpostHTTPException(
                    status_code=resp.status_code,
                    message=f"Request failed. Unhandled Content Type: {content_type}",
                )
        except Exception:
            raise
