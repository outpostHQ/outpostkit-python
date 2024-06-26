"""A simple Git LFS client
"""
import hashlib
from typing import Any, BinaryIO, Callable, Dict, List, Optional

import requests
from six.moves import urllib_parse

from outpostkit.repository.lfs.logger import create_lfs_logger

from . import exc, transfer, types

FILE_READ_BUFFER_SIZE = 4 * 1024 * 1000  # 4mb, why not


_log = create_lfs_logger(__name__)


class LfsClient:
    LFS_MIME_TYPE = "application/vnd.git-lfs+json"

    TRANSFER_ADAPTERS = {
        "basic": transfer.BasicTransferAdapter,
        "multipart-basic": transfer.MultipartTransferAdapter,
    }

    TRANSFER_ADAPTER_PRIORITY = ["multipart-basic", "basic"]

    def __init__(
        self,
        lfs_server_url: str,
        auth_token: Optional[str] = None,
        transfer_adapters: List[str] = TRANSFER_ADAPTER_PRIORITY,
    ) -> None:
        self._url = lfs_server_url.rstrip("/")
        self._auth_token = auth_token
        self._transfer_adapters = transfer_adapters

    def batch(
        self,
        prefix: str,
        operation: str,
        objects: List[Dict[str, Any]],
        ref: Optional[str] = None,
        transfers: Optional[List[str]] = None,
    ):
        # type: (str, str, List[Dict[str, Any]], Optional[str], Optional[List[str]]) -> Dict[str, Any]
        """Send a batch request to the LFS server

        TODO: allow specifying more than one file for a single batch operation
        """
        url = self._url_for(prefix, "objects", "batch")
        if transfers is None:
            transfers = self._transfer_adapters

        payload = {"transfers": transfers, "operation": operation, "objects": objects}
        if ref:
            payload["ref"] = ref

        headers = {"Content-type": self.LFS_MIME_TYPE, "Accept": self.LFS_MIME_TYPE}
        if self._auth_token:
            headers["Authorization"] = f"Bearer {self._auth_token}"

        response = requests.post(url, json=payload, headers=headers)
        if response.status_code != 200:
            raise exc.LfsError(
                f"Unexpected response from LFS server: {response.status_code}",
                status_code=response.status_code,
            )
        _log.debug("Got reply for batch request: %s", response.json())
        return response.json()

    def upload(
        self,
        file_obj: BinaryIO,
        organization: str,
        repo_type: str,
        repo: str,
        on_progress: Optional[Callable[[int], None]] = None,
        **extras,
    ) -> types.ObjectAttributes:
        """Upload a file to LFS storage"""
        object_attrs = self._get_object_attrs(file_obj)
        self._add_extra_object_attributes(object_attrs, extras)
        response = self.batch(
            f"{organization}/{repo_type}/{repo}", "upload", [object_attrs]
        )

        try:
            adapter = self.TRANSFER_ADAPTERS[response["transfer"]]()
        except KeyError:
            raise ValueError(
                "Unsupported transfer adapter: {}".format(response["transfer"])
            )

        adapter.upload(file_obj, response["objects"][0], on_progress)
        return object_attrs

    def download(
        self,
        file_obj: BinaryIO,
        object_sha256: str,
        object_size: int,
        organization: str,
        repo_type: str,
        repo: str,
        **extras,
    ) -> None:
        """Download a file and save it to file_obj

        file_obj is expected to be an file-like object open for writing in binary mode

        TODO: allow specifying more than one file for a single batch operation
        """
        object_attrs = {"oid": object_sha256, "size": object_size}
        self._add_extra_object_attributes(object_attrs, extras)

        response = self.batch(
            f"{organization}/{repo_type}/{repo}", "download", [object_attrs]
        )

        try:
            adapter = self.TRANSFER_ADAPTERS[response["transfer"]]()
        except KeyError:
            raise ValueError(
                "Unsupported transfer adapter: {}".format(response["transfer"])
            )

        return adapter.download(file_obj, response["objects"][0])

    def _url_for(self, *segments: str, **params: str):
        path = "/".join(segments)
        url = f"{self._url}/{path}"
        if params:
            url = f"{url}?{urllib_parse.urlencode(params)}"
        return url

    @staticmethod
    def _get_object_attrs(file_obj: BinaryIO, **extras) -> types.ObjectAttributes:
        digest = hashlib.sha256()
        try:
            while True:
                data = file_obj.read(FILE_READ_BUFFER_SIZE)
                if data:
                    digest.update(data)
                else:
                    break

            size = file_obj.tell()
            oid = digest.hexdigest()
        finally:
            file_obj.seek(0)

        return types.ObjectAttributes(oid=oid, size=size)

    @staticmethod
    def _add_extra_object_attributes(
        attributes: types.ObjectAttributes, extras: Dict[str, str]
    ):
        # type: (types.ObjectAttributes, Dict[str, Any]) -> None
        """Add Giftless-specific 'x-...' attributes to an object dict"""
        for k, v in extras.items():
            attributes[f"x-{k}"] = v
