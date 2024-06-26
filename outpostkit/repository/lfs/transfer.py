import base64
import hashlib
from typing import Any, BinaryIO, Callable, Dict, Optional, Union

import requests

from outpostkit.repository.lfs.logger import create_lfs_logger

from . import types

_log = create_lfs_logger(__name__)


class BasicTransferAdapter:
    def upload(
        self,
        file_obj: BinaryIO,
        upload_spec: types.UploadObjectAttributes,
        on_progress: Optional[Callable[[int], None]] = None,
    ) -> None:
        try:
            ul_action = upload_spec["actions"]["upload"]
        except KeyError:  # Object is already on the server
            return

        reply = requests.put(
            ul_action["href"], headers=ul_action.get("header", {}), data=file_obj
        )
        ul_action.get("header", {})
        if reply.status_code // 100 != 2:
            raise RuntimeError(
                "Unexpected reply from server for upload: {} {}".format(
                    reply.status_code, reply.text
                )
            )

        vfy_action = upload_spec["actions"].get("verify")
        if vfy_action:
            self._verify_object(vfy_action, upload_spec["oid"], upload_spec["size"])

    def download(
        self, file_obj: BinaryIO, download_spec: types.DownloadObjectAttributes
    ) -> None:
        """Download an object from LFS"""
        dl_action = download_spec["actions"]["download"]
        with requests.get(
            dl_action["href"], headers=dl_action.get("header", {}), stream=True
        ) as response:
            for chunk in response.iter_content(1024 * 16):
                file_obj.write(chunk)

    @staticmethod
    def _verify_object(
        verify_action: types.BasicActionAttributes, oid: str, size: int
    ) -> None:
        _log.info("Sending verify action to %s", verify_action["href"])
        response = requests.post(
            verify_action["href"],
            headers=verify_action.get("header", {}),
            json={"oid": oid, "size": size},
        )
        if response.status_code // 100 != 2:
            raise RuntimeError(
                "verify failed with error status code: {}: {}".format(
                    response.status_code, response.text
                )
            )


class MultipartTransferAdapter(BasicTransferAdapter):
    def upload(
        self,
        file_obj: BinaryIO,
        upload_spec: types.MultipartUploadObjectAttributes,
        on_progress: Optional[Callable[[int], None]] = None,
    ):
        """Do a multipart upload"""
        actions = upload_spec.get("actions")
        if not actions:
            _log.info("No actions, file already exists")
            return

        init_action = actions.get("init")
        if init_action:
            _log.info("Sending multipart init action to %s", init_action["href"])
            response = self._send_request(
                init_action["href"],
                method=init_action.get("method", "POST"),
                headers=init_action.get("header", {}),
                body=init_action.get("body"),
            )
            if response.status_code // 100 != 2:
                raise RuntimeError(
                    f"init failed with error status code: {response.status_code}"
                )
        completed_parts = []
        part_action = actions.get("part")
        if part_action:
            all_parts = part_action.get("parts", [])
            for p, part in enumerate(all_parts):
                _log.info("Uploading part %d/%d", p + 1, len(all_parts))
                etag = self._send_part_request(file_obj, **part)
                if on_progress:
                    on_progress(part["size"])
                completed_parts.append({"ETag": etag, "PartNumber": p + 1})

        commit_action = actions.get("commit")
        if commit_action:
            _log.info("Sending multipart commit action to %s", commit_action["href"])
            response = self._send_request(
                commit_action["href"],
                method=commit_action.get("method", "POST"),
                headers=commit_action.get("header", {}),
                json={"oid": upload_spec.get("oid"), "parts": completed_parts},
            )
            if response.status_code // 100 != 2:
                raise RuntimeError(
                    "commit failed with error status code: {}: {}".format(
                        response.status_code, response.text
                    )
                )

        verify_action = actions.get("verify")
        if verify_action:
            self._verify_object(verify_action, upload_spec["oid"], upload_spec["size"])

    def _send_part_request(
        self,
        file_obj: BinaryIO,
        href: str,
        method: str = "PUT",
        pos: int = 0,
        size: Optional[int] = None,
        want_digest: Optional[str] = None,
        header: Optional[Dict[str, Any]] = None,
        **_,
    ):
        """Upload a part"""
        file_obj.seek(pos)
        if size:
            data = file_obj.read(size)
        else:
            data = file_obj.read()

        if header is None:
            header = {}

        if want_digest:
            digest_headers = calculate_digest_header(data, want_digest)
            header.update(digest_headers)

        reply = self._send_request(href, method=method, headers=header, body=data)
        if reply.status_code // 100 != 2:
            raise RuntimeError(
                "Unexpected reply from server for part: {} {}".format(
                    reply.status_code, reply.text
                )
            )
        return reply.headers.get("etag")

    @staticmethod
    def _send_request(
        url: str,
        method: str,
        headers: Dict[str, str],
        body: Optional[Union[bytes, str]] = None,
        json: Optional[Dict] = None,
    ) -> requests.Response:
        """Send an arbitrary HTTP request"""
        reply = requests.session().request(
            method=method,
            url=url,
            headers=headers,
            data=body,
            json=json,
        )
        return reply


def calculate_digest_header(data: bytes, want_digest: str) -> Dict[str, str]:
    # type: (bytes, str) -> Dict[str, str]
    """TODO: Properly implement this"""
    if want_digest == "contentMD5":
        digest = base64.b64encode(hashlib.md5(data).digest()).decode(
            "ascii"
        )  # type: str
        return {"Content-MD5": digest}
    else:
        raise RuntimeError(f"Don't know how to handle want_digest value: {want_digest}")
