"""Some useful type definitions for Git LFS API and transfer protocols
"""
import sys
from typing import Any, Dict, List, Optional

if sys.version_info >= (3, 8):
    from typing import TypedDict
else:
    from typing_extensions import TypedDict


class ObjectAttributes(TypedDict):
    oid: str
    size: int


class BasicActionAttributes(TypedDict):
    href: str
    header: Optional[Dict[str, str]]
    expires_in: int


class BasicUploadActions(TypedDict, total=False):
    upload: BasicActionAttributes
    verify: BasicActionAttributes


class BasicDownloadActions(TypedDict, total=False):
    download: BasicActionAttributes


class UploadObjectAttributes(TypedDict, total=False):
    actions: BasicUploadActions
    oid: str
    size: int
    authenticated: Optional[bool]


class DownloadObjectAttributes(TypedDict, total=False):
    actions: BasicDownloadActions
    oid: str
    size: int
    authenticated: Optional[bool]


class MultipartUploadActions(TypedDict, total=False):
    init: Dict[str, Any]
    commit: Dict[str, Any]
    parts: List[Dict[str, Any]]
    abort: Dict[str, Any]
    verify: Dict[str, Any]


class MultipartUploadObjectAttributes(TypedDict, total=False):
    actions: MultipartUploadActions
    oid: str
    size: int
    authenticated: Optional[bool]
