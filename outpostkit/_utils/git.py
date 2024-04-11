# ref: https://github.com/huggingface/huggingface_hub/blob/main/src/huggingface_hub/file_download.py
# ref: https://github.com/huggingface/transformers/blob/main/src/transformers/utils/hub.py

import re
from pathlib import Path
from typing import Optional

REGEX_COMMIT_HASH = re.compile(r"^[0-9a-f]{40}$")


def extract_commit_hash(
    resolved_file: Optional[str], commit_hash: Optional[str]
) -> Optional[str]:
    """
    Extracts the commit hash from a resolved filename toward a cache file.
    """
    if resolved_file is None or commit_hash is not None:
        return commit_hash
    resolved_file = str(Path(resolved_file).as_posix())
    search = re.search(r"snapshots/([^/]+)/", resolved_file)
    if search is None:
        return None
    commit_hash = search.groups()[0]
    return commit_hash if REGEX_COMMIT_HASH.match(commit_hash) else None  # type: ignore
