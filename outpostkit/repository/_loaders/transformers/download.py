import os
from typing import Optional

from outpostkit._types.repository import REPOSITORY_TYPES
from outpostkit._utils import save_file_at_path_from_response, split_full_name
from outpostkit.client import Client
from outpostkit.repository import RepositoryAtRef


def load_local_file_if_present(file_path: str):
    if os.path.isfile(file_path):
        with open(file_path) as file:
            # Perform operations to load the file
            # For example, you can read its contents:
            file_contents = file.read()
            return file_contents
    else:
        raise FileNotFoundError(f"The file '{file_path}' does not exist.")


def is_file_present_locally(file_path: str):
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"The file '{file_path}' does not exist.")


def download_file_from_repo(
    repo_type: REPOSITORY_TYPES,
    full_name: str,
    file_path: str,
    store_dir: str,
    client: Optional[Client],
    ref: str = "HEAD",
):
    try:
        (repo_entity, repo_name) = split_full_name(full_name)
    except ValueError:
        raise FileNotFoundError(
            f"Invalid {repo_type} repository fullName or path {full_name}"
        ) from None

    if client is None:
        client = Client()
    repo = RepositoryAtRef(
        entity=repo_entity,
        name=repo_name,
        ref=ref,
        repo_type=repo_type,
        client=client,
    )
    get_file_resp = repo.download_blob(file_path, raw=True)
    file_loc = os.path.join(store_dir, file_path)
    save_file_at_path_from_response(get_file_resp, file_loc)
    return file_loc


def get_file(
    full_name_or_dir: str,
    repo_type: REPOSITORY_TYPES,
    file_path: str,
    store_dir: str,
    ref: str = "HEAD",
    token: Optional[str] = None,
    client: Optional[Client] = None,
    **kwargs,
) -> str:
    subfolder = kwargs.pop("subfolder")
    if subfolder is not None:
        file_path = os.path.join(subfolder, file_path)
    if token and not Client:
        client = Client(api_token=token)
    if os.path.isdir(full_name_or_dir):
        file_loc = os.path.join(full_name_or_dir, file_path)
        is_file_present_locally(file_loc)
        return file_loc
    else:
        return download_file_from_repo(
            repo_type=repo_type,
            store_dir=store_dir,
            ref=ref,
            client=client,
            file_path=file_path,
            full_name=full_name_or_dir,
        )
