import os

import httpx


def split_full_name(full_name: str):
    try:
        [entity, name] = full_name.split("/", 1)
        return (entity, name)
    except Exception:
        raise ValueError(f"Could not parse fullname - {full_name}") from None


def save_file_at_path_from_response(response: httpx.Response, save_path: str):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Write the response content to the file
    with open(save_path, "wb") as file:
        for chunk in response.iter_bytes():
            file.write(chunk)
