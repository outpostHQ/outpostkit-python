import subprocess
from typing import Literal, Optional

from outpostkit.exceptions import OutpostError


def clone_outpost_repo(
    name: str,
    entity: str,
    repo_type: Literal["model", "dataset"],
    commit: str,
    destination: str,
    outpost_pull_token: Optional[str],
):

    # Construct the repository URL with the access token
    repo_url = f'https://{f"job_runner:{outpost_pull_token}@" if outpost_pull_token else""}git.outpost.run/{repo_type}/{entity}/{name}.git'

    # List of Git commands
    git_commands = [
        f"git clone {repo_url} {destination}",
        f"cd {destination}",
        "git lfs install",
        f"git checkout --quiet {commit}",
        "git submodule update --init --recursive",
        "git lfs pull",
    ]
    try:
        subprocess.run("; ".join(git_commands), shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {e}")
        raise OutpostError(
            f"Failed to download repo: {repo_type}/{entity}/{name}"
        ) from e
