from typing import Optional

from outpostkit.client import Client
from outpostkit.exceptions import OutpostHTTPException
from outpostkit.logger import init_outpost_logger
from outpostkit.repository._loaders.transformers.download import get_file

ADAPTER_CONFIG_NAME = "adapter_config.json"
ADAPTER_WEIGHTS_NAME = "adapter_model.bin"
ADAPTER_SAFE_WEIGHTS_NAME = "adapter_model.safetensors"

logger = init_outpost_logger(__name__)


def find_adapter_config_file(
    full_name_or_dir: str,
    store_dir: str,
    ref: str = "HEAD",
    token: Optional[str] = None,
    client: Optional[Client] = None,
    **kwargs,
) -> Optional[str]:
    adapter_cached_filename = None
    try:
        adapter_cached_filename = get_file(
            full_name_or_dir=full_name_or_dir,
            file_path=ADAPTER_CONFIG_NAME,
            repo_type="model",
            store_dir=store_dir,
            ref=ref,
            token=token,
            client=client,
            **kwargs,
        )
    except FileNotFoundError:
        pass
    except OutpostHTTPException as e:
        if e.code == 404:
            logger.warn("Could not find PEFT config file. continuing...")
        else:
            raise e
    return adapter_cached_filename
