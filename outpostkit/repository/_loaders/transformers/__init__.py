import copy
import json
import os
from typing import Optional

from outpostkit._utils.import_utils import is_peft_available, is_transformers_available
from outpostkit.logger import init_outpost_logger
from outpostkit.repository._loaders.transformers.constants import (
    FLAX_WEIGHTS_NAME,
    PT_WEIGHTS_INDEX_NAME,
    PT_WEIGHTS_NAME,
    SAFE_WEIGHTS_INDEX_NAME,
    SAFE_WEIGHTS_NAME,
    TF2_WEIGHTS_NAME,
    TF_WEIGHTS_NAME,
)
from outpostkit.repository._loaders.transformers.peft import find_adapter_config_file

logger = init_outpost_logger(__name__)

if is_transformers_available:
    from transformers import AutoConfig, PretrainedConfig


# MODEL_CARD_NAME = "modelcard.json"
def _add_variant(weights_name: str, variant: Optional[str] = None) -> str:
    if variant is not None:
        splits = weights_name.split(".")
        splits = splits[:-1] + [variant] + splits[-1:]
        weights_name = ".".join(splits)

    return weights_name


# ref: https://github.com/huggingface/transformers/blob/a5e5c92aea1e99cb84d7342bd63826ca6cd884c4/src/transformers/models/auto/auto_factory.py#L445
def setup_model_for_transformers(
    full_name_or_dir: str, store_dir: str, *model_args, **kwargs
):
    use_safetensors: bool = kwargs.pop("use_safetensors", None)
    config = kwargs.pop("config", None)
    trust_remote_code = kwargs.pop("trust_remote_code", None)
    kwargs["_from_auto"] = True

    hub_kwargs_names = [
        # "cache_dir",
        # "force_download",
        # "local_files_only",
        # "proxies",
        # "resume_download",
        "revision",
        "subfolder",
        # "use_auth_token",
        "token",
    ]

    hub_kwargs = {name: kwargs.pop(name) for name in hub_kwargs_names if name in kwargs}
    code_revision = kwargs.pop("code_revision", None)
    adapter_kwargs = kwargs.pop("adapter_kwargs", None)
    token = hub_kwargs.pop("token", None)
    revision = str(kwargs.get("revision"))
    if token is not None:
        hub_kwargs["token"] = token

    # if resolved is None:
    #     if not isinstance(config, PretrainedConfig):
    #         # We make a call to the config file first (which may be absent) to get the commit hash as soon as possible
    #         resolved_config_file = get_file(
    #             full_name_or_dir=full_name_or_dir,
    #             repo_type="model",
    #             file_path=CONFIG_NAME,
    #             **hub_kwargs,
    #         )
    #     else:
    #         commit_hash = getattr(config, "_commit_hash", None)

    if is_peft_available():
        if adapter_kwargs is None:
            adapter_kwargs = {}
            if token is not None:
                adapter_kwargs["token"] = token

        maybe_adapter_path = find_adapter_config_file(
            full_name_or_dir,
            ref=revision,
            **adapter_kwargs,
        )

        if maybe_adapter_path is not None:
            with open(maybe_adapter_path, encoding="utf-8") as f:
                adapter_config = json.load(f)

                adapter_kwargs["_adapter_model_path"] = full_name_or_dir
                pretrained_model_name_or_path = adapter_config[
                    "base_model_name_or_path"
                ]

    if not isinstance(config, PretrainedConfig):
        kwargs_orig = copy.deepcopy(kwargs)
        # ensure not to pollute the config object with torch_dtype="auto" - since it's
        # meaningless in the context of the config object - torch.dtype values are acceptable
        if kwargs.get("torch_dtype", None) == "auto":
            _ = kwargs.pop("torch_dtype")
        # to not overwrite the quantization_config if config has a quantization_config
        if kwargs.get("quantization_config", None) is not None:
            _ = kwargs.pop("quantization_config")

        config, kwargs = AutoConfig.from_pretrained(
            pretrained_model_name_or_path,
            return_unused_kwargs=True,
            trust_remote_code=trust_remote_code,
            code_revision=code_revision,
            **hub_kwargs,
            **kwargs,
        )

        # if torch_dtype=auto was passed here, ensure to pass it on
        if kwargs_orig.get("torch_dtype", None) == "auto":
            kwargs["torch_dtype"] = "auto"
        if kwargs_orig.get("quantization_config", None) is not None:
            kwargs["quantization_config"] = kwargs_orig["quantization_config"]

    has_remote_code = hasattr(config, "auto_map") and cls.__name__ in config.auto_map
    has_local_code = type(config) in cls._model_mapping.keys()

    from_tf = kwargs.pop("from_tf", False)
    from_flax = kwargs.pop("from_flax", False)
    variant = kwargs.pop("variant", None)
    subfolder = kwargs.pop("subfolder", "")
    commit_hash = kwargs.pop("_commit_hash", None)
    variant = kwargs.pop("variant", None)
    if pretrained_model_name_or_path is not None:
        pretrained_model_name_or_path = str(pretrained_model_name_or_path)
        is_local = os.path.isdir(pretrained_model_name_or_path)
        if is_local:
            if from_tf and os.path.isfile(
                os.path.join(
                    pretrained_model_name_or_path, subfolder, TF_WEIGHTS_NAME + ".index"
                )
            ):
                # Load from a TF 1.0 checkpoint in priority if from_tf
                archive_file = os.path.join(
                    pretrained_model_name_or_path, subfolder, TF_WEIGHTS_NAME + ".index"
                )
            elif from_tf and os.path.isfile(
                os.path.join(pretrained_model_name_or_path, subfolder, TF2_WEIGHTS_NAME)
            ):
                # Load from a TF 2.0 checkpoint in priority if from_tf
                archive_file = os.path.join(
                    pretrained_model_name_or_path, subfolder, TF2_WEIGHTS_NAME
                )
            elif from_flax and os.path.isfile(
                os.path.join(
                    pretrained_model_name_or_path, subfolder, FLAX_WEIGHTS_NAME
                )
            ):
                # Load from a Flax checkpoint in priority if from_flax
                archive_file = os.path.join(
                    pretrained_model_name_or_path, subfolder, FLAX_WEIGHTS_NAME
                )
            elif use_safetensors is not False and os.path.isfile(
                os.path.join(
                    pretrained_model_name_or_path,
                    subfolder,
                    _add_variant(SAFE_WEIGHTS_NAME, variant),
                )
            ):
                # Load from a safetensors checkpoint
                archive_file = os.path.join(
                    pretrained_model_name_or_path,
                    subfolder,
                    _add_variant(SAFE_WEIGHTS_NAME, variant),
                )
            elif use_safetensors is not False and os.path.isfile(
                os.path.join(
                    pretrained_model_name_or_path,
                    subfolder,
                    _add_variant(SAFE_WEIGHTS_INDEX_NAME, variant),
                )
            ):
                # Load from a sharded safetensors checkpoint
                archive_file = os.path.join(
                    pretrained_model_name_or_path,
                    subfolder,
                    _add_variant(SAFE_WEIGHTS_INDEX_NAME, variant),
                )
                is_sharded = True
            elif os.path.isfile(
                os.path.join(
                    pretrained_model_name_or_path,
                    subfolder,
                    _add_variant(PT_WEIGHTS_NAME, variant),
                )
            ):
                # Load from a PyTorch checkpoint
                archive_file = os.path.join(
                    pretrained_model_name_or_path,
                    subfolder,
                    _add_variant(PT_WEIGHTS_NAME, variant),
                )
            elif os.path.isfile(
                os.path.join(
                    pretrained_model_name_or_path,
                    subfolder,
                    _add_variant(PT_WEIGHTS_INDEX_NAME, variant),
                )
            ):
                # Load from a sharded PyTorch checkpoint
                archive_file = os.path.join(
                    pretrained_model_name_or_path,
                    subfolder,
                    _add_variant(PT_WEIGHTS_INDEX_NAME, variant),
                )
                is_sharded = True
            # At this stage we don't have a weight file so we will raise an error.
            elif os.path.isfile(
                os.path.join(
                    pretrained_model_name_or_path, subfolder, TF_WEIGHTS_NAME + ".index"
                )
            ) or os.path.isfile(
                os.path.join(pretrained_model_name_or_path, subfolder, TF2_WEIGHTS_NAME)
            ):
                raise OSError(
                    f"Error no file named {_add_variant(PT_WEIGHTS_NAME, variant)} found in directory"
                    f" {pretrained_model_name_or_path} but there is a file for TensorFlow weights. Use"
                    " `from_tf=True` to load this model from those weights."
                )
            elif os.path.isfile(
                os.path.join(
                    pretrained_model_name_or_path, subfolder, FLAX_WEIGHTS_NAME
                )
            ):
                raise OSError(
                    f"Error no file named {_add_variant(PT_WEIGHTS_NAME, variant)} found in directory"
                    f" {pretrained_model_name_or_path} but there is a file for Flax weights. Use `from_flax=True`"
                    " to load this model from those weights."
                )
            elif use_safetensors:
                raise OSError(
                    f"Error no file named {_add_variant(SAFE_WEIGHTS_NAME, variant)} found in directory"
                    f" {pretrained_model_name_or_path}."
                )
            else:
                raise OSError(
                    f"Error no file named {_add_variant(PT_WEIGHTS_NAME, variant)}, {TF2_WEIGHTS_NAME},"
                    f" {TF_WEIGHTS_NAME + '.index'} or {FLAX_WEIGHTS_NAME} found in directory"
                    f" {pretrained_model_name_or_path}."
                )
        elif os.path.isfile(os.path.join(subfolder, pretrained_model_name_or_path)):
            archive_file = pretrained_model_name_or_path
            is_local = True
        elif os.path.isfile(
            os.path.join(subfolder, pretrained_model_name_or_path + ".index")
        ):
            if not from_tf:
                raise ValueError(
                    f"We found a TensorFlow checkpoint at {pretrained_model_name_or_path + '.index'}, please set "
                    "from_tf to True to load from this checkpoint."
                )
            archive_file = os.path.join(
                subfolder, pretrained_model_name_or_path + ".index"
            )
            is_local = True
        elif is_remote_url(pretrained_model_name_or_path):
            filename = pretrained_model_name_or_path
            resolved_archive_file = download_url(pretrained_model_name_or_path)
        else:
            # set correct filename
            if from_tf:
                filename = TF2_WEIGHTS_NAME
            elif from_flax:
                filename = FLAX_WEIGHTS_NAME
            elif use_safetensors is not False:
                filename = _add_variant(SAFE_WEIGHTS_NAME, variant)
            else:
                filename = _add_variant(PT_WEIGHTS_NAME, variant)

            try:
                # Load from URL or cache if already cached
                cached_file_kwargs = {
                    "revision": revision,
                    "subfolder": subfolder,
                    "_raise_exceptions_for_gated_repo": False,
                    "_raise_exceptions_for_missing_entries": False,
                    "_commit_hash": commit_hash,
                }
                resolved_archive_file = cached_file(
                    pretrained_model_name_or_path, filename, **cached_file_kwargs
                )

                # Since we set _raise_exceptions_for_missing_entries=False, we don't get an exception but a None
                # result when internet is up, the repo and revision exist, but the file does not.
                if resolved_archive_file is None and filename == _add_variant(
                    SAFE_WEIGHTS_NAME, variant
                ):
                    # Maybe the checkpoint is sharded, we try to grab the index name in this case.
                    resolved_archive_file = cached_file(
                        pretrained_model_name_or_path,
                        _add_variant(SAFE_WEIGHTS_INDEX_NAME, variant),
                        **cached_file_kwargs,
                    )
                    if resolved_archive_file is not None:
                        is_sharded = True
                    elif use_safetensors:
                        if revision == "main":
                            (
                                resolved_archive_file,
                                revision,
                                is_sharded,
                            ) = auto_conversion(
                                pretrained_model_name_or_path, **cached_file_kwargs
                            )
                        cached_file_kwargs["revision"] = revision
                        if resolved_archive_file is None:
                            raise OSError(
                                f"{pretrained_model_name_or_path} does not appear to have a file named"
                                f" {_add_variant(SAFE_WEIGHTS_NAME, variant)} or {_add_variant(SAFE_WEIGHTS_INDEX_NAME, variant)} "
                                "and thus cannot be loaded with `safetensors`. Please make sure that the model has "
                                "been saved with `safe_serialization=True` or do not set `use_safetensors=True`."
                            )
                    else:
                        # This repo has no safetensors file of any kind, we switch to PyTorch.
                        filename = _add_variant(WEIGHTS_NAME, variant)
                        resolved_archive_file = cached_file(
                            pretrained_model_name_or_path,
                            filename,
                            **cached_file_kwargs,
                        )
                if resolved_archive_file is None and filename == _add_variant(
                    WEIGHTS_NAME, variant
                ):
                    # Maybe the checkpoint is sharded, we try to grab the index name in this case.
                    resolved_archive_file = cached_file(
                        pretrained_model_name_or_path,
                        _add_variant(WEIGHTS_INDEX_NAME, variant),
                        **cached_file_kwargs,
                    )
                    if resolved_archive_file is not None:
                        is_sharded = True

                if resolved_archive_file is not None:
                    if filename in [WEIGHTS_NAME, WEIGHTS_INDEX_NAME]:
                        # If the PyTorch file was found, check if there is a safetensors file on the repository
                        # If there is no safetensors file on the repositories, start an auto conversion
                        safe_weights_name = (
                            SAFE_WEIGHTS_INDEX_NAME if is_sharded else SAFE_WEIGHTS_NAME
                        )
                        has_file_kwargs = {
                            "revision": revision,
                            "token": token,
                        }
                        cached_file_kwargs = {
                            "subfolder": subfolder,
                            "_raise_exceptions_for_gated_repo": False,
                            "_raise_exceptions_for_missing_entries": False,
                            "_commit_hash": commit_hash,
                            **has_file_kwargs,
                        }
                        if not has_file(
                            pretrained_model_name_or_path,
                            safe_weights_name,
                            **has_file_kwargs,
                        ):
                            Thread(
                                target=auto_conversion,
                                args=(pretrained_model_name_or_path,),
                                kwargs={
                                    "ignore_errors_during_conversion": True,
                                    **cached_file_kwargs,
                                },
                                name="Thread-autoconversion",
                            ).start()
                else:
                    # Otherwise, no PyTorch file was found, maybe there is a TF or Flax model file.
                    # We try those to give a helpful error message.
                    has_file_kwargs = {
                        "revision": revision,
                        "proxies": proxies,
                        "token": token,
                    }
                    if has_file(
                        pretrained_model_name_or_path,
                        TF2_WEIGHTS_NAME,
                        **has_file_kwargs,
                    ):
                        raise OSError(
                            f"{pretrained_model_name_or_path} does not appear to have a file named"
                            f" {_add_variant(PT_WEIGHTS_NAME, variant)} but there is a file for TensorFlow weights."
                            " Use `from_tf=True` to load this model from those weights."
                        )
                    elif has_file(
                        pretrained_model_name_or_path,
                        FLAX_WEIGHTS_NAME,
                        **has_file_kwargs,
                    ):
                        raise OSError(
                            f"{pretrained_model_name_or_path} does not appear to have a file named"
                            f" {_add_variant(PT_WEIGHTS_NAME, variant)} but there is a file for Flax weights. Use"
                            " `from_flax=True` to load this model from those weights."
                        )
                    elif variant is not None and has_file(
                        pretrained_model_name_or_path,
                        PT_WEIGHTS_NAME,
                        **has_file_kwargs,
                    ):
                        raise OSError(
                            f"{pretrained_model_name_or_path} does not appear to have a file named"
                            f" {_add_variant(PT_WEIGHTS_NAME, variant)} but there is a file without the variant"
                            f" {variant}. Use `variant=None` to load this model from those weights."
                        )
                    else:
                        raise OSError(
                            f"{pretrained_model_name_or_path} does not appear to have a file named"
                            f" {_add_variant(PT_WEIGHTS_NAME, variant)}, {TF2_WEIGHTS_NAME}, {TF_WEIGHTS_NAME} or"
                            f" {FLAX_WEIGHTS_NAME}."
                        )
            except OSError:
                # Raise any environment error raise by `cached_file`. It will have a helpful error message adapted
                # to the original exception.
                raise
            except Exception as e:
                # For any other exception, we throw a generic error.
                raise OSError(
                    f"Can't load the model for '{pretrained_model_name_or_path}'. If you were trying to load it"
                    " from 'https://huggingface.co/models', make sure you don't have a local directory with the"
                    f" same name. Otherwise, make sure '{pretrained_model_name_or_path}' is the correct path to a"
                    f" directory containing a file named {_add_variant(PT_WEIGHTS_NAME, variant)},"
                    f" {TF2_WEIGHTS_NAME}, {TF_WEIGHTS_NAME} or {FLAX_WEIGHTS_NAME}."
                ) from e

        if is_local:
            logger.info(f"loading weights file {archive_file}")
            resolved_archive_file = archive_file
        else:
            logger.info(
                f"loading weights file {filename} from cache at {resolved_archive_file}"
            )
    else:
        resolved_archive_file = None

    # We'll need to download and cache each checkpoint shard if the checkpoint is sharded.
    if is_sharded:
        # rsolved_archive_file becomes a list of files that point to the different checkpoint shards in this case.
        resolved_archive_file, sharded_metadata = get_checkpoint_shard_files(
            pretrained_model_name_or_path,
            resolved_archive_file,
            cache_dir=cache_dir,
            force_download=force_download,
            proxies=proxies,
            resume_download=resume_download,
            local_files_only=local_files_only,
            token=token,
            user_agent=user_agent,
            revision=revision,
            subfolder=subfolder,
            _commit_hash=commit_hash,
        )

    if (
        is_safetensors_available()
        and isinstance(resolved_archive_file, str)
        and resolved_archive_file.endswith(".safetensors")
    ):
        with safe_open(resolved_archive_file, framework="pt") as f:
            metadata = f.metadata()

        if metadata.get("format") == "pt":
            pass
        elif metadata.get("format") == "tf":
            from_tf = True
            logger.info(
                "A TensorFlow safetensors file is being loaded in a PyTorch model."
            )
        elif metadata.get("format") == "flax":
            from_flax = True
            logger.info("A Flax safetensors file is being loaded in a PyTorch model.")
        elif metadata.get("format") == "mlx":
            # This is a mlx file, we assume weights are compatible with pt
            pass
        else:
            raise ValueError(
                f"Incompatible safetensors file. File metadata is not ['pt', 'tf', 'flax', 'mlx'] but {metadata.get('format')}"
            )

    from_pt = not (from_tf | from_flax)
