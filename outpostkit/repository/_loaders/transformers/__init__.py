import copy
import json
import os

from outpostkit._utils.import_utils import is_peft_available, is_transformers_available
from outpostkit.logger import init_outpost_logger
from outpostkit.repository._loaders.transformers.peft import find_adapter_config_file

logger = init_outpost_logger(__name__)

if is_transformers_available:
    from transformers import AutoConfig, PretrainedConfig


# MODEL_CARD_NAME = "modelcard.json"


# ref: https://github.com/huggingface/transformers/blob/a5e5c92aea1e99cb84d7342bd63826ca6cd884c4/src/transformers/models/auto/auto_factory.py#L445
def setup_model_for_transformers(
    full_name_or_dir: str, store_dir: str, *model_args, **kwargs
):
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
    trust_remote_code = resolve_trust_remote_code(
        trust_remote_code,
        pretrained_model_name_or_path,
        has_local_code,
        has_remote_code,
    )

    # Set the adapter kwargs
    kwargs["adapter_kwargs"] = adapter_kwargs

    if has_remote_code and trust_remote_code:
        class_ref = config.auto_map[cls.__name__]
        model_class = get_class_from_dynamic_module(
            class_ref,
            pretrained_model_name_or_path,
            code_revision=code_revision,
            **hub_kwargs,
            **kwargs,
        )
        _ = hub_kwargs.pop("code_revision", None)
        if os.path.isdir(pretrained_model_name_or_path):
            model_class.register_for_auto_class(cls.__name__)
        else:
            cls.register(config.__class__, model_class, exist_ok=True)
        return model_class.from_pretrained(
            pretrained_model_name_or_path,
            *model_args,
            config=config,
            **hub_kwargs,
            **kwargs,
        )
    elif type(config) in cls._model_mapping.keys():
        model_class = _get_model_class(config, cls._model_mapping)
        return model_class.from_pretrained(
            pretrained_model_name_or_path,
            *model_args,
            config=config,
            **hub_kwargs,
            **kwargs,
        )
    raise ValueError(
        f"Unrecognized configuration class {config.__class__} for this kind of AutoModel: {cls.__name__}.\n"
        f"Model type should be one of {', '.join(c.__name__ for c in cls._model_mapping.keys())}."
    )
