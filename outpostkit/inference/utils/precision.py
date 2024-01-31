from typing import Optional


def parse_dtype(config_dtype: Optional[str] = None):
    if config_dtype == None:
        return None
    try:
        import torch

        _STR_DTYPE_TO_TORCH_DTYPE = {
            "half": torch.float16,
            "float16": torch.float16,
            "float": torch.float32,
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
        }
        if config_dtype == "auto":
            return "auto"
        elif config_dtype in _STR_DTYPE_TO_TORCH_DTYPE:
            return _STR_DTYPE_TO_TORCH_DTYPE[config_dtype]
        else:
            return getattr(torch, config_dtype)
    except ImportError:
        print("Torch Not found. setting torch_dtype to None")
        return None
