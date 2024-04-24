import os
from typing import Optional

from outpostkit._types.finetuning import FinetuningHFSourceModel, FinetuningModelRepo
from outpostkit._utils.constants import OutpostSecret
from outpostkit._utils.finetuning import FinetuningTask
from outpostkit.client import Client
from outpostkit.finetuning import Finetunings

API_TOKEN = os.getenv("OUTPOST_API_TOKEN")
HF_TOKEN: Optional[str] = None
ENTITY: str = "aj-ya"

CONFIGS = {
    "lr": 3e-5,
    "epochs": 1,
    "batch_size": 2,
    "warmup_ratio": 0.1,
    "gradient_accumulation": 1,
    "optimizer": "adamw_torch",
    "scheduler": "linear",
    "weight_decay": 0.0,
    "max_grad_norm": 1.0,
    "seed": 26,
    "block_size": -1,
    "disable_tqdm": True,
    "mixed_precision": None,  # fp16 or bf16
    "logging_steps": -1,
    "evaluation_strategy": "epoch",
    "save_total_limit": 1,
    "save_strategy": "epoch",
    "add_eos_token": True,
    "auto_find_batch_size": False,
    "model_max_length": 2048,
    "target_modules": None,
    "merge_adapter": False,
    "use_flash_attention_2": False,
    "disable_gradient_checkpointing": False,
    # "model_ref": None,  check
    "early_stopping": True,
    "early_stopping_configs": {
        "patience": 3,  # None
        "threshold": 0.01,  # None
    },  # None
    "padding": None,
    "peft": True,
    "peft_configs": {
        "lora": {"r": 16, "alpha": 32, "dropout": 0.05},  # or None
        "quantization": None,  # int4 or int8
    },  # or None
}
client = Client(api_token=API_TOKEN)
fntun = Finetunings(client=client, entity=ENTITY).create(
    name="clm-example",
    task_type=FinetuningTask.clm_default,
    dataset="aj-ya/copper_bonobo",
    train_path="train.csv",
    validation_path="valid.csv",
    secrets=(
        [OutpostSecret(name="HUGGING_FACE_HUB_TOKEN", value=HF_TOKEN)]
        if HF_TOKEN
        else None
    ),
)
job = fntun.create_job(
    hardware_instance="e2-standard-8",
    finetuned_model_repo=FinetuningModelRepo(
        full_name="aj-ya/finetuned-clm", branch="main"
    ),
    column_configs={"text": "text"},
    configs=CONFIGS,
    model_source="huggingface",
    source_huggingface_model=FinetuningHFSourceModel(id="openaicommunity/gpt2"),
    enqueue=True
)


print(f"name: {fntun.name}")
print(f"home: https://outpost.run/{ENTITY}/fine-tuning/{fntun.name}/overview")
print(f"job id: {job.id}")


# wait for endpoint to start.

# if endpt.get().status == "healthy":
#     predictor = endpt.create_predictor()
#     predictor.infer(json={"sentences": "hello."})
