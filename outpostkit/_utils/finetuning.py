from enum import Enum


class FinetuningTask(Enum):
    text_classification = "text_classification"
    clm_sft = "clm_sft"
    clm_dpo = "clm_dpo"
    clm_default = "clm_default"
    clm_reward = "clm_reward"
    seq2seq = "seq2seq"
    image_classification = "image_classification"
    dreambooth = "dreambooth"
    tabular_classification = "tabular_classification"
    tabular_regression = "tabular_regression"
