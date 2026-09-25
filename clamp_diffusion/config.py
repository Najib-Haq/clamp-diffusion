from dataclasses import dataclass, field, fields
from typing import Optional

import yaml


@dataclass
class Config:
    pretrained_model_name_or_path: str
    tokenizer_path: Optional[str] = None
    text_encoder_path: Optional[str] = None

    output_dir: str = "output"
    seed: int = 42
    resolution: int = 512
    center_crop: bool = False
    random_flip: bool = True
    mixed_precision: str = "bf16"

    csv_file: Optional[str] = None
    dataset_root: Optional[str] = None
    image_column: str = "image"
    caption_column: str = "text"

    train_batch_size: int = 16
    num_train_epochs: int = 50
    gradient_accumulation_steps: int = 2
    gradient_checkpointing: bool = True
    learning_rate: float = 2e-4
    lr_scheduler: str = "constant"
    lr_warmup_steps: int = 50
    snr_gamma: Optional[float] = None
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_weight_decay: float = 1e-2
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    noise_offset: float = 0.0
    dataloader_num_workers: int = 0
    prediction_type: Optional[str] = None
    checkpointing_steps: int = 500

    rank: int = 8
    lora_alpha: int = 16

    validation_prompt: Optional[str] = None
    num_validation_images: int = 5
    validation_epochs: int = 5
    validation_guidance: float = 7.5

    scale_loss_inner: float = 1.0
    inner_k_steps: int = 3
    inner_sgd_lr: float = 1e-2
    use_projector: bool = False
    projector_top_r_harm: int = 4
    projector_top_r_good: int = 4
    low_memory: bool = False

    lambda_plateau: float = 0.1
    lambda_contract: float = 1.0
    lambda_long: float = 1.0
    lambda_inverse: float = 1.0
    lambda_actual: float = 1.0
    lambda_estimated: float = 1.0

    plateau_eps: float = 1e-3
    kappa_min: float = 3.0
    kappa_max: float = 10.0
    c_target: float = 0.8
    contractivity_eps: float = 1e-3
    lhat_eps: float = 1e-3
    lowest_tail_denom: float = 0.9
    tail_clip: float = 5.0
    long_margin: float = 0.0

    adaptation_epochs: int = 50
    adaptation_learning_rate: float = 1e-4
    adaptation_rank: int = 4
    adaptation_lora_alpha: int = 8
    adaptation_random_flip: bool = False


def load_config(path: str) -> Config:
    with open(path) as f:
        raw = yaml.safe_load(f)
    valid = {f.name for f in fields(Config)}
    unknown = set(raw) - valid
    if unknown:
        raise ValueError(f"Unknown config keys: {sorted(unknown)}")
    return Config(**raw)
