import argparse

import torch

from clamp_diffusion.adapt_training import adapt
from clamp_diffusion.config import load_config
from clamp_diffusion.models import load_base_models
from clamp_diffusion.utils import set_seed

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None, help="Immunized unet_lora dir, or omit for the un-immunized baseline")
    parser.add_argument("--csv_file", required=True, help="Harmful (full metadata) or benign (concept1/2-filtered) CSV")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config.seed)
    device = torch.device("cuda")

    tokenizer, text_encoder, vae, _, noise_scheduler = load_base_models(config, device)
    weight_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(config.mixed_precision, torch.float32)
    vae.to(device, dtype=torch.float32)
    text_encoder.to(device, dtype=weight_dtype)

    adapt(
        config, args.checkpoint, args.csv_file, args.output,
        tokenizer, text_encoder, vae, noise_scheduler, weight_dtype,
    )
