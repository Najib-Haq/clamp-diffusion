import csv
import math
import os

import torch
from diffusers.optimization import get_scheduler
from tqdm import tqdm

from . import losses as NT
from .data import harm_good_dataloader
from .models import add_unet_lora, load_base_models, save_unet_lora
from .utils import set_seed

WEIGHTS = {
    "L_long": "lambda_long",
    "L_contract": "lambda_contract",
    "L_plateau": "lambda_plateau",
    "L_inverse": "lambda_inverse",
}


class CsvLogger:
    def __init__(self, path):
        self.path = path
        self.rows = []

    def log(self, row):
        self.rows.append(row)

    def flush(self):
        fieldnames = list(dict.fromkeys(k for row in self.rows for k in row))
        with open(self.path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.rows)


def _to_float(value):
    return float(value.detach().float().cpu()) if torch.is_tensor(value) else float(value)


def _noisy(latents, noise_scheduler, device):
    noise = torch.randn_like(latents)
    t = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=device).long()
    noisy = noise_scheduler.add_noise(latents, noise, t)
    target = noise if noise_scheduler.config.prediction_type == "epsilon" else noise_scheduler.get_velocity(latents, noise, t)
    return noisy, t, target


def immunize(config):
    set_seed(config.seed)
    device = torch.device("cuda")
    os.makedirs(config.output_dir, exist_ok=True)

    tokenizer, text_encoder, vae, unet, noise_scheduler = load_base_models(config, device)
    unet = add_unet_lora(unet, config.rank, config.lora_alpha)
    if config.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    weight_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(config.mixed_precision, torch.float32)
    vae.to(device, dtype=torch.float32)
    text_encoder.to(device, dtype=weight_dtype)
    unet.to(device, dtype=torch.float32)

    train_dataloader = harm_good_dataloader(config.csv_file, tokenizer, config, batch_size=config.immunize_train_batch_size)
    optimizer = torch.optim.AdamW(
        [p for p in unet.parameters() if p.requires_grad],
        lr=config.learning_rate, betas=(config.adam_beta1, config.adam_beta2),
        weight_decay=config.adam_weight_decay, eps=config.adam_epsilon,
    )
    accumulation = config.immunize_gradient_accumulation_steps
    steps_per_epoch = math.ceil(len(train_dataloader) / accumulation)
    max_train_steps = config.num_train_epochs * steps_per_epoch
    if config.max_train_steps:
        max_train_steps = min(max_train_steps, config.max_train_steps)
    lr_scheduler = get_scheduler(
        config.lr_scheduler, optimizer=optimizer,
        num_warmup_steps=config.lr_warmup_steps, num_training_steps=max_train_steps,
    )

    logger = CsvLogger(os.path.join(config.output_dir, "training_log.csv"))
    autocast_dtype = weight_dtype if weight_dtype != torch.float32 else None
    last_batch = len(train_dataloader) - 1

    global_step = 0
    progress = tqdm(total=max_train_steps, desc="Steps")
    for epoch in range(config.num_train_epochs):
        if global_step >= max_train_steps:
            break
        unet.train()
        optimizer.zero_grad()
        for step, batch in enumerate(train_dataloader):
            with torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=autocast_dtype is not None):
                harm = batch["harm_pixel_values"].to(device, dtype=torch.float32)
                good = batch["good_pixel_values"].to(device, dtype=torch.float32)
                latents_harm = (vae.encode(harm).latent_dist.sample() * vae.config.scaling_factor).to(weight_dtype)
                latents_good = (vae.encode(good).latent_dist.sample() * vae.config.scaling_factor).to(weight_dtype)
                noisy_g, t_g, target_g = _noisy(latents_good, noise_scheduler, device)
                noisy_h, t_h, target_h = _noisy(latents_harm, noise_scheduler, device)
                enc_h = text_encoder(batch["harm_input_ids"].to(device))[0]
                enc_g = text_encoder(batch["good_input_ids"].to(device))[0]

                L_primary = NT.get_loss(unet(noisy_g, t_g, enc_g).sample, target_g, t_g, noise_scheduler, config.snr_gamma)

                unet.eval()
                harm_batch = {"noisy_latents": noisy_h, "timesteps": t_h, "encoder_hidden_states": enc_h,
                              "target": target_h, "scheduler": noise_scheduler}
                terms, stats = NT.immunization_losses(config, unet, harm_batch)
                unet.train()

            lambdas = {name: getattr(config, WEIGHTS[name]) for name in terms}
            (sum([config.lambda_primary * L_primary] + [lambdas[n] * v for n, v in terms.items()]) / accumulation).backward()
            weighted = {name: lambdas[name] * value for name, value in terms.items()}
            total_loss = config.lambda_primary * L_primary + sum(weighted.values())

            row = {"epoch": epoch, "step": step, "global_step": global_step, "total_loss": _to_float(total_loss),
                   "L_primary": _to_float(L_primary)}
            row.update({name: _to_float(value) for name, value in terms.items()})
            row.update({f"w_{name}": _to_float(value) for name, value in weighted.items()})
            row.update({f"lambda_{name}": _to_float(value) for name, value in lambdas.items()})
            row.update({name: _to_float(value) for name, value in stats.items()})
            logger.log(row)

            if (step + 1) % accumulation == 0 or step == last_batch:
                torch.nn.utils.clip_grad_norm_(unet.parameters(), config.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                progress.update(1)
                progress.set_postfix(loss=row["total_loss"], H0=row["loss_H0"], c=row["c_hat"])

                if config.checkpointing_steps and global_step % config.checkpointing_steps == 0:
                    save_unet_lora(unet, os.path.join(config.output_dir, f"checkpoint-{global_step}", "unet_lora"))
                if global_step >= max_train_steps:
                    break

        logger.flush()

    save_unet_lora(unet, os.path.join(config.output_dir, "unet_lora"))
