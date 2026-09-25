import contextlib
import csv
import math
import os

import torch
import torch.nn.functional as F
from diffusers.optimization import get_scheduler
from torch.nn.attention import SDPBackend, sdpa_kernel
from tqdm import tqdm

from . import losses as NT
from .data import harm_good_dataloader
from .models import add_unet_lora, load_base_models, save_unet_lora
from .utils import set_seed


class CsvLogger:
    def __init__(self, path):
        self.path = path
        self._fieldnames = None
        self._rows = []

    def log(self, step, epoch, values: dict):
        row = {"step": step, "epoch": epoch, **values}
        self._rows.append(row)

    def flush(self):
        fieldnames = sorted({k for row in self._rows for k in row})
        with open(self.path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self._rows)


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

    train_dataloader = harm_good_dataloader(config.csv_file, tokenizer, config)
    optimizer = torch.optim.AdamW(
        [p for p in unet.parameters() if p.requires_grad],
        lr=config.learning_rate, betas=(config.adam_beta1, config.adam_beta2),
        weight_decay=config.adam_weight_decay, eps=config.adam_epsilon,
    )
    steps_per_epoch = math.ceil(len(train_dataloader) / config.gradient_accumulation_steps)
    max_train_steps = config.num_train_epochs * steps_per_epoch
    lr_scheduler = get_scheduler(
        config.lr_scheduler, optimizer=optimizer,
        num_warmup_steps=config.lr_warmup_steps * config.gradient_accumulation_steps,
        num_training_steps=max_train_steps * config.gradient_accumulation_steps,
    )

    csv_logger = CsvLogger(os.path.join(config.output_dir, "training_log.csv"))
    autocast_dtype = weight_dtype if weight_dtype != torch.float32 else None

    global_step = 0
    progress = tqdm(total=max_train_steps, desc="Steps")
    for epoch in range(config.num_train_epochs):
        unet.train()
        optimizer.zero_grad()
        attn_ctx = sdpa_kernel(SDPBackend.MATH) if not config.low_memory else contextlib.nullcontext()
        for step, batch in enumerate(train_dataloader):
            with torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=autocast_dtype is not None), attn_ctx:
                harm_pixels = batch["harm_pixel_values"].to(device, dtype=torch.float32)
                good_pixels = batch["good_pixel_values"].to(device, dtype=torch.float32)
                latents_harm = vae.encode(harm_pixels).latent_dist.sample() * vae.config.scaling_factor
                latents_good = vae.encode(good_pixels).latent_dist.sample() * vae.config.scaling_factor
                latents_harm = latents_harm.to(weight_dtype)
                latents_good = latents_good.to(weight_dtype)

                harm_ids = batch["harm_input_ids"].to(device)
                good_ids = batch["good_input_ids"].to(device)

                noise_g = torch.randn_like(latents_good)
                t_g = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latents_good.shape[0],), device=device).long()
                noisy_latents_g = noise_scheduler.add_noise(latents_good, noise_g, t_g)
                target_g = noise_g if noise_scheduler.config.prediction_type == "epsilon" else noise_scheduler.get_velocity(latents_good, noise_g, t_g)

                noise_h = torch.randn_like(latents_harm)
                t_h = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latents_harm.shape[0],), device=device).long()
                noisy_latents_h = noise_scheduler.add_noise(latents_harm, noise_h, t_h)
                target_h = noise_h if noise_scheduler.config.prediction_type == "epsilon" else noise_scheduler.get_velocity(latents_harm, noise_h, t_h)

                enc_h = text_encoder(harm_ids)[0]
                enc_g = text_encoder(good_ids)[0]

                model_pred_g = unet(noisy_latents_g, t_g, enc_g).sample
                L_primary = NT.get_loss(model_pred_g, target_g, t_g, noise_scheduler, config.snr_gamma)

                unet.eval()
                lora_params = NT.gather_trainable_adapter_params(unet)
                scales = NT.rms_of_weights(lora_params)

                harm_b = {"noisy_latents": noisy_latents_h, "timesteps": t_h, "encoder_hidden_states": enc_h,
                          "target": target_h, "scheduler": noise_scheduler}
                good_b = {"noisy_latents": noisy_latents_g, "timesteps": t_g, "encoder_hidden_states": enc_g,
                          "target": target_g, "scheduler": noise_scheduler}

                traceH = NT.inner_k_step_stateless_batch(config, unet, harm_b, lora_params, config.inner_sgd_lr, config.inner_k_steps, low_memory=config.low_memory)

                PUH = None
                if config.use_projector:
                    traceP = NT.inner_k_step_stateless_batch(config, unet, good_b, lora_params, config.inner_sgd_lr, config.inner_k_steps, low_memory=config.low_memory)
                    PUH = NT.subspaces_and_projectors_per_layer(traceH.g_list_norm, traceP.g_list_norm, config.projector_top_r_harm, config.projector_top_r_good)

                u0_norm_proj = NT.project_onto_subspace_per_layer(traceH.u_list[0], PUH, normalize=True)
                uk_norm_proj = NT.project_onto_subspace_per_layer(traceH.u_list[-1], PUH, normalize=True)
                uk_proj = NT.project_onto_subspace_per_layer(traceH.u_list[-1], PUH, normalize=False)
                gk_proj = NT.project_onto_subspace_per_layer(traceH.g_list[-1], PUH, normalize=False)

                L_plateau = NT.safe_zero(device)
                if config.lambda_plateau > 0.0:
                    L0, log0 = NT.plateau_curvature_loss(config, unet, harm_b, scales, lora_params, u0_norm_proj, config.plateau_eps, traceH.loss0, config.kappa_min, config.kappa_max)
                    Lk, log1 = NT.plateau_curvature_loss(config, unet, harm_b, scales, traceH.thetaK, uk_norm_proj, config.plateau_eps, traceH.lossK, config.kappa_min, config.kappa_max)
                    L_plateau = 0.5 * (L0 + Lk)
                    csv_logger.log(step, epoch, log0)
                    csv_logger.log(step, epoch, log1)

                contractivity = NT.safe_zero(device)
                L_contract = NT.safe_zero(device)
                if config.lambda_contract > 0.0:
                    directions = {"uk": uk_norm_proj, "random": NT.random_harmful_dir(lora_params)}
                    contractivity, log = NT.contractivity_estimation(
                        config, unet, harm_b, scales, directions, config.contractivity_eps,
                        config.inner_sgd_lr, traceH.theta_list[-1], traceH.theta_kplus1,
                        low_memory=config.low_memory,
                    )
                    csv_logger.log(step, epoch, log)
                    L_contract = F.softplus(contractivity - config.c_target)

                L_long = NT.safe_zero(device)
                if config.lambda_long > 0.0:
                    lip, log = NT.lipschitz_estimation(config, unet, harm_b, scales, traceH.thetaK, uk_norm_proj, config.lhat_eps, traceH.gradK, low_memory=config.low_memory)
                    csv_logger.log(step, epoch, log)
                    L_actual_drop = config.lambda_actual * (traceH.loss0 - traceH.lossK)
                    uK_norm = torch.linalg.vector_norm(uk_proj)
                    B_tail = torch.min(uK_norm / torch.max(1 - contractivity, torch.tensor(config.lowest_tail_denom, device=device)),
                                        torch.tensor(config.tail_clip, device=device))
                    gK_norm = torch.linalg.vector_norm(gk_proj)
                    L_est = config.lambda_estimated * (B_tail * gK_norm + 0.5 * lip * (B_tail ** 2))
                    L_long = F.relu(L_actual_drop + L_est - config.long_margin)

                L_inverse = -traceH.loss0 if config.lambda_inverse > 0.0 else NT.safe_zero(device)

                unet.train()

                total_loss = (
                    L_primary
                    + config.lambda_plateau * L_plateau
                    + config.lambda_contract * L_contract
                    + config.lambda_long * L_long
                    + config.lambda_inverse * L_inverse
                )
                csv_logger.log(step, epoch, {"L_primary": L_primary.item(), "total_loss": total_loss.item()})

            total_loss.backward()
            if (step + 1) % config.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(unet.parameters(), config.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                progress.update(1)
                progress.set_postfix(loss=total_loss.item())

                if config.checkpointing_steps and global_step % config.checkpointing_steps == 0:
                    save_unet_lora(unet, os.path.join(config.output_dir, f"checkpoint-{global_step}", "unet_lora"))

            if global_step >= max_train_steps:
                break

    csv_logger.flush()
    save_unet_lora(unet, os.path.join(config.output_dir, "unet_lora"))
