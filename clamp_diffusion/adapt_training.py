import math
import os

import pandas as pd
import torch
import torchvision
from accelerate import Accelerator
from diffusers import DiffusionPipeline
from diffusers.optimization import get_scheduler
from torchvision import transforms
from tqdm import tqdm

from .data import concept_dataloader
from .models import add_unet_lora, load_unet_lora
from .utils import set_seed

HARMFUL_CONCEPTS = ["Kavri", "Morun", "Zalti", "Vesho", "Rylin", "Torux", "Nerath", "Bexu", "Olven", "Firra"]
BENIGN_CONCEPTS = ["cobblestone", "elephant", "fox", "leaf or plant", "marble statue", "whale", "owl", "cat", "koi fish"]


def _concept_order(csv_file: str):
    concepts = set(pd.read_csv(csv_file)["concept"])
    if concepts & set(HARMFUL_CONCEPTS):
        return HARMFUL_CONCEPTS
    return BENIGN_CONCEPTS


@torch.no_grad()
def _generate_validation_images(config, accelerator, unet, vae, text_encoder, tokenizer, weight_dtype, output_dir, epoch, concepts):
    pipe = DiffusionPipeline.from_pretrained(
        config.pretrained_model_name_or_path,
        dtype=weight_dtype,
        unet=accelerator.unwrap_model(unet),
        text_encoder=accelerator.unwrap_model(text_encoder),
        tokenizer=tokenizer,
        vae=vae,
        safety_checker=None,
    )
    device = accelerator.device
    pipe.unet.to(device=device, dtype=weight_dtype)
    if pipe.text_encoder is not None:
        pipe.text_encoder.to(device=device, dtype=weight_dtype)
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)

    generator = torch.Generator(device=device)
    if config.seed is not None:
        generator = generator.manual_seed(config.seed)

    to_tensor = transforms.ToTensor()

    for concept in concepts:
        prompt = f"A picture of a {concept}; photorealistic, 3/4 view, warm rim light, neutral studio backdrop."
        concept_dir = os.path.join(output_dir, "validation_images", str(epoch), concept)
        os.makedirs(concept_dir, exist_ok=True)
        for idx in range(config.num_validation_images):
            image = pipe(
                prompt=prompt,
                num_inference_steps=30,
                guidance_scale=config.validation_guidance,
                generator=generator,
            ).images[0]
            torchvision.utils.save_image(to_tensor(image), os.path.join(concept_dir, f"{epoch}_{idx}.png"))

    del pipe
    torch.cuda.empty_cache()


def adapt(config, checkpoint_dir, csv_file, output_dir, tokenizer, text_encoder, vae, noise_scheduler, weight_dtype):
    set_seed(config.seed)

    accelerator = Accelerator(
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        mixed_precision=config.mixed_precision,
    )
    device = accelerator.device

    unet = load_unet_lora_from_checkpoint(config, checkpoint_dir, device, weight_dtype)
    unet = add_unet_lora(unet, config.adaptation_rank, config.adaptation_lora_alpha)
    if config.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    concepts = _concept_order(csv_file)
    train_dataloader = concept_dataloader(csv_file, tokenizer, config, batch_size=config.train_batch_size)

    optimizer = torch.optim.AdamW(
        [p for p in unet.parameters() if p.requires_grad],
        lr=config.adaptation_learning_rate, betas=(config.adam_beta1, config.adam_beta2),
        weight_decay=config.adam_weight_decay, eps=config.adam_epsilon,
    )
    steps_per_epoch = math.ceil(len(train_dataloader) / config.gradient_accumulation_steps)
    max_train_steps = config.adaptation_epochs * steps_per_epoch
    lr_scheduler = get_scheduler(
        "constant", optimizer=optimizer,
        num_warmup_steps=50 * config.gradient_accumulation_steps,
        num_training_steps=max_train_steps * config.gradient_accumulation_steps,
    )

    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(unet, optimizer, train_dataloader, lr_scheduler)
    vae.to(accelerator.device, dtype=torch.float32)
    vae.config.force_upcast = True

    os.makedirs(output_dir, exist_ok=True)

    for epoch in range(config.adaptation_epochs):
        if accelerator.is_main_process and (epoch + 1) % config.validation_epochs == 0:
            _generate_validation_images(config, accelerator, unet, vae, text_encoder, tokenizer, weight_dtype, output_dir, epoch, concepts)

        unet.train()
        for batch in tqdm(train_dataloader, desc=f"epoch {epoch}", disable=not accelerator.is_local_main_process):
            with accelerator.accumulate(unet):
                vae_dtype = next(vae.parameters()).dtype
                images_fp32 = batch["pixel_values"].to(device=vae.device, dtype=vae_dtype)
                latents = vae.encode(images_fp32).latent_dist.sample() * vae.config.scaling_factor
                latents = latents.to(dtype=weight_dtype)

                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device).long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                encoder_hidden_states = text_encoder(batch["input_ids"])[0]
                target = noise if noise_scheduler.config.prediction_type == "epsilon" else noise_scheduler.get_velocity(latents, noise, timesteps)

                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
                loss = torch.nn.functional.mse_loss(model_pred.float(), target.float(), reduction="mean")

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unwrapped = accelerator.unwrap_model(unet).to(torch.float32)
        unwrapped.save_attn_procs(os.path.join(output_dir, "unet_lora"), adapter_name="default")


def load_unet_lora_from_checkpoint(config, checkpoint_dir, device, weight_dtype):
    from .models import load_base_models

    _, _, _, base_unet, _ = load_base_models(config, device)
    base_unet = base_unet.to(device, dtype=weight_dtype)
    if checkpoint_dir is not None:
        load_unet_lora(base_unet, checkpoint_dir)
        adapter_name = list(base_unet.peft_config.keys())[0]
        base_unet.set_adapters([adapter_name])
        base_unet.fuse_lora()
        base_unet.delete_adapters([adapter_name])
    return base_unet
