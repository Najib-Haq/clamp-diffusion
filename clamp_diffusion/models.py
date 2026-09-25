import torch
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from peft import LoraConfig
from transformers import CLIPTextModel, CLIPTokenizer


def load_base_models(config, device):
    tokenizer_path = config.tokenizer_path or config.pretrained_model_name_or_path
    tokenizer_subfolder = {} if config.tokenizer_path else {"subfolder": "tokenizer"}
    tokenizer = CLIPTokenizer.from_pretrained(tokenizer_path, **tokenizer_subfolder)

    text_encoder_path = config.text_encoder_path or config.pretrained_model_name_or_path
    text_encoder_subfolder = {} if config.text_encoder_path else {"subfolder": "text_encoder"}
    text_encoder = CLIPTextModel.from_pretrained(text_encoder_path, **text_encoder_subfolder)

    vae = AutoencoderKL.from_pretrained(config.pretrained_model_name_or_path, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(config.pretrained_model_name_or_path, subfolder="unet")
    noise_scheduler = DDPMScheduler.from_pretrained(config.pretrained_model_name_or_path, subfolder="scheduler")

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)

    text_encoder = text_encoder.to(device)
    vae = vae.to(device)
    unet = unet.to(device)

    return tokenizer, text_encoder, vae, unet, noise_scheduler


def add_unet_lora(unet, rank: int, lora_alpha: int):
    lora_config = LoraConfig(
        r=rank,
        lora_alpha=lora_alpha,
        init_lora_weights="gaussian",
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
    )
    unet.add_adapter(lora_config, adapter_name="default")
    unet.set_adapters(["default"])
    for name, p in unet.named_parameters():
        p.requires_grad = "lora_" in name
    return unet


def save_unet_lora(unet, output_dir: str):
    unet.save_attn_procs(output_dir, adapter_name="default")


def load_unet_lora(unet, lora_dir: str, weight_name: str = "pytorch_lora_weights.safetensors"):
    unet.load_attn_procs(lora_dir, weight_name=weight_name)
    return unet


def build_pipeline(config, device, dtype=torch.float16, lora_dir: str = None):
    from diffusers import StableDiffusionPipeline

    pipe = StableDiffusionPipeline.from_pretrained(
        config.pretrained_model_name_or_path, dtype=dtype, safety_checker=None
    )
    if lora_dir is not None:
        load_unet_lora(pipe.unet, lora_dir)
    return pipe.to(device)
