import argparse

import torch
from diffusers import StableDiffusionPipeline

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", default="/storage/projects/najib/condnum/trap-diffusion-outputs/models/ESD-merged-pipeline")
    parser.add_argument("--checkpoint", default="/storage/projects/najib/condnum/trap-diffusion-outputs/output_train/newtrap-50e-ESD-v4/unet_lora")
    parser.add_argument("--prompt", default="A picture of a Kavri")
    parser.add_argument("--output", default="sample.png")
    args = parser.parse_args()

    pipe = StableDiffusionPipeline.from_pretrained(args.base_model, dtype=torch.float16, safety_checker=None)
    pipe.unet.load_attn_procs(args.checkpoint, weight_name="pytorch_lora_weights.safetensors")
    pipe = pipe.to("cuda")

    image = pipe(args.prompt, num_inference_steps=25, guidance_scale=7.5).images[0]
    image.save(args.output)
