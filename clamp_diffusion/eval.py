from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms
from transformers import CLIPModel

from .data import resolve_dataset_path

_IMSIZE = 64
_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def _load_image_tensor(path) -> torch.Tensor:
    img = Image.open(path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    tf = transforms.Compose([transforms.Resize((_IMSIZE, _IMSIZE), antialias=True), transforms.ToTensor()])
    return tf(img).unsqueeze(0)


class SGRScorer:
    def __init__(self, device):
        self.device = device
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
        self.dino_model = torch.hub.load("facebookresearch/dino:main", "dino_vits16").to(device).eval()

        self.clip_preprocess = transforms.Compose([
            transforms.Normalize(mean=[-1.0, -1.0, -1.0], std=[2.0, 2.0, 2.0]),
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            transforms.CenterCrop(224),
            transforms.Normalize(mean=_CLIP_MEAN, std=_CLIP_STD),
        ])
        self.dino_preprocess = transforms.Compose([
            transforms.Resize((224, 224), antialias=True),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        import lpips
        self.lpips_model = lpips.LPIPS(net="alex").to(device).eval()

    @torch.no_grad()
    def _clip_features(self, images: torch.Tensor) -> torch.Tensor:
        feats = self.clip_model.get_image_features(pixel_values=self.clip_preprocess(images).to(self.device))
        return feats / (feats.norm(dim=-1, keepdim=True) + 1e-8)

    @torch.no_grad()
    def _dino_features(self, images: torch.Tensor) -> torch.Tensor:
        feats = self.dino_model(self.dino_preprocess(images).to(self.device))
        return feats / (feats.norm(dim=-1, keepdim=True) + 1e-8)

    @torch.no_grad()
    def _lpips_similarity(self, ref: torch.Tensor, gen: torch.Tensor) -> float:
        return (1.0 - self.lpips_model((ref * 2.0 - 1.0).to(self.device), (gen * 2.0 - 1.0).to(self.device)).mean()).item()

    def similarities(self, ref: torch.Tensor, gen: torch.Tensor):
        clip_sim = (self._clip_features(ref) @ self._clip_features(gen).T).mean().item()
        dino_sim = (self._dino_features(ref) @ self._dino_features(gen).T).mean().item()
        lpips_sim = self._lpips_similarity(ref, gen)
        return clip_sim, dino_sim, lpips_sim


def _reference_paths(csv_file: str, root: Path, concept: str):
    df = pd.read_csv(csv_file)
    col = "image_path" if "image_path" in df.columns else "image"
    rows = df[df["concept"] == concept]
    return [resolve_dataset_path(root, p) for p in rows[col]]


def sgr(base_value: float, immunized_value: float) -> float:
    if abs(base_value) < 1e-8:
        return float("nan")
    return (base_value - immunized_value) / base_value * 100.0


def score_arm(scorer: SGRScorer, csv_file: str, base_root: str, immunized_root: str, epochs):
    root = Path(csv_file).parent
    concepts = sorted(pd.read_csv(csv_file)["concept"].unique())

    results = {}
    for epoch in epochs:
        per_concept_sgr = []
        for concept in concepts:
            base_dir = Path(base_root) / str(epoch) / concept
            imm_dir = Path(immunized_root) / str(epoch) / concept
            if not base_dir.is_dir() or not imm_dir.is_dir():
                continue
            base_paths = sorted(base_dir.glob("*.png"))
            imm_paths = sorted(imm_dir.glob("*.png"))
            ref_paths = _reference_paths(csv_file, root, concept)

            n = min(len(ref_paths), len(base_paths), len(imm_paths))
            if n == 0:
                continue

            ref = torch.cat([_load_image_tensor(p) for p in ref_paths[:n]], dim=0)
            base = torch.cat([_load_image_tensor(p) for p in base_paths[:n]], dim=0)
            imm = torch.cat([_load_image_tensor(p) for p in imm_paths[:n]], dim=0)

            base_clip, base_dino, base_lpips = scorer.similarities(ref, base)
            imm_clip, imm_dino, imm_lpips = scorer.similarities(ref, imm)

            sgr_avg = np.mean([sgr(base_clip, imm_clip), sgr(base_dino, imm_dino), sgr(base_lpips, imm_lpips)])
            per_concept_sgr.append(sgr_avg)

        if per_concept_sgr:
            results[epoch] = float(np.mean(per_concept_sgr))

    return results
