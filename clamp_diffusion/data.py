import random
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


def _build_transform(resolution: int, center_crop: bool, random_flip: bool):
    return transforms.Compose([
        transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(resolution) if center_crop else transforms.RandomCrop(resolution),
        transforms.RandomHorizontalFlip() if random_flip else transforms.Lambda(lambda x: x),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])


def resolve_dataset_path(root: Path, path: str) -> Path:
    p = Path(path)
    if p.is_absolute():
        parts = p.parts
        if "dataset" in parts:
            idx = len(parts) - 1 - parts[::-1].index("dataset")
            p = Path(*parts[idx + 1:])
        else:
            p = Path(p.name)
    return root / p


def _tokenize(tokenizer, text: str):
    return tokenizer(
        text, max_length=tokenizer.model_max_length, padding="max_length",
        truncation=True, return_tensors="pt",
    ).input_ids[0]


class HarmGoodDataset(Dataset):
    def __init__(self, csv_file: str, tokenizer, config):
        self.df = pd.read_csv(csv_file)
        self.root = Path(csv_file).parent
        self.tokenizer = tokenizer
        self.transform = _build_transform(config.resolution, config.center_crop, config.random_flip)

    def __len__(self):
        return len(self.df)

    def _load(self, path):
        return Image.open(resolve_dataset_path(self.root, path)).convert("RGB")

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        harm_img = self.transform(self._load(row["image_path"] if "image_path" in row else row["image"]))
        harm_ids = _tokenize(self.tokenizer, row["prompt"] if "prompt" in row else row["text"])

        good_choice = "1" if random.random() < 0.5 else "2"
        good_img = self.transform(self._load(row[f"image{good_choice}"]))
        good_ids = _tokenize(self.tokenizer, row[f"text{good_choice}"])

        return {
            "harm_pixel_values": harm_img, "harm_input_ids": harm_ids,
            "good_pixel_values": good_img, "good_input_ids": good_ids,
        }


class ConceptImageDataset(Dataset):
    def __init__(self, csv_file: str, tokenizer, config):
        self.df = pd.read_csv(csv_file)
        self.root = Path(csv_file).parent
        self.tokenizer = tokenizer
        self.transform = _build_transform(config.resolution, config.center_crop, config.adaptation_random_flip)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = row["image_path"] if "image_path" in row else row["image"]
        image = self.transform(Image.open(resolve_dataset_path(self.root, path)).convert("RGB"))
        input_ids = _tokenize(self.tokenizer, row["prompt"] if "prompt" in row else row["text"])
        return {"pixel_values": image, "input_ids": input_ids}


def harm_good_dataloader(csv_file, tokenizer, config, shuffle=True):
    dataset = HarmGoodDataset(csv_file, tokenizer, config)
    return DataLoader(
        dataset, shuffle=shuffle, batch_size=config.train_batch_size,
        num_workers=config.dataloader_num_workers,
        collate_fn=lambda batch: {k: torch.stack([b[k] for b in batch]) for k in batch[0]},
    )


def concept_dataloader(csv_file, tokenizer, config, batch_size, shuffle=True):
    dataset = ConceptImageDataset(csv_file, tokenizer, config)
    return DataLoader(
        dataset, shuffle=shuffle, batch_size=batch_size,
        num_workers=config.dataloader_num_workers,
        collate_fn=lambda batch: {k: torch.stack([b[k] for b in batch]) for k in batch[0]},
    )
