"""Shared torchvision augmentations for train and inference TTA."""
from __future__ import annotations

import torch
from torchvision.transforms import v2 as T


def train_transform() -> T.Compose:
    return T.Compose([
        T.RandomAffine(degrees=8, translate=(0.05, 0.05), scale=(1.0, 1.08), fill=0),
        T.RandomApply([T.ColorJitter(brightness=0.1, contrast=0.1)], p=0.8),
    ])


_TRANSFORM = train_transform()


def apply_augment(imgs: torch.Tensor, seed: int | None = None) -> torch.Tensor:
    """Apply train-style augment to a bag tensor [..., slice, H, W] (uint8)."""
    lead = imgs.shape[:-3]
    x = imgs.reshape(-1, *imgs.shape[-3:]).float().div_(255.0)
    if seed is not None:
        state = torch.get_rng_state()
        torch.manual_seed(seed)
        x = _TRANSFORM(x)
        torch.set_rng_state(state)
    else:
        x = _TRANSFORM(x)
    return (x * 255).round().clamp(0, 255).to(torch.uint8).reshape(*lead, *x.shape[-3:])
