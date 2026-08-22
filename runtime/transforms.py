"""GPU-side augmentations using torchvision.transforms.v2.

torchvision v2 transforms 支持直接作用于 GPU float tensor，
因此把 augment 从 Dataset worker 移到训练 loop（GPU 上）后，
CPU worker 只需做 DICOM 解码，GPU 做增强，互不阻塞。
"""
from __future__ import annotations

import torch
from torchvision.transforms import v2 as T
from torchvision.transforms.v2 import functional as F


class RandomGamma:
    """Per-image gamma in [0, 1]. gamma<1 brightens, gamma>1 darkens."""

    def __init__(self, gamma: tuple[float, float] = (0.8, 1.25), p: float = 0.5):
        self.lo, self.hi = gamma
        self.p = p

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        if img.ndim == 3:
            if torch.rand((), device=img.device) >= self.p:
                return img
            g = torch.empty((), device=img.device).uniform_(self.lo, self.hi).item()
            return F.adjust_gamma(img, g)
        n = img.shape[0]
        apply = torch.rand(n, device=img.device) < self.p
        if not bool(apply.any()):
            return img
        gammas = torch.empty(n, device=img.device).uniform_(self.lo, self.hi)
        powered = img.clamp(0, 1).pow(gammas.view(-1, 1, 1, 1))
        return torch.where(apply.view(-1, 1, 1, 1), powered, img)


def gpu_train_transform() -> T.Compose:
    """返回可作用于 GPU float32 tensor [N,1,H,W]∈[0,1] 的增强流水线。"""
    return T.Compose([
        T.RandomAffine(degrees=8, translate=(0.05, 0.05), scale=(1.0, 1.08), fill=0),
        T.RandomApply([T.ColorJitter(brightness=0.1, contrast=0.1)], p=0.8),
        RandomGamma(gamma=(0.8, 1.25), p=0.5),
        T.RandomApply([T.GaussianNoise(mean=0.0, sigma=0.03, clip=True)], p=0.5),
        T.RandomErasing(p=0.5, scale=(0.02, 0.12), ratio=(0.3, 3.3), value=0.0),
    ])


# 保留旧名，dataset.py 里 train=False 时不再使用
def train_transform() -> T.Compose:
    return gpu_train_transform()
