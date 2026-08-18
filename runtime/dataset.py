"""PyTorch Dataset — on-demand DICOM loading with torchvision augmentations."""
from __future__ import annotations

import torch
from torch.utils.data import Dataset

from runtime.config import GROUP, LOAD_IMG, SLOTS
from runtime.dicom.io import load_study
from runtime.transforms import train_transform

N_SLOT = len(SLOTS)


class KneeStudyDataset(Dataset):
    """One item = one study bag [n_slot, n_slice, H, W]."""

    def __init__(
        self,
        studies: list[str],
        slot_map: dict,
        lat_map: dict,
        *,
        y: torch.Tensor | None = None,
        w: torch.Tensor | None = None,
        img_size: int = LOAD_IMG,
        n_slices: int = GROUP,
        train: bool = False,
    ) -> None:
        self.studies = studies
        self.slot_map = slot_map
        self.lat_map = lat_map
        self.y = y
        self.w = w
        self.img_size = img_size
        self.n_slices = n_slices
        self.transform = train_transform() if train else None

    def __len__(self) -> int:
        return len(self.studies)

    def __getitem__(self, idx: int) -> dict:
        study = self.studies[idx]
        imgs, mask = load_study(
            study,
            self.slot_map,
            self.lat_map,
            n_slice=self.n_slices,
            out_size=self.img_size,
        )
        if self.transform is not None:
            s, g, h, w = imgs.shape
            x = imgs.reshape(s * g, h, w).float().div_(255.0)
            x = self.transform(x)
            imgs = (x * 255).round().clamp(0, 255).to(torch.uint8).reshape(s, g, h, w)

        out = {"imgs": imgs, "mask": mask, "study": study}
        if self.y is not None:
            out["y"] = self.y[idx]
        if self.w is not None:
            out["w"] = self.w[idx]
        return out


def collate_studies(batch: list[dict]) -> dict:
    out: dict = {
        "imgs": torch.stack([b["imgs"] for b in batch]),
        "mask": torch.stack([b["mask"] for b in batch]),
        "study": [b["study"] for b in batch],
    }
    if "y" in batch[0]:
        out["y"] = torch.stack([b["y"] for b in batch])
    if "w" in batch[0]:
        out["w"] = torch.stack([b["w"] for b in batch])
    return out
