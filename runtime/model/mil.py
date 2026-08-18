"""DINOv2 MIL model — build, load, predict."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from runtime.config import (
    EVAL_BATCH,
    POOL_PARTS,
    SLOTS,
    SLOT_PRIOR_STRENGTH,
    SLOT_PRIOR_TABLE,
    TARGETS,
    UNFREEZE_LAST,
)
from runtime.dataset import collate_studies

N_SLOT = len(SLOTS)

# model name -> (transformers variant, pool, prior)
MODELS = {
    "dinov2-small": ("small", "cls_mean", False),
    "dinov2-base": ("base", "cls_mean", False),
    "dinov2-small-focal": ("small", "cls_mean_focal", True),
}


class SlotHead(nn.Module):
    def __init__(self, dim, n_slot, n_out, hidden=256, p=0.2, prior=False):
        super().__init__()
        self.proj = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, hidden), nn.GELU())
        self.slot_emb = nn.Parameter(torch.randn(n_slot, hidden) * 0.02)
        self.query = nn.Parameter(torch.randn(n_out, hidden) * 0.02)
        self.drop = nn.Dropout(p)
        self.out = nn.Linear(hidden, n_out)
        self.hidden = hidden
        p_ = torch.zeros(n_out, n_slot)
        if prior and n_slot == len(SLOTS) and n_out == len(TARGETS):
            for t, slots in SLOT_PRIOR_TABLE.items():
                if t in TARGETS:
                    p_[TARGETS.index(t), list(slots)] = SLOT_PRIOR_STRENGTH
        self.prior = prior
        if prior:
            self.register_buffer("slot_prior", p_)

    def forward(self, x, mask):
        h = self.proj(x) + self.slot_emb
        att = torch.einsum("bsh,oh->bos", h, self.query) / self.hidden ** 0.5
        if self.prior:
            att = att + self.slot_prior.unsqueeze(0)
        att = att.masked_fill(mask.unsqueeze(1) < 0.5, -1e4).softmax(-1)
        ctx = self.drop(torch.einsum("bos,bsh->boh", att, h))
        return (ctx * self.out.weight.unsqueeze(0)).sum(-1) + self.out.bias


class Model(nn.Module):
    def __init__(self, backbone, dim, pool="cls_mean", prior=False):
        super().__init__()
        self.backbone = backbone
        self.pool = pool
        self.head = SlotHead(dim * POOL_PARTS[pool], N_SLOT, len(TARGETS), prior=prior)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, imgs, mask, img_size=None):
        b, s = imgs.shape[:2]
        x = imgs.reshape(b * s, *imgs.shape[2:]).float().div_(255.0)
        if img_size is not None and img_size != x.shape[-1]:
            x = F.interpolate(x, size=(img_size, img_size), mode="bilinear", align_corners=False)
        x = (x - self.mean) / self.std
        out = self.backbone(pixel_values=x).last_hidden_state
        patch = out[:, 1:]
        parts = [out[:, 0], patch.mean(1)]
        if self.pool == "cls_mean_focal":
            k = max(1, patch.shape[1] // 8)
            parts.append(patch.topk(k, dim=1).values.mean(1))
        feat = torch.cat(parts, dim=1).reshape(b, s, -1)
        return self.head(feat, mask)


def build_model(
    name: str = "dinov2-small",
    *,
    unfreeze_last: int = UNFREEZE_LAST,
    dinov2_path: str | Path,
) -> Model:
    if name not in MODELS:
        raise ValueError(f"unknown model {name!r}; choose from {list(MODELS)}")
    _, pool, prior = MODELS[name]
    from transformers import AutoModel

    path = Path(dinov2_path)
    if not path.is_dir():
        raise FileNotFoundError(f"DINOv2 目录不存在: {path}")
    bb = AutoModel.from_pretrained(str(path))
    n_layer = len(bb.encoder.layer)
    for p in bb.parameters():
        p.requires_grad = False
    for blk in bb.encoder.layer[max(0, n_layer - unfreeze_last):]:
        for p in blk.parameters():
            p.requires_grad = True
    for p in bb.layernorm.parameters():
        p.requires_grad = True
    dim = bb.config.hidden_size
    print(f"model {name}: {n_layer} blocks, unfreeze last {unfreeze_last}, dim {dim}", flush=True)
    return Model(bb, dim, pool=pool, prior=prior)


def save_checkpoint(
    path: str | Path,
    model: Model,
    *,
    model_name: str,
    img_size: int,
    unfreeze_last: int = UNFREEZE_LAST,
    dinov2_path: str | Path | None = None,
    extra: dict | None = None,
) -> None:
    variant, pool, prior = MODELS[model_name]
    payload = {
        "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
        "model_name": model_name,
        "img_size": img_size,
        "unfreeze_last": unfreeze_last,
        "variant": variant,
        "pool": pool,
        "prior": prior,
        "dinov2_path": str(dinov2_path) if dinov2_path else None,
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)
    print(f"saved {path}", flush=True)


def load_model(
    path: str | Path,
    device: torch.device | None = None,
    *,
    dinov2_path: str | Path | None = None,
) -> tuple[Model, int]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model_name = ckpt.get("model_name", "dinov2-small")
    img_size = int(ckpt.get("img_size", ckpt.get("img", 224)))
    backbone = dinov2_path or ckpt.get("dinov2_path")
    if not backbone:
        raise FileNotFoundError("checkpoint 未保存 dinov2_path，请传 --dinov2")
    model = build_model(
        model_name,
        unfreeze_last=int(ckpt.get("unfreeze_last", UNFREEZE_LAST)),
        dinov2_path=backbone,
    )
    model.load_state_dict(ckpt["state_dict"])
    if device is not None:
        model.to(device)
    model.eval()
    print(f"loaded {path}  model={model_name}  img_size={img_size}", flush=True)
    return model, img_size


@torch.no_grad()
def predict(model: Model, dataset, device: torch.device, img_size: int | None = None) -> np.ndarray:
    model.eval()
    loader = DataLoader(dataset, batch_size=EVAL_BATCH, shuffle=False, collate_fn=collate_studies)
    out = []
    for batch in loader:
        imgs = batch["imgs"].to(device)
        mask = batch["mask"].to(device)
        with torch.autocast("cuda", enabled=device.type == "cuda"):
            z = model(imgs, mask, img_size).float()
        out.append(torch.sigmoid(z).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, len(TARGETS)), np.float32)


def macro_auc(y, p) -> float:
    from sklearn.metrics import roc_auc_score

    return float(
        np.nanmean(
            [
                roc_auc_score(y[:, j], p[:, j]) if len(set(y[:, j])) > 1 else np.nan
                for j in range(y.shape[1])
            ]
        )
    )
