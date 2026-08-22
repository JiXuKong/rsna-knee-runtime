"""DINOv2 MIL training with DepthCompress stem (GROUP slices → 3ch ImageNet).

Standalone of runtime/train.py. Each slot's G grayscale slices are compressed
to 3 channels (learnable stem), then one DINOv2 forward per slot.
"""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from runtime.config import (
    BATCH_STUDIES,
    DATA_ROOT,
    DINOV2_PATH,
    EPOCHS,
    GROUP,
    IMG_SIZE,
    LABELS_PATH,
    LOAD_IMG,
    LR_BACKBONE,
    LR_HEAD,
    MODEL_NAME,
    SAVE_PATH,
    SEED,
    TARGETS,
    TB_LOG_DIR,
    NUM_WORKERS,
    PREFETCH_FACTOR,
    UNFREEZE_LAST,
    PIN_MEMORY,
    PERSISTENT_WORKERS,
    WEIGHT_DECAY,
    POOL_PARTS,
)
from runtime.data_prep import prepare_slot_maps
from runtime.dataset import KneeStudyDataset, collate_studies
from runtime.transforms import gpu_slot_stack_transform
from runtime.model.mil import (
    MODELS,
    MeanHead,
    N_SLOT,
    build_model as _build_backbone_model,
    macro_auc,
    predict,
)

LABEL_COLS = TARGETS + [t + "__conf" for t in TARGETS]
N_FOLDS = 5

# 与默认 train 权重分开存
SAVE_PATH_C = SAVE_PATH.with_name(f"{SAVE_PATH.stem}_compress{SAVE_PATH.suffix}")
TB_LOG_DIR_C = TB_LOG_DIR.parent / f"{TB_LOG_DIR.name}_compress"

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _seed_worker(worker_id: int) -> None:
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    random.seed(seed)


def load_labels(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "StudyInstanceUID" not in df.columns:
        raise ValueError(f"{path}: missing StudyInstanceUID column")
    missing = [c for c in LABEL_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {missing[:3]}...")
    return df.set_index("StudyInstanceUID")


def build_supervision(st_tr: list[str], train_df: pd.DataFrame, lab: pd.DataFrame):
    gold = train_df.set_index("StudyInstanceUID")[TARGETS]
    gold = gold[gold.notna().all(axis=1)]
    y = np.zeros((len(st_tr), len(TARGETS)), np.float32)
    w = np.zeros_like(y)
    for i, st in enumerate(st_tr):
        if st in gold.index:
            y[i], w[i] = gold.loc[st].values, 3.0
        elif st in lab.index:
            row = lab.loc[st]
            y[i] = row[TARGETS].values
            w[i] = 0.25 + 0.75
    return y, w, np.where(w.sum(1) > 0)[0]


def _study_ds(studies, data, decode_size, *, idx=None, y=None, w=None, train=False):
    ids = [studies[i] for i in idx] if idx is not None else studies
    kw = {}
    if y is not None:
        kw["y"] = torch.from_numpy(y[idx])
        kw["w"] = torch.from_numpy(w[idx])
    return KneeStudyDataset(ids, data["slots_tr"], data["lat_tr"], img_size=decode_size, train=train, **kw)


DIV_ALPHA_WEIGHT = 0.05  # 三路注意力分布两两余弦相似度惩罚


class DynamicSliceCompress(nn.Module):
    """方案 A：切片嵌入 + 每 slot 三个 query → softmax 加权合成伪 RGB。

    RGB_c = Σ_g α_{c,g}(x) · Slice_g，α 随输入变化；prior 保证开局接近均匀三层选片。
    """

    def __init__(
        self,
        n_slice: int = GROUP,       # 每个 slot 的切片数 G（与 config.GROUP 一致）
        n_slot: int = N_SLOT,       # MRI 槽位数（当前 6）
        out_ch: int = 3,            # 输出通道数，固定为 3 以对接 DINOv2 的 RGB 输入
        emb_dim: int = 32,          # 每张切片编码后的向量维度 d
        temperature: float = 1.0,   # softmax 温度：越小越接近「只选一层」
    ):
        super().__init__()
        self.n_slice = n_slice
        self.n_slot = n_slot
        self.out_ch = out_ch
        self.emb_dim = emb_dim
        self.temperature = temperature
        # 切片编码器：单张灰度图 → d 维全局描述（先局部卷积再 GAP）
        self.slice_enc = nn.Sequential(
            nn.Conv2d(1, emb_dim, 3, padding=1, bias=False),  # 提局部纹理
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),  # 空间池化成 [N*G, d, 1, 1]
        )
        # 每个 slot 有 out_ch 个可学习 query，用于动态打分「更看哪一层」
        self.query = nn.Parameter(torch.randn(n_slot, out_ch, emb_dim) * 0.02)
        # 开局偏向前/中/后切片，避免三路注意力塌成同一层
        prior = torch.zeros(n_slot, out_ch, n_slice)  # [slot, 通道c, 切片g]
        idx = torch.linspace(0, n_slice - 1, out_ch).round().long()  # 如 G=10 → [0,4,9]
        for c, i in enumerate(idx.tolist()):
            prior[:, c, i] = 2.0  # 给对应 (c, i) 一个大正的 score bias
        self.register_buffer("prior", prior)  # 不参与梯度，但随模型一起存盘
        self.prior_scale = nn.Parameter(torch.tensor(1.0))  # 可学习：prior 强度可被训练调弱/调强
        # ImageNet 归一化，使伪 RGB 统计接近 DINOv2 预训练分布
        self.register_buffer("mu", torch.tensor(IMAGENET_MEAN).view(1, -1, 1, 1))
        self.register_buffer("sd", torch.tensor(IMAGENET_STD).view(1, -1, 1, 1))
        self._last_alpha: torch.Tensor | None = None  # 缓存本次 forward 的 α，供 diversity_loss 使用

    def forward(self, x: torch.Tensor, slot_ids: torch.Tensor) -> torch.Tensor:
        """x: [N,G,H,W]∈[0,1]；slot_ids: [N]∈[0,n_slot)。返回 [N,3,H,W]（已 ImageNet 归一化）。"""
        n, g, h, w = x.shape
        if g != self.n_slice:
            raise ValueError(f"expected n_slice={self.n_slice}, got {g}")
        slice_keep = x.amax(dim=(2, 3)) > 0  # [N,G]：该切片是否非空（全 0 视为缺失）

        # 每张切片独立编码 → e: [N, G, d]
        e = self.slice_enc(x.reshape(n * g, 1, h, w)).flatten(1).reshape(n, g, self.emb_dim)
        q = self.query[slot_ids]  # 按 slot 取 query → [N, 3, d]
        # 缩放点积注意力分数 s_{c,g} = <e_g, q_c> / sqrt(d)
        scores = torch.einsum("ngd,ncd->ncg", e, q) * (self.emb_dim ** -0.5)
        scores = scores / max(self.temperature, 1e-6)  # 温度缩放
        scores = scores + self.prior_scale * self.prior[slot_ids]  # 加上深度先验 bias
        scores = scores.masked_fill(~slice_keep.unsqueeze(1), -1e4)  # 空切片禁止被选中
        alpha = scores.softmax(dim=-1)  # [N,3,G]，每行对 G 归一化且非负
        self._last_alpha = alpha  # 留给 diversity_loss

        # 伪 RGB：每个输出通道是全部切片的加权和（仍是「真实图像」的凸组合）
        rgb = torch.einsum("ncg,nghw->nchw", alpha, x)
        keep = slice_keep.any(dim=1).to(dtype=x.dtype).view(n, 1, 1, 1)  # slot 是否至少有一张有效切片
        rgb = (rgb - self.mu.to(dtype=rgb.dtype)) / self.sd.to(dtype=rgb.dtype)  # ImageNet 标准化
        return rgb * keep  # 全空 slot 输出清零，避免假信号进 backbone

    def diversity_loss(self) -> torch.Tensor:
        """惩罚三路 α 过于相似（希望三条通道关注不同深度）。"""
        alpha = self._last_alpha
        if alpha is None:
            return torch.zeros((), device=self.query.device)
        loss = alpha.new_zeros(())
        n_pair = 0
        for i in range(self.out_ch):
            for j in range(i + 1, self.out_ch):
                # cos↑ 表示两路选层分布越像 → 加入 loss 拉低相似度
                loss = loss + F.cosine_similarity(alpha[:, i], alpha[:, j], dim=-1).mean()
                n_pair += 1
        return loss / max(n_pair, 1)


class CompressModel(nn.Module):
    """DINOv2 + 动态切片加权 stem：每 slot 一次 backbone。"""

    def __init__(self, backbone, dim, pool="cls_mean", prior=False, n_slice: int = GROUP):
        super().__init__()
        self.backbone = backbone
        self.pool = pool
        self.n_slice = n_slice
        self.compress = DynamicSliceCompress(n_slice=n_slice, n_slot=N_SLOT, out_ch=3)
        self.aug = gpu_slot_stack_transform()
        self.head = MeanHead(dim * POOL_PARTS[pool], N_SLOT, len(TARGETS), prior=prior)

    def forward(self, imgs, mask, img_size=None):
        b, s, g, h, w = imgs.shape
        if g != self.n_slice:
            raise ValueError(f"expected GROUP/n_slice={self.n_slice}, got {g}")
        x = imgs.float().div(255.0)
        if img_size is not None and img_size != h:
            x = F.interpolate(
                x.reshape(b * s * g, 1, h, w),
                size=(img_size, img_size),
                mode="bilinear",
                align_corners=False,
            )
            h = w = img_size
            x = x.reshape(b * s, g, h, w)
        else:
            x = x.reshape(b * s, g, h, w)

        if self.training:
            with torch.autocast(device_type="cuda", enabled=False):
                x = self.aug(x.float().clamp(0.0, 1.0))

        slot_ids = torch.arange(s, device=x.device).repeat(b)
        x = self.compress(x, slot_ids)  # [B*S, 3, H, W]
        out = self.backbone(pixel_values=x).last_hidden_state
        patch = out[:, 1:]
        parts = [out[:, 0], patch.mean(1)]
        if self.pool == "cls_mean_focal":
            k = max(1, patch.shape[1] // 8)
            parts.append(patch.topk(k, dim=1).values.mean(1))
        feat = torch.cat(parts, dim=1).reshape(b, s, -1)
        return self.head(feat, mask)


def build_compress_model(
    name: str = MODEL_NAME,
    *,
    unfreeze_last: int = UNFREEZE_LAST,
    dinov2_path: str | Path = DINOV2_PATH,
    n_slice: int = GROUP,
) -> CompressModel:
    base = _build_backbone_model(name, unfreeze_last=unfreeze_last, dinov2_path=dinov2_path)
    _, pool, prior = MODELS[name]
    dim = base.backbone.config.hidden_size
    model = CompressModel(base.backbone, dim, pool=pool, prior=prior, n_slice=n_slice)
    n_stem = sum(p.numel() for p in model.compress.parameters())
    idx = torch.linspace(0, n_slice - 1, 3).round().long().tolist()
    print(
        f"stem DynamicSlice: n_slice={n_slice} → 3ch  params={n_stem}  "
        f"prior_peaks={idx}  div_w={DIV_ALPHA_WEIGHT}",
        flush=True,
    )
    return model


def save_compress_checkpoint(path, model, *, model_name, img_size, unfreeze_last, dinov2_path, extra=None):
    variant, pool, prior = MODELS[model_name]
    payload = {
        "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
        "model_name": model_name,
        "img_size": img_size,
        "unfreeze_last": unfreeze_last,
        "variant": variant,
        "pool": pool,
        "prior": prior,
        "stem": "dynamic_slice",
        "n_slice": model.n_slice,
        "dinov2_path": str(dinov2_path) if dinov2_path else None,
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)
    print(f"saved {path}", flush=True)


def train_fold(model, loader, val_ds, y_val, img_size, epochs, device, fold, writer, step):
    opt = torch.optim.AdamW(
        [
            {"params": [p for p in model.backbone.parameters() if p.requires_grad], "lr": LR_BACKBONE},
            {"params": model.head.parameters(), "lr": LR_HEAD},
            {"params": model.compress.parameters(), "lr": LR_HEAD},
        ],
        weight_decay=WEIGHT_DECAY,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best_auc, best_state = -1.0, None
    n_batch = max(len(loader), 1)
    warmup_steps = max(2 * n_batch, 1)
    sched = torch.optim.lr_scheduler.LinearLR(
        opt, start_factor=1e-3, end_factor=1.0, total_iters=warmup_steps,
    )
    for ep in tqdm(range(epochs), desc=f"fold{fold}", unit="ep"):
        model.train()
        ep_loss = 0.0
        ep_pred, ep_y = [], []
        pbar = tqdm(loader, desc=f"fold{fold} ep{ep + 1}", leave=False, unit="batch", mininterval=1.0)
        for bi, batch in enumerate(pbar):
            imgs, mask, yt, wt = (
                batch["imgs"].to(device), batch["mask"].to(device),
                batch["y"].to(device), batch["w"].to(device),
            )
            with torch.autocast("cuda", enabled=device.type == "cuda"):
                logits = model(imgs, mask, img_size)
                bce = (
                    F.binary_cross_entropy_with_logits(logits, yt, reduction="none") * wt
                ).mean()
                div = model.compress.diversity_loss()
                loss = bce + DIV_ALPHA_WEIGHT * div
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()

            with torch.no_grad():
                ep_pred.append(torch.sigmoid(logits).float().cpu())
                ep_y.append(yt.detach().cpu())
            lv = loss.item()
            ep_loss += lv

            pfx = f"fold{fold}"
            writer.add_scalar(f"{pfx}/batch/loss", lv, step)
            writer.add_scalar(f"{pfx}/batch/bce", bce.item(), step)
            writer.add_scalar(f"{pfx}/batch/div", float(div.detach()), step)
            writer.add_scalar(f"{pfx}/batch/lr_backbone", opt.param_groups[0]["lr"], step)
            writer.add_scalar(f"{pfx}/batch/lr_head", opt.param_groups[1]["lr"], step)
            writer.add_scalar(f"{pfx}/batch/lr_stem", opt.param_groups[2]["lr"], step)
            pbar.set_postfix(loss=f"{lv:.4f}", div=f"{float(div.detach()):.3f}")
            step += 1

        pred = predict(model, val_ds, device, img_size, desc=f"fold{fold} val ep{ep + 1}")
        pred = np.nan_to_num(pred, nan=0.5)
        val_auc = macro_auc(y_val, pred)
        y_tr = (torch.cat(ep_y).numpy() > 0.5).astype(int)
        ep_train_auc = macro_auc(y_tr, torch.cat(ep_pred).numpy())
        ep_train_loss = ep_loss / n_batch
        pfx = f"fold{fold}"
        writer.add_scalar(f"{pfx}/epoch/train_loss", ep_train_loss, ep)
        writer.add_scalar(f"{pfx}/epoch/train_auc", ep_train_auc, ep)
        writer.add_scalar(f"{pfx}/epoch/val_auc", val_auc, ep)
        print(
            f"epoch {ep + 1}/{epochs}  loss {ep_train_loss:.4f}  "
            f"train_auc {ep_train_auc:.4f}  val_auc {val_auc:.4f}",
            flush=True,
        )
        if val_auc > best_auc:
            best_auc = val_auc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            writer.add_scalar(f"{pfx}/epoch/best_val_auc", best_auc, ep)
    if best_state is not None:
        model.load_state_dict(best_state)
    return best_auc, predict(model, val_ds, device, img_size), step


def train() -> None:
    print(f"[train_compress] GROUP={GROUP}  stem=DynamicSlice(query-attn) → 3ch", flush=True)
    print("[train_compress] prepare slot maps...", flush=True)
    data = prepare_slot_maps(DATA_ROOT.resolve())
    print("[train_compress] load labels...", flush=True)
    lab = load_labels(LABELS_PATH)
    y, w, keep = build_supervision(data["st_tr"], data["train_df"], lab)
    print(f"[train_compress] supervision: {len(keep)} studies with labels", flush=True)
    decode_size = max(IMG_SIZE, LOAD_IMG)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    groups = np.array([data["pid_tr"].get(data["st_tr"][i], data["st_tr"][i]) for i in keep])
    if len(np.unique(groups)) < N_FOLDS:
        groups = np.array([data["st_tr"][i] for i in keep])
    oof = np.zeros_like(y)
    fold_aucs = []
    SAVE_PATH_C.parent.mkdir(parents=True, exist_ok=True)
    TB_LOG_DIR_C.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(TB_LOG_DIR_C)
    step = 0

    for fold, (tr, va) in enumerate(GroupKFold(N_FOLDS).split(keep, groups=groups)):
        tr_idx, va_idx = keep[tr], keep[va]
        print(f"\nfold {fold + 1}/{N_FOLDS}  train {len(tr_idx)}  val {len(va_idx)}", flush=True)
        torch.manual_seed(SEED + fold)
        train_ds = _study_ds(data["st_tr"], data, decode_size, idx=tr_idx, y=y, w=w, train=True)
        val_ds = _study_ds(data["st_tr"], data, decode_size, idx=va_idx)
        g = torch.Generator()
        g.manual_seed(SEED + fold)
        loader = DataLoader(
            train_ds, batch_size=BATCH_STUDIES, shuffle=True,
            collate_fn=collate_studies, drop_last=len(train_ds) >= BATCH_STUDIES,
            num_workers=NUM_WORKERS,
            pin_memory=PIN_MEMORY and device.type == "cuda",
            persistent_workers=PERSISTENT_WORKERS and NUM_WORKERS > 0,
            prefetch_factor=PREFETCH_FACTOR if NUM_WORKERS > 0 else None,
            worker_init_fn=_seed_worker if NUM_WORKERS > 0 else None,
            generator=g,
        )
        model = build_compress_model(
            MODEL_NAME, unfreeze_last=UNFREEZE_LAST, dinov2_path=DINOV2_PATH, n_slice=GROUP,
        ).to(device)
        auc, pred, step = train_fold(
            model, loader, val_ds, (y[va_idx] > 0.5).astype(int), IMG_SIZE, EPOCHS, device, fold, writer, step,
        )
        oof[va_idx] = pred
        fold_aucs.append(auc)
        writer.add_scalar("summary/fold_val_auc", auc, fold)
        fold_path = SAVE_PATH_C.with_name(f"{SAVE_PATH_C.stem}_fold{fold}{SAVE_PATH_C.suffix}")
        save_compress_checkpoint(
            fold_path, model, model_name=MODEL_NAME, img_size=IMG_SIZE,
            unfreeze_last=UNFREEZE_LAST, dinov2_path=DINOV2_PATH,
            extra={"val_auc": auc, "fold": fold},
        )
        print(f"fold {fold + 1} auc={auc:.4f}  weights={fold_path}", flush=True)

    y_bin = (y[keep] > 0.5).astype(int)
    oof_auc = macro_auc(y_bin, oof[keep])
    mean_auc = float(np.mean(fold_aucs))
    for i, a in enumerate(fold_aucs):
        print(f"fold {i + 1} auc={a:.4f}")
        writer.add_scalar("summary/fold_val_auc_final", a, i)
    writer.add_scalar("summary/mean_val_auc", mean_auc, 0)
    writer.add_scalar("summary/oof_auc", oof_auc, 0)
    writer.close()
    print(f"mean auc={mean_auc:.4f}  oof auc={oof_auc:.4f}", flush=True)
    print(f"tensorboard logdir={TB_LOG_DIR_C}", flush=True)
    print(f"done  mean_auc={mean_auc:.4f}  oof_auc={oof_auc:.4f}", flush=True)


def main() -> None:
    train()


if __name__ == "__main__":
    main()
