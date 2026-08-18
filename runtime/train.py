"""DINOv2 MIL training."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from runtime.config import (
    BATCH_STUDIES,
    DATA_ROOT,
    EPOCHS,
    GROUP,
    LOAD_IMG,
    LR_BACKBONE,
    LR_HEAD,
    SEED,
    TARGETS,
    UNFREEZE_LAST,
    WEIGHT_DECAY,
)
from runtime.data_prep import prepare_slot_maps
from runtime.dataset import KneeStudyDataset, collate_studies
from runtime.model.mil import MODELS, build_model, macro_auc, predict, save_checkpoint
from runtime.submission import write_submission

LABEL_COLS = TARGETS + [t + "__conf" for t in TARGETS]


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
            w[i] = 0.25 + 0.75 * row[[t + "__conf" for t in TARGETS]].values
    keep = np.where(w.sum(1) > 0)[0]
    n_val = max(1, len(keep) // 5)
    tr, va = keep[n_val:], keep[:n_val]
    return y, w, tr, va


def train(
    *,
    data_root: Path,
    labels_path: Path,
    model_name: str,
    img_size: int,
    save_path: Path,
    epochs: int,
    dinov2_path: Path,
    unfreeze_last: int,
) -> None:
    root = data_root.resolve()
    data = prepare_slot_maps(root)
    lab = load_labels(labels_path)

    y, w, tr_idx, va_idx = build_supervision(data["st_tr"], data["train_df"], lab)
    train_studies = [data["st_tr"][i] for i in tr_idx]
    val_studies = [data["st_tr"][i] for i in va_idx]
    decode_size = max(img_size, LOAD_IMG)

    train_ds = KneeStudyDataset(
        train_studies, data["slots_tr"], data["lat_tr"],
        y=torch.from_numpy(y[tr_idx]), w=torch.from_numpy(w[tr_idx]),
        img_size=decode_size, train=True,
    )
    val_ds = KneeStudyDataset(
        val_studies, data["slots_tr"], data["lat_tr"],
        img_size=decode_size, train=False,
    )
    test_ds = KneeStudyDataset(
        data["st_te"], data["slots_te"], data["lat_te"],
        img_size=decode_size, train=False,
    )
    loader = DataLoader(
        train_ds, batch_size=BATCH_STUDIES, shuffle=True,
        collate_fn=collate_studies, drop_last=len(train_ds) >= BATCH_STUDIES,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(SEED)
    model = build_model(model_name, unfreeze_last=unfreeze_last, dinov2_path=dinov2_path).to(device)
    opt = torch.optim.AdamW(
        [
            {"params": [p for p in model.backbone.parameters() if p.requires_grad], "lr": LR_BACKBONE},
            {"params": model.head.parameters(), "lr": LR_HEAD},
        ],
        weight_decay=WEIGHT_DECAY,
    )

    best_auc, best_state = -1.0, None
    y_val = (y[va_idx] > 0.5).astype(int)

    for ep in range(epochs):
        model.train()
        total = 0.0
        for batch in loader:
            imgs, mask, yt, wt = (
                batch["imgs"].to(device), batch["mask"].to(device),
                batch["y"].to(device), batch["w"].to(device),
            )
            with torch.autocast("cuda", enabled=device.type == "cuda"):
                loss = (
                    F.binary_cross_entropy_with_logits(model(imgs, mask, img_size), yt, reduction="none") * wt
                ).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += loss.item()

        val_pred = predict(model, val_ds, device, img_size)
        auc = macro_auc(y_val, val_pred)
        print(f"epoch {ep + 1}/{epochs}  loss {total / max(len(loader), 1):.4f}  val_auc {auc:.4f}", flush=True)
        if auc > best_auc:
            best_auc = auc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint(
        save_path, model, model_name=model_name, img_size=img_size,
        unfreeze_last=unfreeze_last, dinov2_path=dinov2_path, extra={"val_auc": best_auc},
    )

    test_pred = predict(model, test_ds, device, img_size)
    write_submission(test_pred, data["st_te"], data["test_df"], save_path.with_suffix(".submission.csv"))
    print(f"done  val_auc={best_auc:.4f}  weights={save_path}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description="Train DINOv2 MIL")
    p.add_argument("--data-root", type=Path, default=DATA_ROOT, help="竞赛数据目录")
    p.add_argument("--labels", type=Path, required=True, help="derived_labels.csv")
    p.add_argument("--model", default="dinov2-small", choices=list(MODELS))
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--save", type=Path, default=Path("outputs/model.pt"))
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--dinov2", type=Path, required=True, help="DINOv2 权重目录")
    p.add_argument("--unfreeze-last", type=int, default=UNFREEZE_LAST)
    args = p.parse_args()
    train(
        data_root=args.data_root,
        labels_path=args.labels,
        model_name=args.model,
        img_size=args.img_size,
        save_path=args.save,
        epochs=args.epochs,
        dinov2_path=args.dinov2,
        unfreeze_last=args.unfreeze_last,
    )


if __name__ == "__main__":
    main()
