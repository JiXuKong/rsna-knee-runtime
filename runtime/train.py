"""DINOv2 MIL training."""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
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
)
from runtime.data_prep import prepare_slot_maps
from runtime.dataset import KneeStudyDataset, collate_studies
from runtime.model.mil import build_model, macro_auc, predict, save_checkpoint
from runtime.submission import write_submission

LABEL_COLS = TARGETS + [t + "__conf" for t in TARGETS]
N_FOLDS = 5


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
            w[i] = 0.25 + 0.75 #* row[[t + "__conf" for t in TARGETS]].values
    return y, w, np.where(w.sum(1) > 0)[0]


def _study_ds(studies, data, decode_size, *, idx=None, y=None, w=None, train=False):
    ids = [studies[i] for i in idx] if idx is not None else studies
    kw = {}
    if y is not None:
        kw["y"] = torch.from_numpy(y[idx])
        kw["w"] = torch.from_numpy(w[idx])
    return KneeStudyDataset(ids, data["slots_tr"], data["lat_tr"], img_size=decode_size, train=train, **kw)


def train_fold(model, loader, val_ds, y_val, img_size, epochs, device, fold, writer, step):
    opt = torch.optim.AdamW(
        [
            {"params": [p for p in model.backbone.parameters() if p.requires_grad], "lr": LR_BACKBONE},
            {"params": model.head.parameters(), "lr": LR_HEAD},
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
                loss = (
                    F.binary_cross_entropy_with_logits(logits, yt, reduction="none") * wt
                ).mean()
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
            writer.add_scalar(f"{pfx}/batch/lr_backbone", opt.param_groups[0]["lr"], step)
            writer.add_scalar(f"{pfx}/batch/lr_head", opt.param_groups[1]["lr"], step)
            pbar.set_postfix(loss=f"{lv:.4f}")
            step += 1

        pred = predict(model, val_ds, device, img_size, desc=f"fold{fold} val ep{ep + 1}")
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
    print("[train] prepare slot maps...", flush=True)
    data = prepare_slot_maps(DATA_ROOT.resolve())
    print("[train] load labels...", flush=True)
    lab = load_labels(LABELS_PATH)
    y, w, keep = build_supervision(data["st_tr"], data["train_df"], lab)
    print(f"[train] supervision: {len(keep)} studies with labels", flush=True)
    decode_size = max(IMG_SIZE, LOAD_IMG)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_ds = KneeStudyDataset(data["st_te"], data["slots_te"], data["lat_te"], img_size=decode_size, train=False)

    groups = np.array([data["pid_tr"].get(data["st_tr"][i], data["st_tr"][i]) for i in keep])
    if len(np.unique(groups)) < N_FOLDS:
        groups = np.array([data["st_tr"][i] for i in keep])
    oof = np.zeros_like(y)
    fold_aucs, test_preds = [], []
    SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    TB_LOG_DIR.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(TB_LOG_DIR)
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
        model = build_model(MODEL_NAME, unfreeze_last=UNFREEZE_LAST, dinov2_path=DINOV2_PATH).to(device)
        auc, pred, step = train_fold(
            model, loader, val_ds, (y[va_idx] > 0.5).astype(int), IMG_SIZE, EPOCHS, device, fold, writer, step,
        )
        oof[va_idx] = pred
        fold_aucs.append(auc)
        writer.add_scalar("summary/fold_val_auc", auc, fold)
        fold_path = SAVE_PATH.with_name(f"{SAVE_PATH.stem}_fold{fold}{SAVE_PATH.suffix}")
        save_checkpoint(
            fold_path, model, model_name=MODEL_NAME, img_size=IMG_SIZE,
            unfreeze_last=UNFREEZE_LAST, dinov2_path=DINOV2_PATH, extra={"val_auc": auc, "fold": fold},
        )
        # test_preds.append(predict(model, test_ds, device, IMG_SIZE, desc=f"fold{fold} test"))
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
    print(f"tensorboard logdir={TB_LOG_DIR}", flush=True)

    # write_submission(np.mean(test_preds, 0), data["st_te"], data["test_df"], SAVE_PATH.with_suffix(".submission.csv"))
    print(f"done  mean_auc={mean_auc:.4f}  oof_auc={oof_auc:.4f}", flush=True)


def main() -> None:
    train()


if __name__ == "__main__":
    main()
