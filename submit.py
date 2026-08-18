#!/usr/bin/env python3
"""
RSNA Knee — Kaggle 单文件提交脚本
复制本文件到 Notebook 单元格运行，或:  python submit.py

运行前修改下方「路径常量」。
"""
from __future__ import annotations

import os
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# =============================================================================
# 路径常量 —— 按你的 Kaggle Dataset 修改
# =============================================================================
DATA_ROOT = Path("/kaggle/input/competitions/rsna-knee-abnormality-detection")
WEIGHTS_PATH = Path("/kaggle/input/your-weights/model.pt")
DINOv2_PATH = Path("/kaggle/input/dinov2-small")
OUTPUT_PATH = Path("/kaggle/working/submission.csv")

# =============================================================================
# 模型 / 数据常量
# =============================================================================
TARGETS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
    "Medial OA", "Lateral OA", "PF OA", "Effusion", "Synovitis",
    "Baker's", "Contusion", "Fracture",
]

CROP_MM = 130.0
LOAD_IMG = 336
GROUP = 3
EVAL_BATCH = 8
HDR_THREADS = 16
UNFREEZE_LAST = 6
LAT_MIN_OFFSET_MM = 20.0
SLICE_BAND = (0.20, 0.80)
RULES = {"order": "normal", "lat": "centre", "slot_fallback": False, "decode_fill": "nearest"}

SLOTS = [
    ("SAG_FLUID_FS", "Sagittal", True, True),
    ("COR_FLUID_FS", "Coronal", True, True),
    ("AX_FLUID_FS", "Axial", True, True),
    ("SAG_FLUID_NOFS", "Sagittal", True, False),
    ("COR_T1", "Coronal", False, False),
    ("SAG_T1", "Sagittal", False, False),
]
N_SLOT = len(SLOTS)
POOL_PARTS = {"cls_mean": 2, "cls_mean_focal": 3}
SLOT_PRIOR_TABLE = {
    "ACL": (0, 3, 5), "MCL": (1, 4),
    "Medial Meniscus": (0, 1, 3, 4), "Lateral Meniscus": (0, 1, 3, 4),
    "Medial OA": (1, 4, 5), "Lateral OA": (1, 4, 5), "PF OA": (0, 2, 5),
    "Effusion": (0, 2), "Synovitis": (0, 2), "Baker's": (0,),
    "Contusion": (0, 1, 2), "Fracture": (0, 1, 2, 4, 5),
}
SLOT_PRIOR_STRENGTH = 0.55
MODELS = {
    "dinov2-small": ("small", "cls_mean", False),
    "dinov2-base": ("base", "cls_mean", False),
    "dinov2-small-focal": ("small", "cls_mean_focal", True),
}

FATSAT_OPTS = {"FS", "FATSAT", "FAT_SAT", "FSAT"}
_SEP = re.compile(r"[_\-.]")
_FATSAT_RX = re.compile(
    r"\bfs\b|fatsat|fat sat|\bstir\b|\bspair\b|\bspir\b|\bwe\b|"
    r"water excit|\btirm\b|\bsting\b|\bfatsup\b"
)
_T1_RX = re.compile(r"\bt1\b|\bt1w\b")
_T2_RX = re.compile(r"\bt2\b|\bt2w\b")
_PD_RX = re.compile(r"\bpd\b|\bpdw\b|proton|\bdp\b|dens")
ORDER_TAGS = [(0x0020, 0x0032), (0x0020, 0x0037), (0x0020, 0x0013)]
HDR_TAGS = [
    "SeriesDescription", "SequenceName", "ScanOptions", "ScanningSequence",
    "RepetitionTime", "EchoTime", "Laterality", "PixelSpacing",
    "Rows", "Columns", "RescaleSlope", "RescaleIntercept",
    "ImagePositionPatient", "ImageOrientationPatient",
]


# =============================================================================
# DICOM 头解析
# =============================================================================
def _hdr_vec(s, n):
    if not isinstance(s, str):
        return None
    try:
        v = [float(x) for x in s.split("|")]
    except ValueError:
        return None
    return np.array(v) if len(v) >= n else None


def side_from_geometry(h):
    cx = {}
    for r in h.itertuples(index=False):
        ipp = _hdr_vec(getattr(r, "ImagePositionPatient", None), 3)
        iop = _hdr_vec(getattr(r, "ImageOrientationPatient", None), 6)
        ps = _hdr_vec(getattr(r, "PixelSpacing", None), 2)
        rows, cols = getattr(r, "Rows", None), getattr(r, "Columns", None)
        if ipp is None or iop is None or ps is None or not rows or not cols:
            continue
        try:
            c = ipp[:3] + iop[:3] * ps[1] * float(cols) / 2 + iop[3:6] * ps[0] * float(rows) / 2
        except (TypeError, ValueError):
            continue
        cx.setdefault(r.StudyInstanceUID, []).append(float(c[0]))
    out = {}
    for st, xs in cx.items():
        m = float(np.median(xs))
        out[st] = None if abs(m) < LAT_MIN_OFFSET_MM else ("R" if m < 0 else "L")
    return out


def lat_of(h):
    geo = side_from_geometry(h)
    d = {}
    for st, g in h.groupby("StudyInstanceUID"):
        v = [str(x).strip().upper() for x in g["Laterality"].dropna()]
        v = [x[0] for x in v if x and x[0] in ("L", "R")]
        d[st] = v[0] if v else geo.get(st)
    return d


def probe(item):
    split, study, series, path = item
    row = {"split": split, "StudyInstanceUID": study, "SeriesInstanceUID": series, "dir": path}
    try:
        files = sorted(e.name for e in os.scandir(path) if e.name.endswith(".dcm"))
        row["files"] = files
        row["n_slices"] = len(files)
        if not files:
            return row
        ds = pydicom.dcmread(os.path.join(path, files[len(files) // 2]), stop_before_pixels=True, force=True)
        for t in HDR_TAGS:
            v = getattr(ds, t, None)
            if v is None:
                row[t] = None
            elif isinstance(v, (list, tuple)) or type(v).__name__ == "MultiValue":
                row[t] = "|".join(str(x) for x in v)
            else:
                row[t] = str(v)
    except Exception as exc:
        row["err"] = str(exc)[:120]
    return row


def walk(root: Path, split: str):
    base = root / split
    items = []
    if not base.is_dir():
        return pd.DataFrame(columns=["split", "StudyInstanceUID", "SeriesInstanceUID",
                                     "dir", "files", "n_slices"] + HDR_TAGS)
    for study in os.scandir(base):
        if study.is_dir():
            for series in os.scandir(study.path):
                if series.is_dir():
                    items.append((split, study.name, series.name, series.path))
    with ThreadPoolExecutor(max_workers=HDR_THREADS) as pool:
        rows = list(pool.map(probe, items))
    return pd.DataFrame(rows)


def annotate(df):
    desc = (df["SeriesDescription"].fillna("") + " " + df["SequenceName"].fillna(""))
    desc = desc.str.lower().str.replace(_SEP, " ", regex=True)
    opts = df["ScanOptions"].fillna("").str.upper().str.split("|")
    opts_fs = opts.apply(lambda ts: any(t.strip() in FATSAT_OPTS for t in ts))
    df["fatsat"] = desc.str.contains(_FATSAT_RX) | opts_fs
    tr = pd.to_numeric(df["RepetitionTime"], errors="coerce")
    te = pd.to_numeric(df["EchoTime"], errors="coerce")
    gre = df["ScanningSequence"].fillna("").str.upper().str.contains("GR")
    t1, t2, pdw = desc.str.contains(_T1_RX), desc.str.contains(_T2_RX), desc.str.contains(_PD_RX)
    df["weight"] = np.where(
        t1 & ~t2 & ~pdw, "T1",
        np.where(t2 & ~pdw, "T2",
        np.where(pdw, "PD",
        np.where(gre, "GRE",
        np.where(tr < 800, "T1",
        np.where(te > 60, "T2",
        np.where(tr >= 800, "PD", "UNK")))))))
    df["fluid"] = np.isin(df["weight"], ["PD", "T2"])
    df["px"] = pd.to_numeric(
        df["PixelSpacing"].fillna("").str.split("|").str[0].replace("", np.nan), errors="coerce")
    return df


def pick_slots(series_df, plane_map):
    series_df = series_df.copy()
    series_df["plane"] = series_df["SeriesInstanceUID"].map(plane_map)
    out = {}
    for study, g in series_df.groupby("StudyInstanceUID"):
        chosen = {}
        for name, plane, fluid, fs in SLOTS:
            sel = (g["plane"] == plane) & (g["fatsat"] == fs)
            if fluid is not None:
                sel &= (g["fluid"] == fluid)
            cand = g[sel]
            if len(cand):
                chosen[name] = cand.sort_values("n_slices", ascending=False).iloc[0]
        out[study] = chosen
    return out


# =============================================================================
# DICOM 像素读取
# =============================================================================
def order_slices(rec):
    files, d = rec["files"], rec["dir"]
    keyed = []
    for f in files:
        k = None
        try:
            ds = pydicom.dcmread(os.path.join(d, f), force=True, stop_before_pixels=True,
                                 specific_tags=ORDER_TAGS)
            iop = np.asarray(ds.ImageOrientationPatient, dtype=float)
            ipp = np.asarray(ds.ImagePositionPatient, dtype=float)
            k = float(np.dot(ipp, np.cross(iop[:3], iop[3:])))
        except Exception:
            try:
                k = float(ds.InstanceNumber)
            except Exception:
                k = None
        keyed.append((k, f))
    if any(k is None for k, _ in keyed):
        return files, False
    return [f for _, f in sorted(keyed, key=lambda t: t[0])], True


def read_slot(rec, n_slice=GROUP, out_size=LOAD_IMG):
    if "ordered" not in rec:
        rec["ordered"], _ = order_slices(rec)
    files, d, px = rec["ordered"], rec["dir"], rec["px"]
    n = len(files)
    if n == 0:
        return None
    lo, hi = int(SLICE_BAND[0] * (n - 1)), int(SLICE_BAND[1] * (n - 1))
    idx = np.unique(np.linspace(lo, hi, n_slice).astype(int)) if hi > lo else np.array([n // 2])
    while len(idx) < n_slice:
        idx = np.append(idx, idx[-1])
    planes = []
    for i in idx[:n_slice]:
        try:
            ds = pydicom.dcmread(os.path.join(d, files[int(i)]), force=True)
            a = ds.pixel_array.astype(np.float32)
            a = a * float(getattr(ds, "RescaleSlope", 1) or 1) + float(getattr(ds, "RescaleIntercept", 0) or 0)
        except Exception:
            a = None
        planes.append(a)
    got = [k for k, p in enumerate(planes) if p is not None]
    if not got:
        return None
    for k, p in enumerate(planes):
        if p is None:
            planes[k] = planes[min(got, key=lambda j: abs(j - k))]
    shp = planes[0].shape
    planes = [p if p.shape == shp else np.zeros(shp, np.float32) for p in planes]
    vol = np.stack(planes)
    if px and np.isfinite(px) and px > 0:
        want = int(round(CROP_MM / px))
        h, w = shp
        if 16 < want < min(h, w):
            cy, cx = h // 2, w // 2
            half = want // 2
            vol = vol[:, max(0, cy - half):cy + half, max(0, cx - half):cx + half]
    lo_v, hi_v = np.percentile(vol, [1, 99])
    vol = np.clip((vol - lo_v) / max(hi_v - lo_v, 1e-6), 0, 1)
    t = torch.from_numpy(np.ascontiguousarray(vol)).unsqueeze(0)
    t = F.interpolate(t, size=(out_size, out_size), mode="bilinear", align_corners=False)
    return (t.squeeze(0) * 255).round().clamp(0, 255).to(torch.uint8)


def normalise_laterality(img, plane, lat):
    if lat != "R":
        return img
    if plane in ("Coronal", "Axial"):
        return torch.flip(img, dims=[-1])
    return torch.flip(img, dims=[0])


def load_study(study_id, slot_map, lat_map, n_slice=GROUP, out_size=LOAD_IMG):
    imgs = torch.zeros(N_SLOT, n_slice, out_size, out_size, dtype=torch.uint8)
    mask = torch.zeros(N_SLOT, dtype=torch.float32)
    chosen = slot_map.get(study_id)
    if not chosen:
        return imgs, mask
    lat = lat_map.get(study_id)
    for k, (name, plane, _, _) in enumerate(SLOTS):
        rec = chosen.get(name)
        if rec is None:
            continue
        img = read_slot(rec, n_slice, out_size)
        if img is None:
            continue
        imgs[k] = normalise_laterality(img, plane, lat)
        mask[k] = 1.0
    return imgs, mask


# =============================================================================
# Dataset
# =============================================================================
class KneeStudyDataset(Dataset):
    def __init__(self, studies, slot_map, lat_map, img_size=LOAD_IMG, n_slices=GROUP):
        self.studies = studies
        self.slot_map = slot_map
        self.lat_map = lat_map
        self.img_size = img_size
        self.n_slices = n_slices

    def __len__(self):
        return len(self.studies)

    def __getitem__(self, idx):
        study = self.studies[idx]
        imgs, mask = load_study(study, self.slot_map, self.lat_map, self.n_slices, self.img_size)
        return {"imgs": imgs, "mask": mask, "study": study}


def collate_studies(batch):
    return {
        "imgs": torch.stack([b["imgs"] for b in batch]),
        "mask": torch.stack([b["mask"] for b in batch]),
        "study": [b["study"] for b in batch],
    }


# =============================================================================
# 模型
# =============================================================================
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
        if prior:
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


def build_model(name, dinov2_path, unfreeze_last=UNFREEZE_LAST):
    if name not in MODELS:
        raise ValueError(f"unknown model {name!r}")
    _, pool, prior = MODELS[name]
    from transformers import AutoModel
    bb = AutoModel.from_pretrained(str(dinov2_path))
    n_layer = len(bb.encoder.layer)
    for p in bb.parameters():
        p.requires_grad = False
    for blk in bb.encoder.layer[max(0, n_layer - unfreeze_last):]:
        for p in blk.parameters():
            p.requires_grad = True
    for p in bb.layernorm.parameters():
        p.requires_grad = True
    return Model(bb, bb.config.hidden_size, pool=pool, prior=prior)


def load_model(weights_path, dinov2_path, device):
    ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
    name = ckpt.get("model_name", "dinov2-small")
    img_size = int(ckpt.get("img_size", 224))
    model = build_model(name, dinov2_path, int(ckpt.get("unfreeze_last", UNFREEZE_LAST)))
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    print(f"loaded {name}  img_size={img_size}", flush=True)
    return model, img_size


@torch.no_grad()
def predict(model, dataset, device, img_size):
    loader = DataLoader(dataset, batch_size=EVAL_BATCH, shuffle=False, collate_fn=collate_studies)
    out = []
    for batch in loader:
        imgs = batch["imgs"].to(device)
        mask = batch["mask"].to(device)
        with torch.autocast("cuda", enabled=device.type == "cuda"):
            z = model(imgs, mask, img_size).float()
        out.append(torch.sigmoid(z).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, len(TARGETS)), np.float32)


def write_submission(pred, studies, test_df, path):
    sub = pd.DataFrame(pd.DataFrame(pred).rank(pct=True).values, columns=TARGETS)
    sub.insert(0, "StudyInstanceUID", studies)
    sub = test_df[["StudyInstanceUID"]].merge(sub, on="StudyInstanceUID", how="left")
    sub[TARGETS] = sub[TARGETS].fillna(0.5)
    sub.to_csv(path, index=False)
    return sub


# =============================================================================
# 主流程
# =============================================================================
def main():
    root = DATA_ROOT.resolve()
    print(f"data={root}", flush=True)
    print(f"weights={WEIGHTS_PATH}", flush=True)
    print(f"dinov2={DINOv2_PATH}", flush=True)

    test_df = pd.read_csv(root / "test.csv")
    test_series = pd.read_csv(root / "test_series.csv")
    plane_map = dict(zip(test_series["SeriesInstanceUID"], test_series["Anatomical_Plane"]))

    hte = annotate(walk(root, "test_series"))
    print(f"test series: {len(hte)}", flush=True)
    slots_te = pick_slots(hte, plane_map)
    st_te = sorted(slots_te)
    lat_te = lat_of(hte)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, img_size = load_model(WEIGHTS_PATH, DINOv2_PATH, device)

    test_ds = KneeStudyDataset(st_te, slots_te, lat_te, img_size=max(img_size, LOAD_IMG))
    pred = predict(model, test_ds, device, img_size)
    sub = write_submission(pred, st_te, test_df, OUTPUT_PATH)
    print(f"wrote {OUTPUT_PATH}  shape={sub.shape}", flush=True)


if __name__ == "__main__":
    main()
