"""
RSNA Knee — Kaggle 单文件提交
整份粘贴到 Notebook 一个单元格即可运行。不要 import 本仓库其它文件。

只需改下面四个路径。
"""
from __future__ import annotations

import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoModel

# ---------------------------------------------------------------------------
# 路径（按你的 Kaggle Dataset 改）
# ---------------------------------------------------------------------------
DATA_ROOT = Path("/kaggle/input/rsna-knee-abnormality-detection")
WEIGHTS = Path("/kaggle/input/your-weights")          # 单个 .pt，或含 model_fold*.pt 的目录
DINOV2_PATH = Path("/kaggle/input/dinov2-small")
OUTPUT_PATH = Path("/kaggle/working/submission.csv")

# ---------------------------------------------------------------------------
# 与训练一致的常量
# ---------------------------------------------------------------------------
TARGETS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
    "Medial OA", "Lateral OA", "PF OA", "Effusion", "Synovitis",
    "Baker's", "Contusion", "Fracture",
]
SLOTS = [
    ("SAG_FLUID_FS", "Sagittal", True, True),
    ("COR_FLUID_FS", "Coronal", True, True),
    ("AX_FLUID_FS", "Axial", True, True),
    ("SAG_FLUID_NOFS", "Sagittal", True, False),
    ("COR_T1", "Coronal", False, False),
    ("SAG_T1", "Sagittal", False, False),
]
N_SLOT = len(SLOTS)
CROP_MM, LOAD_IMG, GROUP, IMG_SIZE = 130.0, 336, 5, 224  # GROUP 与训练 config 一致
DECODE_SIZE = max(IMG_SIZE, LOAD_IMG)
EVAL_BATCH, HDR_THREADS, NUM_WORKERS = 8, 16, 2
LAT_MIN_OFFSET_MM = 20.0
SLICE_BAND = (0.20, 0.80)

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


# ---------------------------------------------------------------------------
# 1) 扫 DICOM 头，给每个 study 选 6 个槽
# ---------------------------------------------------------------------------
def _vec(s, n):
    if not isinstance(s, str):
        return None
    try:
        v = [float(x) for x in s.split("|")]
    except ValueError:
        return None
    return np.array(v) if len(v) >= n else None


def _probe(item):
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
    except Exception as e:
        row["err"] = str(e)[:120]
    return row


def walk_headers(root: Path, split: str) -> pd.DataFrame:
    base = root / split
    items = []
    for study in os.scandir(base):
        if not study.is_dir():
            continue
        for series in os.scandir(study.path):
            if series.is_dir():
                items.append((split, study.name, series.name, series.path))
    print(f"[1] {split}: {len(items)} series", flush=True)
    with ThreadPoolExecutor(max_workers=HDR_THREADS) as pool:
        rows = list(tqdm(pool.map(_probe, items), total=len(items), desc="headers"))
    return pd.DataFrame(rows)


def annotate(df: pd.DataFrame) -> pd.DataFrame:
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
                sel &= g["fluid"] == fluid
            cand = g[sel]
            if len(cand):
                chosen[name] = cand.sort_values("n_slices", ascending=False).iloc[0]
        out[study] = chosen
    return out


def laterality(h):
    cx = {}
    for r in h.itertuples(index=False):
        ipp, iop = _vec(getattr(r, "ImagePositionPatient", None), 3), _vec(getattr(r, "ImageOrientationPatient", None), 6)
        ps = _vec(getattr(r, "PixelSpacing", None), 2)
        rows, cols = getattr(r, "Rows", None), getattr(r, "Columns", None)
        if ipp is None or iop is None or ps is None or not rows or not cols:
            continue
        try:
            c = ipp[:3] + iop[:3] * ps[1] * float(cols) / 2 + iop[3:6] * ps[0] * float(rows) / 2
        except (TypeError, ValueError):
            continue
        cx.setdefault(r.StudyInstanceUID, []).append(float(c[0]))
    geo = {}
    for st, xs in cx.items():
        m = float(np.median(xs))
        geo[st] = None if abs(m) < LAT_MIN_OFFSET_MM else ("R" if m < 0 else "L")
    out = {}
    for st, g in h.groupby("StudyInstanceUID"):
        v = [str(x).strip().upper() for x in g["Laterality"].dropna()]
        v = [x[0] for x in v if x and x[0] in ("L", "R")]
        out[st] = v[0] if v else geo.get(st)
    return out


# ---------------------------------------------------------------------------
# 2) 读像素：每个槽抽 GROUP 张切片，裁切、归一化、翻到左侧
# ---------------------------------------------------------------------------
def order_slices(rec):
    files, d = rec["files"], rec["dir"]
    keyed = []
    for f in files:
        k = None
        try:
            ds = pydicom.dcmread(os.path.join(d, f), force=True, stop_before_pixels=True, specific_tags=ORDER_TAGS)
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
        return files
    return [f for _, f in sorted(keyed, key=lambda t: t[0])]


def read_slot(rec, n_slice=GROUP, out_size=LOAD_IMG):
    files = rec.get("ordered") or order_slices(rec)
    rec["ordered"] = files
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
            ds = pydicom.dcmread(os.path.join(rec["dir"], files[int(i)]), force=True)
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
    px = rec["px"]
    if px and np.isfinite(px) and px > 0:
        want = int(round(CROP_MM / px))
        h, w = shp
        if 16 < want < min(h, w):
            cy, cx, half = h // 2, w // 2, want // 2
            vol = vol[:, max(0, cy - half):cy + half, max(0, cx - half):cx + half]
    lo_v, hi_v = np.percentile(vol, [1, 99])
    vol = np.clip((vol - lo_v) / max(hi_v - lo_v, 1e-6), 0, 1)
    t = torch.from_numpy(np.ascontiguousarray(vol)).unsqueeze(0)
    t = F.interpolate(t, size=(out_size, out_size), mode="bilinear", align_corners=False)
    return (t.squeeze(0) * 255).round().clamp(0, 255).to(torch.uint8)


def load_study(study_id, slot_map, lat_map):
    imgs = torch.zeros(N_SLOT, GROUP, DECODE_SIZE, DECODE_SIZE, dtype=torch.uint8)
    mask = torch.zeros(N_SLOT, dtype=torch.float32)
    chosen = slot_map.get(study_id)
    if not chosen:
        return imgs, mask
    lat = lat_map.get(study_id)
    for k, (name, plane, _, _) in enumerate(SLOTS):
        rec = chosen.get(name)
        if rec is None:
            continue
        img = read_slot(rec, n_slice=GROUP, out_size=DECODE_SIZE)
        if img is None:
            continue
        if lat == "R":
            img = torch.flip(img, dims=[-1] if plane in ("Coronal", "Axial") else [0])
        imgs[k] = img
        mask[k] = 1.0
    return imgs, mask


class KneeDS(Dataset):
    def __init__(self, studies, slot_map, lat_map):
        self.studies, self.slot_map, self.lat_map = studies, slot_map, lat_map

    def __len__(self):
        return len(self.studies)

    def __getitem__(self, i):
        st = self.studies[i]
        imgs, mask = load_study(st, self.slot_map, self.lat_map)
        return {"imgs": imgs, "mask": mask, "study": st}


def collate(batch):
    return {
        "imgs": torch.stack([b["imgs"] for b in batch]),
        "mask": torch.stack([b["mask"] for b in batch]),
        "study": [b["study"] for b in batch],
    }


# ---------------------------------------------------------------------------
# 3) 模型（与当前训练 MeanHead 一致；若权重是旧 SlotHead 则自动切换）
# ---------------------------------------------------------------------------
class MeanHead(nn.Module):
    def __init__(self, dim, n_slot, n_out, hidden=256, p=0.2):
        super().__init__()
        self.proj = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, hidden), nn.GELU())
        self.slot_emb = nn.Parameter(torch.randn(n_slot, hidden) * 0.02)
        self.drop = nn.Dropout(p)
        self.out = nn.Linear(hidden, n_out)

    def forward(self, x, mask):
        h = self.proj(x) + self.slot_emb
        w = mask.unsqueeze(-1).clamp(min=0)
        mean = (h * w).sum(1) / w.sum(1).clamp_min(1e-6)
        return self.out(self.drop(mean))


class SlotHead(nn.Module):
    def __init__(self, dim, n_slot, n_out, hidden=256, p=0.2):
        super().__init__()
        self.proj = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, hidden), nn.GELU())
        self.slot_emb = nn.Parameter(torch.randn(n_slot, hidden) * 0.02)
        self.query = nn.Parameter(torch.randn(n_out, hidden) * 0.02)
        self.drop = nn.Dropout(p)
        self.out = nn.Linear(hidden, n_out)
        self.hidden = hidden

    def forward(self, x, mask):
        h = self.proj(x) + self.slot_emb
        att = torch.einsum("bsh,oh->bos", h, self.query) / self.hidden ** 0.5
        att = att.masked_fill(mask.unsqueeze(1) < 0.5, -1e4).softmax(-1)
        ctx = self.drop(torch.einsum("bos,bsh->boh", att, h))
        return (ctx * self.out.weight.unsqueeze(0)).sum(-1) + self.out.bias


class Model(nn.Module):
    def __init__(self, backbone, dim, use_slot_head=False):
        super().__init__()
        self.backbone = backbone
        Head = SlotHead if use_slot_head else MeanHead
        self.head = Head(dim * 2, N_SLOT, len(TARGETS))
        self.register_buffer("mean3", torch.tensor([0.485, 0.456, 0.406]))
        self.register_buffer("std3", torch.tensor([0.229, 0.224, 0.225]))

    def forward(self, imgs, mask, img_size=IMG_SIZE):
        b, s, g, h, w = imgs.shape
        x = imgs.float().div_(255.0)
        if img_size != h:
            x = F.interpolate(
                x.reshape(b * s * g, 1, h, w),
                size=(img_size, img_size),
                mode="bilinear",
                align_corners=False,
            )
            h = w = img_size
        else:
            x = x.reshape(b * s * g, 1, h, w)

        if g == 3:
            x = x.reshape(b * s, 3, h, w)
        else:
            x = x.repeat(1, 3, 1, 1)

        mean = self.mean3.view(1, 3, 1, 1)
        std = self.std3.view(1, 3, 1, 1)
        x = (x - mean) / std
        out = self.backbone(pixel_values=x).last_hidden_state
        feat = torch.cat([out[:, 0], out[:, 1:].mean(1)], dim=1)
        if g == 3:
            feat = feat.reshape(b, s, -1)
        else:
            feat = feat.reshape(b, s, g, -1).mean(2)
        return self.head(feat, mask)


def list_weights(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    files = sorted(path.glob("*.pt"))
    if not files:
        raise FileNotFoundError(f"没有找到权重: {path}")
    return files


def count_gpus() -> int:
    vis = os.environ.get("CUDA_VISIBLE_DEVICES")
    if vis is not None:
        ids = [x.strip() for x in vis.split(",") if x.strip() and x.strip() != "-1"]
        return len(ids)
    if not os.path.isdir("/dev"):
        return 0
    return sum(1 for p in os.listdir("/dev") if p.startswith("nvidia") and p[6:].isdigit())


def load_one(weights_path: Path, device):
    ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    bb = AutoModel.from_pretrained(str(DINOV2_PATH), local_files_only=True)
    use_slot = any(k.startswith("head.query") for k in state)
    model = Model(bb, bb.config.hidden_size, use_slot_head=use_slot)
    model.load_state_dict(state)
    model.to(device).eval()
    img_size = int(ckpt.get("img_size", IMG_SIZE)) if isinstance(ckpt, dict) else IMG_SIZE
    print(f"  loaded {weights_path.name}  head={'slot' if use_slot else 'mean'}  img={img_size}  {device}", flush=True)
    return model, img_size


@torch.no_grad()
def predict(models, dataset, device):
    # Notebook 单元格无法 pickle Dataset 给 spawn worker；模型已占 CUDA 也不能 fork
    nw = 0 if "ipykernel" in sys.modules else NUM_WORKERS
    loader = DataLoader(
        dataset, batch_size=EVAL_BATCH, shuffle=False, collate_fn=collate,
        num_workers=nw, pin_memory=device.type == "cuda",
        persistent_workers=nw > 0,
        prefetch_factor=2 if nw > 0 else None,
        multiprocessing_context="spawn" if nw > 0 else None,
    )
    n = len(models)
    out = []
    for batch in tqdm(loader, desc="predict"):
        imgs = batch["imgs"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        acc = None
        for model, img_size in models:
            with torch.autocast("cuda", enabled=device.type == "cuda"):
                z = torch.sigmoid(model(imgs, mask, img_size).float())
            acc = z if acc is None else acc + z
        out.append((acc / n).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, len(TARGETS)), np.float32)


def run_on_gpu(rank, n_gpu, studies, slot_map, lat_map, weight_paths, bag):
    idx = list(range(rank, len(studies), n_gpu))
    shard = [studies[i] for i in idx]
    if not shard:
        bag[rank] = (idx, np.zeros((0, len(TARGETS)), np.float32))
        return
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")
    else:
        device = torch.device("cpu")
    models = [load_one(Path(wp), device) for wp in weight_paths]
    pred = predict(models, KneeDS(shard, slot_map, lat_map), device)
    bag[rank] = (idx, pred)


# ---------------------------------------------------------------------------
# 4) 主流程
# ---------------------------------------------------------------------------
def main():
    print(f"data={DATA_ROOT}\nweights={WEIGHTS}\ndinov2={DINOV2_PATH}", flush=True)
    test_df = pd.read_csv(DATA_ROOT / "test.csv")
    test_series = pd.read_csv(DATA_ROOT / "test_series.csv")
    plane_map = dict(zip(test_series["SeriesInstanceUID"], test_series["Anatomical_Plane"]))

    hte = annotate(walk_headers(DATA_ROOT, "test_series"))
    slots = pick_slots(hte, plane_map)
    studies = sorted(slots)
    lat = laterality(hte)
    slots = {st: {k: (r.to_dict() if hasattr(r, "to_dict") else r) for k, r in ch.items()} for st, ch in slots.items()}
    print(f"[2] studies={len(studies)}  test.csv={len(test_df)}", flush=True)

    wts = [str(p) for p in list_weights(WEIGHTS)]
    n_gpu = max(count_gpus(), 1)
    print(f"[3] {len(wts)} checkpoint(s)  gpu={n_gpu}", flush=True)

    bag = [None] * n_gpu
    if n_gpu == 1:
        run_on_gpu(0, 1, studies, slots, lat, wts, bag)
    else:
        # Notebook 里 mp.spawn 无法 pickle 单元格函数；每卡一个线程，study 仍按卡切分
        with ThreadPoolExecutor(max_workers=n_gpu) as pool:
            futs = [
                pool.submit(run_on_gpu, r, n_gpu, studies, slots, lat, wts, bag)
                for r in range(n_gpu)
            ]
            for f in futs:
                f.result()

    pred = np.zeros((len(studies), len(TARGETS)), np.float32)
    for idx, p in bag:
        pred[list(idx)] = p

    # 按 test.csv 行序写出；缺的 study 填 0.5。rank(pct) 不改变 AUC，只把分数拉到 (0,1)
    scored = pd.DataFrame(pd.DataFrame(pred, columns=TARGETS).rank(pct=True).values, columns=TARGETS)
    scored.insert(0, "StudyInstanceUID", studies)
    sub = test_df[["StudyInstanceUID"]].merge(scored, on="StudyInstanceUID", how="left")
    sub[TARGETS] = sub[TARGETS].fillna(0.5)
    sub = sub[["StudyInstanceUID"] + TARGETS]
    sub.to_csv(OUTPUT_PATH, index=False)
    print(f"[4] wrote {OUTPUT_PATH}  shape={sub.shape}", flush=True)


if __name__ == "__main__":
    main()
