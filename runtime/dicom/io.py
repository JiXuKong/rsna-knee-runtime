"""DICOM slice ordering, decoding, and on-demand study loading."""
from __future__ import annotations

import os
import re

import numpy as np
import pydicom
import torch
import torch.nn.functional as F

from runtime.config import CROP_MM, GROUP, LOAD_IMG, ORDER_TAGS, RULES, SLICE_BAND, SLICE_MODE, SLOTS

N_SLOT = len(SLOTS)
DECODE_FAILED: list[str] = []


def _natural_key(name):
    return tuple(int(x) if x.isdigit() else x.lower() for x in re.split(r"(\d+)", str(name)))


def _order_dominant_axis(rec):
    files, d = rec["files"], rec["dir"]
    rows = []
    for pos, f in enumerate(files):
        ipp = inst = None
        try:
            ds = pydicom.dcmread(
                os.path.join(d, f),
                force=True,
                stop_before_pixels=True,
                specific_tags=["ImagePositionPatient", "InstanceNumber"],
            )
            raw = getattr(ds, "ImagePositionPatient", None)
            if raw is not None and len(raw) >= 3:
                c = np.asarray(raw[:3], dtype=np.float64)
                if np.isfinite(c).all():
                    ipp = c
            n = getattr(ds, "InstanceNumber", None)
            if n is not None:
                inst = float(n)
        except Exception:
            pass
        rows.append((f, ipp, inst, pos))

    placed = [r for r in rows if r[1] is not None]
    need = max(2, int(0.8 * len(rows)))
    if len(placed) >= need:
        xyz = np.stack([r[1] for r in placed])
        axis = int(np.argmax(np.ptp(xyz, axis=0)))
        spare = float(np.nanmedian(xyz[:, axis]))
        rows.sort(
            key=lambda r: (
                float(r[1][axis]) if r[1] is not None else spare,
                r[2] if r[2] is not None else float("inf"),
                r[3],
            )
        )
    elif sum(r[2] is not None for r in rows) >= need:
        rows.sort(key=lambda r: (r[2] if r[2] is not None else float("inf"), r[3]))
    else:
        rows.sort(key=lambda r: _natural_key(r[0]))
    return [r[0] for r in rows], True


def order_slices(rec):
    if RULES["order"] == "dominant_axis":
        return _order_dominant_axis(rec)
    files, d = rec["files"], rec["dir"]
    keyed = []
    for f in files:
        k = None
        try:
            ds = pydicom.dcmread(
                os.path.join(d, f),
                force=True,
                stop_before_pixels=True,
                specific_tags=ORDER_TAGS,
            )
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


def _ensure_ordered(rec: dict) -> dict:
    if "ordered" not in rec:
        files, _ = order_slices(rec)
        rec["ordered"] = files
    return rec


def read_slot(rec, n_slice=None, out_size=None):
    n_slice = GROUP if n_slice is None else n_slice
    out_size = LOAD_IMG if out_size is None else out_size
    rec = _ensure_ordered(rec)
    files, d, px = rec["ordered"], rec["dir"], rec["px"]
    n = len(files)
    if n == 0:
        return None

    if SLICE_MODE == "center":
        c = n // 2
        start = max(0, min(c - n_slice // 2, n - n_slice))
        idx = np.arange(start, min(start + n_slice, n), dtype=int)
    else:
        lo, hi = int(SLICE_BAND[0] * (n - 1)), int(SLICE_BAND[1] * (n - 1))
        idx = np.unique(np.linspace(lo, hi, n_slice).astype(int)) if hi > lo else np.array([n // 2])
    while len(idx) < n_slice:
        idx = np.append(idx, idx[-1])

    planes = []
    for i in idx[:n_slice]:
        try:
            ds = pydicom.dcmread(os.path.join(d, files[int(i)]), force=True)
            a = ds.pixel_array.astype(np.float32)
            sl = float(getattr(ds, "RescaleSlope", 1) or 1)
            ic = float(getattr(ds, "RescaleIntercept", 0) or 0)
            a = a * sl + ic
        except Exception:
            a = None
        planes.append(a)

    got = [k for k, p in enumerate(planes) if p is not None]
    if RULES["decode_fill"] == "zero":
        if not got:
            DECODE_FAILED.append(rec.get("SeriesInstanceUID", d))
        planes = [
            np.zeros((out_size, out_size), np.float32) if p is None else p for p in planes
        ]
        got = list(range(len(planes)))
    if not got:
        DECODE_FAILED.append(rec.get("SeriesInstanceUID", d))
        return None
    if len(got) < len(planes):
        DECODE_FAILED.append(rec.get("SeriesInstanceUID", d))
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
            vol = vol[:, max(0, cy - half) : cy + half, max(0, cx - half) : cx + half]

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
    """Load one study: imgs [n_slot, n_slice, H, W], mask [n_slot]."""
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


def fetch_batch(indices, studies, slot_map, lat_map, n_slice, out_size):
    """Load a mini-batch into arrays shaped like the old in-memory cache."""
    b = len(indices)
    cache = np.zeros((b, N_SLOT, n_slice, out_size, out_size), np.uint8)
    mask = np.zeros((b, N_SLOT), np.float32)
    for bi, idx in enumerate(indices):
        study = studies[idx]
        imgs, m = load_study(study, slot_map, lat_map, n_slice, out_size)
        cache[bi] = imgs.numpy()
        mask[bi] = m.numpy()
    return cache, mask
