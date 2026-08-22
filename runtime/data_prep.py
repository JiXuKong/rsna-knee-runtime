"""Prepare DICOM slot maps for train / infer."""
from __future__ import annotations

import os
import pickle
from pathlib import Path

import pandas as pd

from runtime.dicom.headers import annotate, lat_of, walk
from runtime.dicom.slots import pick_slots


def _load(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def _save(path, obj):
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[prep] saved {path.name}", flush=True)


def prepare_slot_maps(root: Path, *, use_cache: bool = True):
    from runtime.config import CACHE_DIR
    cache_dir = CACHE_DIR / "prep"
    cache_dir.mkdir(parents=True, exist_ok=True)
    ss = os.environ.get("SLOT_SCHEME", "recovered")
    rn = root.name

    # 三段独立缓存
    c_hdrs = cache_dir / f"hdrs_{ss}_{rn}.pkl"
    c_slots = cache_dir / f"slots_{ss}_{rn}.pkl"
    c_lat = cache_dir / f"lat_{ss}_{rn}.pkl"

    print("[prep] loading csv...", flush=True)
    test_df = pd.read_csv(root / "test.csv")
    test_series = pd.read_csv(root / "test_series.csv")
    train_df = pd.read_csv(root / "train.csv")
    train_series = pd.read_csv(root / "train_series.csv")
    plane_map = dict(zip(
        pd.concat([train_series, test_series])["SeriesInstanceUID"],
        pd.concat([train_series, test_series])["Anatomical_Plane"],
    ))

    # 段 1：DICOM header 扫描（最耗时）
    if use_cache and c_hdrs.exists():
        print(f"[prep] load hdrs cache", flush=True)
        htr, hte = _load(c_hdrs)
    else:
        print("[prep] scanning train headers...", flush=True)
        htr = annotate(walk(root, "train_series"))
        print("[prep] scanning test headers...", flush=True)
        hte = annotate(walk(root, "test_series"))
        if use_cache:
            _save(c_hdrs, (htr, hte))
    print(f"[prep] series: train {len(htr)}  test {len(hte)}", flush=True)

    # 段 2：pick_slots（约 28s）
    if use_cache and c_slots.exists():
        print(f"[prep] load slots cache", flush=True)
        slots_tr, slots_te, pid_tr, st_tr, st_te = _load(c_slots)
    else:
        print("[prep] pick_slots train...", flush=True)
        slots_tr = pick_slots(htr, plane_map)
        print("[prep] pick_slots test...", flush=True)
        slots_te = pick_slots(hte, plane_map)
        pid = htr["PatientID"].fillna("").astype(str).str.strip().replace({"nan": "", "None": ""})
        pid = pid.where(pid.ne(""), htr["StudyInstanceUID"])
        st_tr, st_te = sorted(slots_tr), sorted(slots_te)
        pid_tr = dict(zip(htr["StudyInstanceUID"], pid))
        if use_cache:
            _save(c_slots, (slots_tr, slots_te, pid_tr, st_tr, st_te))
    print(f"[prep] studies: train {len(st_tr)}  test {len(st_te)}", flush=True)

    # 段 3：laterality（很快，也缓存省掉打印）
    if use_cache and c_lat.exists():
        print(f"[prep] load lat cache", flush=True)
        lat_tr, lat_te = _load(c_lat)
    else:
        print("[prep] laterality...", flush=True)
        lat_tr = lat_of(htr, "train ")
        lat_te = lat_of(hte, "test ")
        if use_cache:
            _save(c_lat, (lat_tr, lat_te))

    print("[prep] done", flush=True)
    return {
        "train_df": train_df,
        "test_df": test_df,
        "st_tr": st_tr,
        "st_te": st_te,
        "slots_tr": slots_tr,
        "slots_te": slots_te,
        "lat_tr": lat_tr,
        "lat_te": lat_te,
        "pid_tr": pid_tr,
    }
