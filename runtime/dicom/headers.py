"""DICOM header parsing and series annotation."""
from __future__ import annotations

import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
from tqdm import tqdm

from runtime.config import (
    FATSAT_OPTS,
    HDR_TAGS,
    HDR_THREADS,
    LAT_MIN_OFFSET_MM,
    LEGACY_LAT_OFFSET_MM,
    RULES,
    _FATSAT_RX,
    _PD_RX,
    _SEP,
    _T1_RX,
    _T2_RX,
)

def _hdr_vec(s, n):
    """Parse a DICOM multi-value string as stored by probe(): floats joined by `|`."""
    if not isinstance(s, str):
        return None
    try:
        v = [float(x) for x in s.split("|")]
    except ValueError:
        return None
    return np.array(v) if len(v) >= n else None


def side_from_geometry(h):
    """Study -> 'L' / 'R' / None, from where the image sits in the patient.

    `Laterality` (0020,0060) is Type 2C and may legitimately be absent; in this corpus it
    is missing on exactly half the studies, and the vendors it is missing from are whole
    vendors rather than scattered series. A study with no tag is not a left knee, but the
    normalisation upstream treats it as one, so half the corpus was never normalised and
    the five side-defined targets - the two menisci, the two tibiofemoral compartments
    and the medial collateral ligament - saw that axis reversed on a large minority of it.

    The patient coordinate system fixes this without the tag: +x is the patient's left, so
    the centre of a right knee sits at negative x. The centre is used rather than
    `ImagePositionPatient` itself because that is the corner of the image, which is offset
    by half a field of view - enough to change the sign on a knee near the midline.

    The median over a study's series is what is thresholded, not a single series: probe()
    reads one arbitrary slice per series, which on a sagittal stack can sit anywhere
    across the joint. Studies whose centre falls near the midline are left unresolved
    rather than guessed - measured against the tagged half, the rule is right 97% of the
    time overall and no better than chance inside 20 mm.
    """
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


def side_from_corner_x(h):
    """The laterality an imported member was fitted under.

    It thresholds the median raw `ImagePositionPatient` x over a study's series. That is
    the x of the image *corner*, not of its centre, so it differs from the rule above by
    up to half a field of view - which is enough to reverse the sign on a knee scanned
    near the midline. The dead zone is 5 mm rather than 20 mm, so it also commits on
    studies the rule above leaves unresolved.

    Neither difference changes a shape. Each one decides whether a study is mirrored, and
    a study mirrored one way at training and the other at inference presents the five
    side-defined targets with their axis reversed.
    """
    out = {}
    for st, g in h.groupby("StudyInstanceUID"):
        xs = []
        for r in g.itertuples(index=False):
            ipp = _hdr_vec(getattr(r, "ImagePositionPatient", None), 3)
            if ipp is not None and np.isfinite(ipp).all():
                xs.append(float(ipp[0]))
        if not xs:
            out[st] = None
            continue
        x = float(np.median(xs))
        # DICOM patient coordinates are LPS: +x is the patient's left.
        out[st] = None if abs(x) < LEGACY_LAT_OFFSET_MM else ("R" if x < 0 else "L")
    return out


def lat_of(h, tag=""):
    """Study -> 'L' / 'R' / None: the tag where it exists, geometry where it does not.

    The tag is present on exactly half the studies here and is sometimes an empty
    string rather than absent, which is not the same as NaN. Treating the other half
    as left-sided is what `normalise_laterality` did by omission, so the geometry
    fallback is not a refinement - it is the difference between normalising half the
    corpus and normalising all of it.
    """
    geo = side_from_corner_x(h) if RULES["lat"] == "corner_x" else side_from_geometry(h)
    d, n_tag, n_geo, n_none, n_disagree = {}, 0, 0, 0, 0
    for st, g in h.groupby("StudyInstanceUID"):
        v = [str(x).strip().upper() for x in g["Laterality"].dropna()]
        if RULES["lat"] == "corner_x" and "ImageLaterality" in g.columns:
            # The legacy rule reads the second tag too, so a study tagged only there is
            # resolved from the tag rather than from geometry.
            v += [str(x).strip().upper() for x in g["ImageLaterality"].dropna()]
        v = [x[0] for x in v if x and x[0] in ("L", "R")]
        side = v[0] if v else None
        if side is not None:
            n_tag += 1
            if geo.get(st) is not None and geo[st] != side:
                n_disagree += 1
        else:
            side = geo.get(st)
            n_geo += side is not None
            n_none += side is None
        d[st] = side
    print(f"{tag}laterality: {n_tag} from the tag, {n_geo} from geometry, "
          f"{n_none} unresolved; tag and geometry disagree on {n_disagree} "
          f"({n_disagree / max(n_tag, 1):.1%} of the tagged)", flush=True)
    return d



def probe(item):
    split, study, series, path = item
    row = {"split": split, "StudyInstanceUID": study, "SeriesInstanceUID": series,
           "dir": path}
    try:
        files = sorted(e.name for e in os.scandir(path) if e.name.endswith(".dcm"))
        row["files"] = files
        row["n_slices"] = len(files)
        if not files:
            return row
        ds = pydicom.dcmread(os.path.join(path, files[len(files) // 2]),
                             stop_before_pixels=True, force=True)
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
    """Every series directory of a split, with one header read per series."""
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
    print(f"[walk] {split}: {len(items)} series, probing headers...", flush=True)
    with ThreadPoolExecutor(max_workers=HDR_THREADS) as pool:
        rows = list(tqdm(
            pool.map(probe, items), total=len(items),
            desc=f"DICOM {split}", unit="series", mininterval=1.0,
        ))
    return pd.DataFrame(rows)


def annotate(df):
    """从 DICOM 头恢复压脂标志、加权类型，供后面选槽使用。"""
    # 序列描述 + 序列名拼成一段文本；缺省当空串
    desc = (df["SeriesDescription"].fillna("") + " " + df["SequenceName"].fillna(""))
    # 转小写，把 _ - . 换成空格，方便正则匹配 t1 / fatsat 等
    desc = desc.str.lower().str.replace(_SEP, " ", regex=True)

    # ScanOptions 是多值，probe() 用 | 拼过；转大写再拆成 token 列表
    opts = df["ScanOptions"].fillna("").str.upper().str.split("|")
    # GE 的 SAT_GEMS 是空间饱和不是压脂；必须整词匹配 FS/FATSAT，不能用 "SAT" 子串
    opts_fs = opts.apply(lambda ts: any(t.strip() in FATSAT_OPTS for t in ts))
    # 描述里命中 STIR/SPAIR/FS 等，或 ScanOptions 命中压脂 token → fatsat=True
    df["fatsat"] = desc.str.contains(_FATSAT_RX) | opts_fs

    # TR（重复时间，ms）；转不成数字则为 NaN
    tr = pd.to_numeric(df["RepetitionTime"], errors="coerce")
    # TE（回波时间，ms）
    te = pd.to_numeric(df["EchoTime"], errors="coerce")
    # ScanningSequence 含 GR → 梯度回波 GRE
    gre = df["ScanningSequence"].fillna("").str.upper().str.contains("GR")
    # 描述里是否出现 T1 / T2 / PD 字样（三路布尔列）
    t1, t2, pdw = desc.str.contains(_T1_RX), desc.str.contains(_T2_RX), desc.str.contains(_PD_RX)

    # 加权优先级：名字里的 T1 > T2 > PD > GRE；名字没有再靠 TR/TE 猜
    df["weight"] = np.where(t1 & ~t2 & ~pdw, "T1",          # 只写了 T1
                     np.where(t2 & ~pdw, "T2",              # 写了 T2（且不是 PD）
                       np.where(pdw, "PD",                  # 写了 PD / proton
                         np.where(gre, "GRE",               # 梯度回波
                           np.where(tr < 800, "T1",         # 短 TR → T1
                             np.where(te > 60, "T2",        # 长 TE → T2
                               np.where(tr >= 800, "PD", "UNK")))))))  # 长 TR 当 PD，否则未知
    # PD/T2 对积液、骨髓水肿敏感，当作「液体序列」去填 FLUID 槽
    df["fluid"] = np.isin(df["weight"], ["PD", "T2"])
    # 像素间距取第一维（行方向 mm）；后面按 130mm 视野中心裁切要用
    df["px"] = pd.to_numeric(
        df["PixelSpacing"].fillna("").str.split("|").str[0].replace("", np.nan),
        errors="coerce")
    return df