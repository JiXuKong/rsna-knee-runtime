"""Prepare DICOM slot maps for train / infer."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from runtime.dicom.headers import annotate, lat_of, walk
from runtime.dicom.slots import pick_slots


def prepare_slot_maps(root: Path):
    test_df = pd.read_csv(root / "test.csv")
    test_series = pd.read_csv(root / "test_series.csv")
    train_df = pd.read_csv(root / "train.csv")
    train_series = pd.read_csv(root / "train_series.csv")
    print(f"train {train_df.shape}  test {test_df.shape}", flush=True)

    both = pd.concat([train_series, test_series])
    plane_map = dict(zip(both["SeriesInstanceUID"], both["Anatomical_Plane"]))

    htr = annotate(walk(root, "train_series"))
    hte = annotate(walk(root, "test_series"))
    print(f"series: train {len(htr)}  test {len(hte)}", flush=True)

    slots_tr = pick_slots(htr, plane_map)
    slots_te = pick_slots(hte, plane_map)
    return {
        "train_df": train_df,
        "test_df": test_df,
        "st_tr": sorted(slots_tr),
        "st_te": sorted(slots_te),
        "slots_tr": slots_tr,
        "slots_te": slots_te,
        "lat_tr": lat_of(htr, "train "),
        "lat_te": lat_of(hte, "test "),
    }
