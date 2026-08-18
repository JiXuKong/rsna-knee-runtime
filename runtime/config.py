"""Shared constants for RSNA knee abnormality detection."""
from __future__ import annotations

import os
import re
from pathlib import Path

# 路径：由 CLI 传入，这里只提供默认值
DATA_ROOT = Path("data")
DINOv2_PATH: Path | None = None

TARGETS = [
    "ACL",
    "MCL",
    "Medial Meniscus",
    "Lateral Meniscus",
    "Medial OA",
    "Lateral OA",
    "PF OA",
    "Effusion",
    "Synovitis",
    "Baker's",
    "Contusion",
    "Fracture",
]

SEED = 2026
CROP_MM = 130.0
LOAD_IMG = 336
GROUP = 3
HDR_THREADS = 16

IMG_SIZE = 224
LAT_MIN_OFFSET_MM = 20.0
SLICE_BAND = (0.20, 0.80)

RULES_NATIVE = {
    "order": "normal",
    "lat": "centre",
    "slot_fallback": False,
    "decode_fill": "nearest",
}
RULES_LEGACY = {
    "order": "dominant_axis",
    "lat": "corner_x",
    "slot_fallback": True,
    "decode_fill": "zero",
}
RULES = dict(RULES_NATIVE)
LEGACY_LAT_OFFSET_MM = 5.0

LR_HEAD = 1e-3
LR_BACKBONE = 8e-6
UNFREEZE_LAST = 6
WEIGHT_DECAY = 0.02
EVAL_BATCH = 8
EPOCHS = 10
BATCH_STUDIES = 8

SLOTS_RECOVERED = [
    ("SAG_FLUID_FS", "Sagittal", True, True),
    ("COR_FLUID_FS", "Coronal", True, True),
    ("AX_FLUID_FS", "Axial", True, True),
    ("SAG_FLUID_NOFS", "Sagittal", True, False),
    ("COR_T1", "Coronal", False, False),
    ("SAG_T1", "Sagittal", False, False),
]

SLOTS_PUBLIC = [
    ("SAG_FLUID", "Sagittal", None, True),
    ("COR_FLUID", "Coronal", None, True),
    ("AX_FLUID", "Axial", None, True),
    ("SAG_STRUCT", "Sagittal", None, False),
    ("COR_STRUCT", "Coronal", None, False),
    ("AX_STRUCT", "Axial", None, False),
]

SLOT_SCHEME = os.environ.get("SLOT_SCHEME", "recovered")
SLOTS = SLOTS_PUBLIC if SLOT_SCHEME == "public" else SLOTS_RECOVERED
N_SLOT = len(SLOTS)

POOL_PARTS = {"cls_mean": 2, "cls_mean_focal": 3}

SLOT_PRIOR_TABLE = {
    "ACL": (0, 3, 5),
    "MCL": (1, 4),
    "Medial Meniscus": (0, 1, 3, 4),
    "Lateral Meniscus": (0, 1, 3, 4),
    "Medial OA": (1, 4, 5),
    "Lateral OA": (1, 4, 5),
    "PF OA": (0, 2, 5),
    "Effusion": (0, 2),
    "Synovitis": (0, 2),
    "Baker's": (0,),
    "Contusion": (0, 1, 2),
    "Fracture": (0, 1, 2, 4, 5),
}
SLOT_PRIOR_STRENGTH = 0.55

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
    "SeriesDescription",
    "SequenceName",
    "ScanOptions",
    "ScanningSequence",
    "RepetitionTime",
    "EchoTime",
    "Laterality",
    "PixelSpacing",
    "Rows",
    "Columns",
    "RescaleSlope",
    "RescaleIntercept",
    "ImagePositionPatient",
    "ImageOrientationPatient",
]
