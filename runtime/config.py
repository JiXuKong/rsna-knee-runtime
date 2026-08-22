"""Shared constants for RSNA knee abnormality detection."""
from __future__ import annotations

import os
import re
from pathlib import Path

_KAGGLE = Path(os.environ.get("KAGGLE_DIR", "/root/autodl-tmp/kaggle"))
_REPO = Path(__file__).resolve().parents[1]

DATA_ROOT = _KAGGLE / "data"
LABELS_PATH = DATA_ROOT / "train_cursor.csv"
DINOV2_PATH = _KAGGLE / "models" / "dinov2-small"
SAVE_PATH = _REPO / "outputs" / "model.pt"
TB_LOG_DIR = _REPO / "outputs" / "tensorboard"
MODEL_NAME = "dinov2-small"

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
GROUP = 10
HDR_THREADS = 16

IMG_SIZE = 224
LAT_MIN_OFFSET_MM = 20.0
SLICE_BAND = (0.0, 1.0)
SLICE_MODE = "center"  # band: SLICE_BAND 内均匀采样；center: 以中间层为中心连续取 GROUP 张

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

LR_HEAD = 3e-4
LR_BACKBONE = 8e-6
UNFREEZE_LAST = 6
WEIGHT_DECAY = 0.02
EVAL_BATCH = 8
EPOCHS = 10
BATCH_STUDIES = 8

# DataLoader 并行解码参数：DICOM 读取/像素解码在 CPU 上耗时较多
# 增大 num_workers 通常能显著提升 GPU 利用率。
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", "4"))
PREFETCH_FACTOR = int(os.environ.get("PREFETCH_FACTOR", "2"))
PIN_MEMORY = True
PERSISTENT_WORKERS = True

# batch 级别 train_auc 计算很慢（CPU + sklearn），所以需要降频
BATCH_AUC_EVERY = int(os.environ.get("BATCH_AUC_EVERY", "50"))

# 图像预处理缓存：True 时 dataset 直接读 .npz，跳过 DICOM 解码
# 用 scripts/preprocess.py 生成缓存后开启
USE_IMG_CACHE = 1#os.environ.get("USE_IMG_CACHE", "0") == "1"
CACHE_DIR = _REPO / "cache"
IMG_CACHE_DIR = CACHE_DIR / "img"

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
    "PatientID",
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
