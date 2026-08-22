#!/usr/bin/env python3
"""
预处理脚本：把每个 study 的 load_study() 结果提前算好并存成 .npz。
运行一次后，设置 USE_IMG_CACHE=1 即可让 dataset 跳过 DICOM 解码。

用法：
    python scripts/preprocess.py [--workers N] [--split train|test|all] [--force]

生成文件：
    cache/img/<StudyUID>.npz
        imgs: uint8 [n_slot, GROUP, H, W]   # GROUP 来自 config.py
        mask: float32 [n_slot]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from tqdm import tqdm

from runtime.config import DATA_ROOT, GROUP, IMG_CACHE_DIR, IMG_SIZE, LOAD_IMG, N_SLOT
from runtime.data_prep import prepare_slot_maps
from runtime.dicom.io import load_study

_DECODE_SIZE = max(IMG_SIZE, LOAD_IMG)

_SLOT_MAP = None
_LAT_MAP = None
_N_SLICE = None
_OUT_SIZE = None
_CACHE_DIR = None
_COMPRESS = None
_FORCE = False


def _expected_shape() -> tuple[int, int, int, int]:
    return N_SLOT, GROUP, _DECODE_SIZE, _DECODE_SIZE


def _cache_ok(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        with np.load(path) as d:
            return tuple(d["imgs"].shape) == _expected_shape()
    except Exception:
        return False


def _init_worker(slot_map, lat_map, n_slice, out_size, cache_dir, compress, force):
    global _SLOT_MAP, _LAT_MAP, _N_SLICE, _OUT_SIZE, _CACHE_DIR, _COMPRESS, _FORCE
    _SLOT_MAP = slot_map
    _LAT_MAP = lat_map
    _N_SLICE = n_slice
    _OUT_SIZE = out_size
    _CACHE_DIR = cache_dir
    _COMPRESS = compress
    _FORCE = force


def _process_one(study: str):
    global _SLOT_MAP, _LAT_MAP, _N_SLICE, _OUT_SIZE, _CACHE_DIR, _COMPRESS, _FORCE
    out_path = Path(_CACHE_DIR) / f"{study}.npz"
    if not _FORCE and _cache_ok(out_path):
        return study, "skip"
    try:
        imgs, mask = load_study(study, _SLOT_MAP, _LAT_MAP, n_slice=_N_SLICE, out_size=_OUT_SIZE)
        imgs_np = imgs.numpy().astype(np.uint8)
        mask_np = mask.numpy().astype(np.float32)
        if tuple(imgs_np.shape) != _expected_shape():
            return study, f"err:shape {tuple(imgs_np.shape)} != {_expected_shape()}"
        if _COMPRESS:
            np.savez_compressed(out_path, imgs=imgs_np, mask=mask_np)
        else:
            np.savez(out_path, imgs=imgs_np, mask=mask_np)
        return study, "ok"
    except Exception as e:
        return study, f"err:{e}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--workers", type=int, default=int(os.environ.get("NUM_WORKERS", "8")))
    p.add_argument("--split", choices=["train", "test", "all"], default="all")
    p.add_argument("--force", action="store_true", help="忽略已有缓存，全部重算")
    p.add_argument(
        "--compress",
        type=int,
        default=int(os.environ.get("IMG_CACHE_COMPRESS", "0")),
        help="0=不压缩（更快），1=使用 np.savez_compressed（更省空间但更慢）",
    )
    args = p.parse_args()

    IMG_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    exp = _expected_shape()
    print(f"[preprocess] cache dir: {IMG_CACHE_DIR}", flush=True)
    print(f"[preprocess] GROUP={GROUP} decode_size={_DECODE_SIZE} expect imgs{exp}", flush=True)

    data = prepare_slot_maps(DATA_ROOT.resolve())
    compress = args.compress == 1

    def run_bucket(studies: list[str], slot_map: dict, lat_map: dict) -> tuple[int, int, int]:
        to_do = []
        stale = 0
        for s in studies:
            pth = IMG_CACHE_DIR / f"{s}.npz"
            if args.force:
                to_do.append(s)
            elif _cache_ok(pth):
                continue
            else:
                if pth.exists():
                    stale += 1
                to_do.append(s)
        already = len(studies) - len(to_do)
        if stale:
            print(f"[preprocess] stale cache (GROUP/size changed): {stale}", flush=True)
        if not to_do:
            print(f"[preprocess] bucket already cached: {already}/{len(studies)}", flush=True)
            return 0, already, 0

        ok_b = err_b = 0
        skip_b = already
        with ProcessPoolExecutor(
            max_workers=args.workers,
            initializer=_init_worker,
            initargs=(slot_map, lat_map, GROUP, _DECODE_SIZE, str(IMG_CACHE_DIR), compress, args.force),
        ) as pool:
            futs = {pool.submit(_process_one, s): s for s in to_do}
            with tqdm(total=len(to_do), unit="study") as bar:
                for fut in as_completed(futs):
                    study, status = fut.result()
                    if status == "ok":
                        ok_b += 1
                    elif status == "skip":
                        skip_b += 1
                    else:
                        err_b += 1
                        tqdm.write(f"  FAIL {study}: {status}")
                    bar.set_postfix(ok=ok_b, err=err_b)
                    bar.update(1)
        return ok_b, skip_b, err_b

    if args.split == "train":
        studies_all = data["st_tr"]
    elif args.split == "test":
        studies_all = data["st_te"]
    else:
        studies_all = data["st_tr"] + data["st_te"]

    already = sum(1 for s in studies_all if _cache_ok(IMG_CACHE_DIR / f"{s}.npz"))
    print(
        f"[preprocess] {len(studies_all)} studies total, {already} valid cached, "
        f"{len(studies_all) - already} to process",
        flush=True,
    )

    ok = skip = err = 0
    if args.split in ("train", "all"):
        o, sk, e = run_bucket(data["st_tr"], data["slots_tr"], data["lat_tr"])
        ok += o; skip += sk; err += e
    if args.split in ("test", "all"):
        o, sk, e = run_bucket(data["st_te"], data["slots_te"], data["lat_te"])
        ok += o; skip += sk; err += e

    print(f"[preprocess] done: ok={ok} skip={skip} err={err}", flush=True)
    cached = list(IMG_CACHE_DIR.glob("*.npz"))
    if cached:
        print(f"[preprocess] cache size: {sum(f.stat().st_size for f in cached) / 1e9:.1f} GB", flush=True)


if __name__ == "__main__":
    main()
