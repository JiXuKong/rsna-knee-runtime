#!/usr/bin/env python3
"""
检查图像缓存是否和 load_study() 输出一致。

用法：
  python scripts/check_cache.py --split train --n 5
  python scripts/check_cache.py --split test --n 3
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse

import numpy as np
import torch

from runtime.config import GROUP, IMG_CACHE_DIR, IMG_SIZE, LOAD_IMG, USE_IMG_CACHE
from runtime.data_prep import prepare_slot_maps
from runtime.dicom.io import load_study
from runtime.config import DATA_ROOT


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", choices=["train", "test", "all"], default="train")
    p.add_argument("--n", type=int, default=5)
    p.add_argument("--seed", type=int, default=2026)
    args = p.parse_args()

    data = prepare_slot_maps(DATA_ROOT.resolve())

    if args.split == "train":
        studies = data["st_tr"]
        slot_map, lat_map = data["slots_tr"], data["lat_tr"]
    elif args.split == "test":
        studies = data["st_te"]
        slot_map, lat_map = data["slots_te"], data["lat_te"]
    else:
        studies = data["st_tr"] + data["st_te"]
        # split=all 时 slot_map/lat_map 分别来自 train/test；这里选择 data["slots_tr"]/["lat_tr"] 只用于抽样一致性检查
        slot_map, lat_map = data["slots_tr"], data["lat_tr"]

    rng = np.random.default_rng(args.seed)
    pick = studies[:]
    if len(pick) > args.n:
        pick = list(rng.choice(pick, size=args.n, replace=False))

    print(f"[check_cache] USE_IMG_CACHE={USE_IMG_CACHE} IMG_CACHE_DIR={IMG_CACHE_DIR}")
    print(f"[check_cache] checking {len(pick)} studies from split={args.split}")

    ok = 0
    for st in pick:
        cache_path = IMG_CACHE_DIR / f"{st}.npz"
        if not cache_path.exists():
            print(f"  MISSING cache {cache_path}")
            continue
        d = np.load(cache_path)
        imgs_c = torch.from_numpy(d["imgs"])
        mask_c = torch.from_numpy(d["mask"])

        # 训练时 decode_size = max(IMG_SIZE, LOAD_IMG)
        decode_size = max(IMG_SIZE, LOAD_IMG)
        imgs_d, mask_d = load_study(
            st,
            slot_map,
            lat_map,
            n_slice=GROUP,
            out_size=decode_size,
        )

        # imgs 为 uint8，直接比较逐元素是否一致；mask 为 float32，允许极小浮点误差
        same_imgs = torch.equal(imgs_c, imgs_d)
        same_mask = torch.allclose(mask_c, mask_d, atol=1e-6, rtol=0)

        if same_imgs and same_mask:
            ok += 1
            print(f"  OK {st}")
        else:
            imgs_diff = (imgs_c != imgs_d).sum().item()
            mask_max = (mask_c - mask_d).abs().max().item()
            print(f"  FAIL {st}: imgs_diff_elems={imgs_diff} mask_max_abs={mask_max}")

    print(f"[check_cache] done ok={ok}/{len(pick)}")


if __name__ == "__main__":
    main()

