#!/usr/bin/env python3
"""
离线诊断：缓存的 imgs/mask 是否与现场 load_study 完全一致？

定位策略：
1) 先比较 load_study(study_id) 的 imgs/mask 与 cache/img/<StudyUID>.npz 的 imgs/mask
2) 若不一致，进一步对每个 slot k：
   - 取 slot_map/lat_map 选中的 rec
   - 运行 read_slot(rec) 得到“未做 laterality 的 slot 图”
   - 再跑 normalise_laterality(...) 得到“与 load_study 一致的 slot 图”
   - 分别与缓存的 imgs[k] 对比，判断差异发生在 read_slot 还是 laterality/stacking

用法示例：
  python scripts/diagnose_cache.py --split train --n 5
  python scripts/diagnose_cache.py --study <StudyInstanceUID>
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import numpy as np
import torch

from runtime.config import (
    DATA_ROOT,
    GROUP,
    IMG_CACHE_DIR,
    IMG_SIZE,
    LOAD_IMG,
    SLOTS,
    USE_IMG_CACHE,
)
from runtime.data_prep import prepare_slot_maps
from runtime.dicom.io import load_study, normalise_laterality, read_slot


def _tensor_stats(a: torch.Tensor, b: torch.Tensor) -> tuple[int, float]:
    """
    返回：
      - 不同元素个数（逐元素不等）
      - mask/数值差异的最大绝对值（浮点用）
    """
    if a.dtype == torch.uint8:
        diff = (a != b).sum().item()
        return int(diff), 0.0
    else:
        diff = (a != b).sum().item()
        max_abs = (a - b).abs().max().item()
        return int(diff), float(max_abs)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", choices=["train", "test", "all"], default="train")
    p.add_argument("--n", type=int, default=3, help="随机抽样研究数（仅当没指定 --study）")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--study", type=str, default=None, help="指定 StudyInstanceUID 诊断（会忽略 --n）")
    args = p.parse_args()

    decode_size = max(IMG_SIZE, LOAD_IMG)
    print(f"[diag] USE_IMG_CACHE={USE_IMG_CACHE} cache_dir={IMG_CACHE_DIR}")
    print(f"[diag] GROUP={GROUP} LOAD_IMG={LOAD_IMG} IMG_SIZE={IMG_SIZE} => decode_size={decode_size}")
    print(f"[diag] slots={len(SLOTS)}")

    data = prepare_slot_maps(DATA_ROOT.resolve())

    if args.split == "train":
        studies = data["st_tr"]
        slot_map = data["slots_tr"]
        lat_map = data["lat_tr"]
    elif args.split == "test":
        studies = data["st_te"]
        slot_map = data["slots_te"]
        lat_map = data["lat_te"]
    else:
        studies = data["st_tr"] + data["st_te"]
        # all 模式下 slot_map/lat_map 的选择对“现场 load_study”和“缓存一致性”并不影响太大，
        # 因为 load_study 内最终选的是 chosen=slot_map.get(study_id)。
        # 这里我们优先用 train 的字典作为默认抽样；若 study 来自 test 且 train 字典取不到 rec，会自动成为全零对比。
        slot_map = data["slots_tr"]
        lat_map = data["lat_tr"]

    if args.study:
        picks = [args.study]
    else:
        rng = np.random.default_rng(args.seed)
        if len(studies) <= args.n:
            picks = list(studies)
        else:
            picks = list(rng.choice(studies, size=args.n, replace=False))

    print(f"[diag] checking {len(picks)} studies ...")

    ok = 0
    for st in picks:
        cache_path = IMG_CACHE_DIR / f"{st}.npz"
        if not cache_path.exists():
            print(f"\n[diag] MISSING cache: {cache_path}")
            continue

        # 1) load_study 现场 vs 缓存
        d = np.load(cache_path)
        imgs_c = torch.from_numpy(d["imgs"])
        mask_c = torch.from_numpy(d["mask"])

        imgs_d, mask_d = load_study(
            st,
            slot_map,
            lat_map,
            n_slice=GROUP,
            out_size=decode_size,
        )

        imgs_diff, _ = _tensor_stats(imgs_c, imgs_d)
        mask_diff, mask_max_abs = _tensor_stats(mask_c, mask_d)
        same = (imgs_diff == 0) and (mask_max_abs < 1e-6)

        print(f"\n[diag] Study {st}")
        print(f"  cache imgs shape={tuple(imgs_c.shape)} dtype={imgs_c.dtype}")
        print(f"  imgs_diff_elems={imgs_diff}")
        print(f"  mask_max_abs={mask_max_abs:.6g} mask_diff_elems={mask_diff}")

        if same:
            ok += 1
            print("  => OK (cache ==现场 load_study)")
            continue

        # 2) slot 级别定位：到底是 read_slot 还是 laterality
        chosen = slot_map.get(st)
        lat = lat_map.get(st)

        if not chosen:
            print("  chosen slot_map empty: 现场与缓存都可能是全零，但你这里已检测到不一致。请检查 cache 文件是否被写入了其他 study。")
            continue

        print("  => locating mismatched slots ...")
        mismatch_slots: list[int] = []
        for k, (_name, plane, _, _) in enumerate(SLOTS):
            if not torch.equal(imgs_c[k], imgs_d[k]):
                mismatch_slots.append(k)

        print(f"  mismatch slot indices: {mismatch_slots}")

        # 对前几个 mismatch slot 做细分解释（避免输出爆炸）
        for k in mismatch_slots[:5]:
            name, plane, _, _ = SLOTS[k]
            rec = chosen.get(name)
            if rec is None:
                print(f"    slot{k} name={name}: rec missing in slot_map (cache mismatch may come from caching with different SLOT_SCHEME)")
                continue

            img_raw = read_slot(rec, n_slice=GROUP, out_size=decode_size)  # uint8 [n_slice,H,W]?
            if img_raw is None:
                print(f"    slot{k} name={name}: read_slot returned None (decode failed) -> check cache generation")
                continue

            img_after_lat = normalise_laterality(img_raw, plane, lat)

            diff_raw = (imgs_c[k] != img_raw).sum().item()
            diff_lat = (imgs_c[k] != img_after_lat).sum().item()

            print(f"    slot{k} name={name} plane={plane} lat={lat}")
            print(f"      diff(cache, read_slot) elems={diff_raw}")
            print(f"      diff(cache, after_laterality) elems={diff_lat}")

    print(f"\n[diag] done ok={ok}/{len(picks)}")


if __name__ == "__main__":
    main()

