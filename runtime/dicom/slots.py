"""每个检查、每个成像槽只选一条序列。"""
from __future__ import annotations

import pandas as pd
from tqdm import tqdm

from runtime.config import RULES, SLOTS

def pick_slots(series_df, plane_map):
    """每个 study、每个槽只留一条 series。

    同一槽有多条候选时取切片数最多的：层更密，后面抽 3 张切片时更不容易贴边。
    """
    series_df = series_df.copy()
    series_df["plane"] = series_df["SeriesInstanceUID"].map(plane_map)
    out = {}
    for study, g in tqdm(series_df.groupby("StudyInstanceUID")):
        chosen = {}
        for name, plane, fluid, fs in SLOTS:
            sel = (g["plane"] == plane) & (g["fatsat"] == fs)
            # fluid=None：不按加权筛选（public 方案，一个标志同时代表液体/压脂）
            if fluid is not None:
                sel &= (g["fluid"] == fluid)
            cand = g[sel]
            # 没有匹配的序列就让槽空着，不拿邻近槽的序列来凑。
            # 若放宽加权去填 T1 槽，候选会和 SAG_FLUID_NOFS 同一批：训练集里
            # 2383/4407 个检查会出现同一条序列占两个槽，56% 的 T1 槽实际是 PD/T2。
            # 存在 mask 会声称拍过 T1，注意力再把同一采集算两次。
            # mask 的本意就是：没拍到就是缺席。
            if len(cand) == 0 and RULES["slot_fallback"] and fluid is False:
                # 上面拒绝的放宽：只为兼容旧权重。那个模型训练时过半 T1 槽
                # 装的不是 T1；现在若留空，mask 分布会对不上它见过的输入。
                cand = g[(g["plane"] == plane) & (~g["fatsat"])]
            if len(cand):
                chosen[name] = cand.sort_values("n_slices", ascending=False).iloc[0]
        out[study] = chosen
    return out