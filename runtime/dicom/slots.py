"""Choose one series per imaging slot for each study."""
from __future__ import annotations

import pandas as pd

from runtime.config import RULES, SLOTS

def pick_slots(series_df, plane_map):
    """One series per slot per study.

    Ties are broken toward the stack with the most slices: a thicker stack samples the
    joint more densely, and the three-slice sampler below benefits from the margin.
    """
    series_df = series_df.copy()
    series_df["plane"] = series_df["SeriesInstanceUID"].map(plane_map)
    out = {}
    for study, g in series_df.groupby("StudyInstanceUID"):
        chosen = {}
        for name, plane, fluid, fs in SLOTS:
            sel = (g["plane"] == plane) & (g["fatsat"] == fs)
            # fluid=None means "do not condition on weighting" - the public scheme,
            # where the single provided flag stands in for both axes at once.
            if fluid is not None:
                sel &= (g["fluid"] == fluid)
            cand = g[sel]
            # A slot with no series matching its predicate stays empty, and no substitute
            # is admitted from a neighbouring predicate. Relaxing the weighting to fill a
            # T1 slot would draw from the pool `SAG_FLUID_NOFS` selects from, since that
            # pool is what remains once the weighting is dropped: over the training corpus
            # it would put one series in two slots for 2383 of 4407 studies and leave 56%
            # of the T1 slot holding PD or T2. The presence mask would then assert a
            # sequence that was never acquired, and the per-diagnosis softmax of §6 would
            # divide its attention across two identical slots, giving one acquisition
            # about twice the weight it carries in a study that holds both. The mask is
            # there to say a slot is absent, which is what an absent slot is.
            if len(cand) == 0 and RULES["slot_fallback"] and fluid is False:
                # The relaxation the paragraph above rejects, reproduced because an
                # imported member was fitted with its T1 slots filled this way: over half
                # of that member's training studies had a T1 slot holding a series that
                # is not T1. Leaving those slots empty would present it with a presence
                # mask it never saw.
                cand = g[(g["plane"] == plane) & (~g["fatsat"])]
            if len(cand):
                chosen[name] = cand.sort_values("n_slices", ascending=False).iloc[0]
        out[study] = chosen
    return out