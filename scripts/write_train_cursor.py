#!/usr/bin/env python3
"""Gold copy + merge checkpoint shards into train_cursor.csv."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm

DATA = Path("/root/autodl-tmp/kaggle/data")
TRAIN_CSV = DATA / "train.csv"
OUT_CSV = DATA / "train_cursor.csv"
SHARD_DIR = DATA / "cursor_shards"
GOLD_JSONL = SHARD_DIR / "gold.jsonl"

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
COLS = ["StudyInstanceUID"] + [x for t in TARGETS for x in (t, t + "__conf")]
SHARD_SIZE = 80


def load_train() -> pd.DataFrame:
    return pd.read_csv(TRAIN_CSV)


def gold_mask(df: pd.DataFrame) -> pd.Series:
    return df[TARGETS].notna().all(axis=1)


def gold_rec(row: pd.Series) -> dict:
    rec = {"StudyInstanceUID": row["StudyInstanceUID"]}
    for t in TARGETS:
        rec[t] = float(row[t])
        rec[t + "__conf"] = 1.0
    return rec


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in rows:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def cmd_init(shard_size: int) -> None:
    df = load_train()
    gmask = gold_mask(df)
    gold_df = df.loc[gmask]
    unlabeled = df.loc[~gmask]
    print(f"studies={len(df)}  gold={len(gold_df)}  unlabeled={len(unlabeled)}")

    SHARD_DIR.mkdir(parents=True, exist_ok=True)
    gold_rows = [gold_rec(row) for _, row in tqdm(gold_df.iterrows(), total=len(gold_df), desc="gold")]
    write_jsonl(GOLD_JSONL, gold_rows)
    print(f"wrote {GOLD_JSONL}  n={len(gold_rows)}")

    n_shard = 0
    for i in tqdm(range(0, len(unlabeled), shard_size), desc="shards"):
        chunk = unlabeled.iloc[i : i + shard_size]
        payload = [
            {"StudyInstanceUID": r["StudyInstanceUID"], "Report": r["Report"] or ""}
            for _, r in chunk.iterrows()
        ]
        path = SHARD_DIR / f"shard_{n_shard:03d}.json"
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        n_shard += 1
    print(f"wrote {n_shard} report shards under {SHARD_DIR}  size={shard_size}")


def iter_label_files() -> list[Path]:
    files = [GOLD_JSONL] if GOLD_JSONL.exists() else []
    files += sorted(SHARD_DIR.glob("shard_*.labels.jsonl"))
    return files


def cmd_merge() -> None:
    df = load_train()
    gmask = gold_mask(df)
    gold_ids = set(df.loc[gmask, "StudyInstanceUID"])
    by_id: dict[str, dict] = {}
    n_files = 0
    for path in tqdm(iter_label_files(), desc="load jsonl"):
        n_files += 1
        for rec in read_jsonl(path):
            uid = rec.get("StudyInstanceUID")
            if uid:
                by_id[uid] = rec
    print(f"files={n_files}  unique_rows={len(by_id)}")

    missing = [u for u in df["StudyInstanceUID"] if u not in by_id]
    extra = [u for u in by_id if u not in set(df["StudyInstanceUID"])]
    if missing:
        raise SystemExit(f"missing {len(missing)} studies, e.g. {missing[:3]}")
    if extra:
        raise SystemExit(f"extra {len(extra)} studies not in train.csv")

    rows = []
    n_gold_ok = 0
    for _, row in tqdm(df.iterrows(), total=len(df), desc="validate"):
        rec = by_id[row["StudyInstanceUID"]]
        for t in TARGETS:
            if t not in rec or t + "__conf" not in rec:
                raise SystemExit(f"{row['StudyInstanceUID']} missing {t}")
            if pd.isna(rec[t]) or pd.isna(rec[t + "__conf"]):
                raise SystemExit(f"{row['StudyInstanceUID']} NaN {t}")
        if row["StudyInstanceUID"] in gold_ids:
            for t in TARGETS:
                if float(rec[t]) != float(row[t]) or float(rec[t + "__conf"]) != 1.0:
                    raise SystemExit(f"gold mismatch {row['StudyInstanceUID']} {t}")
            n_gold_ok += 1
        rows.append({c: rec[c] for c in COLS})

    out = pd.DataFrame(rows, columns=COLS)
    out.to_csv(OUT_CSV, index=False)
    print(f"wrote {OUT_CSV}  rows={len(out)}  gold_ok={n_gold_ok}")


def cmd_check() -> None:
    bad = []
    shards = sorted(SHARD_DIR.glob("shard_*.json"))
    for src in tqdm(shards, desc="check shards"):
        lab = src.with_name(src.name.replace(".json", ".labels.jsonl"))
        if not lab.exists():
            continue
        reports = json.loads(src.read_text(encoding="utf-8"))
        rows = read_jsonl(lab)
        uids_src = [r["StudyInstanceUID"] for r in reports]
        uids_lab = [r["StudyInstanceUID"] for r in rows]
        if uids_src != uids_lab:
            bad.append(f"{lab.name} uid mismatch {len(uids_lab)} vs {len(uids_src)}")
            continue
        for rec in rows:
            for t in TARGETS:
                v, c = rec.get(t), rec.get(t + "__conf")
                if not isinstance(v, (int, float)) or not isinstance(c, (int, float)):
                    bad.append(f"{lab.name} {rec['StudyInstanceUID']} bad type {t}")
                    break
                if not (0.0 <= float(v) <= 1.0 and 0.0 <= float(c) <= 1.0):
                    bad.append(f"{lab.name} {rec['StudyInstanceUID']} out of range {t}")
                    break
    if bad:
        print("\n".join(bad[:20]))
        raise SystemExit(f"check failed: {len(bad)} issues")
    print("check ok")


def cmd_status() -> None:
    df = load_train()
    done = set()
    for path in iter_label_files():
        for rec in read_jsonl(path):
            done.add(rec["StudyInstanceUID"])
    unlabeled_ids = set(df.loc[~gold_mask(df), "StudyInstanceUID"])
    gold_ids = set(df.loc[gold_mask(df), "StudyInstanceUID"])
    print(f"gold_done={len(done & gold_ids)}/{len(gold_ids)}")
    print(f"unlabeled_done={len(done & unlabeled_ids)}/{len(unlabeled_ids)}")
    shards = sorted(SHARD_DIR.glob("shard_*.json"))
    labeled = {p.name.replace(".labels.jsonl", ".json") for p in SHARD_DIR.glob("shard_*.labels.jsonl")}
    pending = [p.name for p in shards if p.name not in labeled]
    print(f"shards_pending={len(pending)}/{len(shards)}")
    if pending[:8]:
        print("next", pending[:8])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["init", "merge", "status", "check"])
    p.add_argument("--shard-size", type=int, default=SHARD_SIZE)
    args = p.parse_args()
    if args.cmd == "init":
        cmd_init(args.shard_size)
    elif args.cmd == "merge":
        cmd_merge()
    elif args.cmd == "check":
        cmd_check()
    else:
        cmd_status()


if __name__ == "__main__":
    main()
