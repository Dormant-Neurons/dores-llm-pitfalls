#!/usr/bin/env python3
"""
Create:
  • 1 000-row pre-cut-off chat fine-tune file *without* the function body
  • two 200-row evaluation splits (before/after) in the original PrimeVul row format
Folder layout expected:
    datasets/primevul/
        ├─ primevul_train.jsonl
        ├─ primevul_valid.jsonl
        ├─ primevul_test.jsonl
        └─ commit_dates.jsonl
"""

from __future__ import annotations
import json, random
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

# ─── Config ────────────────────────────────────────────────────────────────────
DATA_DIR        = Path("datasets/primevul")
OUT_TRAIN_CHAT  = DATA_DIR / "primevul_before1000_chat_nofunc.jsonl"
OUT_EVAL_BEFORE = DATA_DIR / "primevul_eval_before200.jsonl"
OUT_EVAL_AFTER  = DATA_DIR / "primevul_eval_after200.jsonl"

CUTOFF        = pd.Timestamp("2021-09-01", tz="UTC")
N_TRAIN       = 1_000
N_EVAL_SPLIT  = 200
SEED          = 42

SYSTEM_PROMPT = (
    "You are a commit message assistant. I will give you project+commit "
    "and the first half of the commit message. Predict the full original "
    "commit message only. No markdown or explanation."
)

# ─── Helpers copied from your evaluation script (tiny tweaks) ──────────────────
def _read_jsonl(path: Path) -> pd.DataFrame:
    with path.open(encoding="utf-8") as fh:
        return pd.DataFrame(json.loads(l) for l in fh)

def load_merged() -> pd.DataFrame:
    train, valid, test = (DATA_DIR / f"primevul_{p}.jsonl" for p in ("train", "valid", "test"))
    df_all = pd.concat(map(_read_jsonl, (train, valid, test)), ignore_index=True)
    df_all = df_all[df_all["commit_message"] != "None"]

    dates = _read_jsonl(DATA_DIR / "commit_dates.jsonl")
    dates["commit_date"] = pd.to_datetime(dates["commit_date"], utc=True)

    return df_all.merge(
        dates[["project", "commit_id", "commit_date"]],
        on="commit_id", how="left"
    ).dropna(subset=["commit_date"])

def build_prompt(project: str, commit: str, partial_msg: str) -> str:
    """⚠️ NO function body included anymore."""
    return (
        f"Project: {project}\n"
        f"Commit: {commit}\n"
        f'Partial commit message: "{partial_msg}"'
    )

def normalise(txt: str) -> str:
    return "\n".join(l.rstrip() for l in txt.strip().splitlines())

# ─── Output helpers ────────────────────────────────────────────────────────────
def write_jsonl(path: Path, rows: list[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

# ─── Main ──────────────────────────────────────────────────────────────────────
def main():
    random.seed(SEED); np.random.seed(SEED)

    df = load_merged()
    before_df = df[df["commit_date"] < CUTOFF]
    after_df  = df[df["commit_date"] >= CUTOFF]

    # ── 1) TRAIN 1 000 rows (before cutoff, unique) ───────────────────────────
    train_df = before_df.sample(N_TRAIN, random_state=SEED)

    train_chat_rows = []
    for r in train_df.itertuples():
        full_msg  = normalise(r.commit_message)
        half_hint = " ".join(full_msg.split()[: len(full_msg.split()) // 2])

        train_chat_rows.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",    "content": build_prompt(
                    r.project_url, r.commit_id, half_hint)},
                {"role": "assistant", "content": full_msg}
            ]
        })

    write_jsonl(OUT_TRAIN_CHAT, train_chat_rows)
    print(f"✅  {OUT_TRAIN_CHAT}  ({len(train_chat_rows)} rows)")

    # Remove training rows so they don't leak into eval-before
    before_df_no_train = before_df.drop(train_df.index)

    # ── 2) EVAL-BEFORE (200) ──────────────────────────────────────────────────
    eval_before = before_df_no_train.sample(N_EVAL_SPLIT, random_state=SEED + 1)
    # Keep only the columns the original script expects
    eval_before_rows = eval_before[["project_url", "commit_id", "func", "commit_message"]].to_dict(orient="records")
    write_jsonl(OUT_EVAL_BEFORE, eval_before_rows)
    print(f"✅  {OUT_EVAL_BEFORE}  ({len(eval_before_rows)} rows)")

    # ── 3) EVAL-AFTER (200) ───────────────────────────────────────────────────
    eval_after = after_df.sample(N_EVAL_SPLIT, random_state=SEED)
    eval_after_rows = eval_after[["project_url", "commit_id", "func", "commit_message"]].to_dict(orient="records")
    write_jsonl(OUT_EVAL_AFTER, eval_after_rows)
    print(f"✅  {OUT_EVAL_AFTER}  ({len(eval_after_rows)} rows)")

if __name__ == "__main__":
    main()