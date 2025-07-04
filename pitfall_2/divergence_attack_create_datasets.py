#!/usr/bin/env python3
"""
Create:
  • 1 000-row pre-cut-off chat fine-tune file *without* the function body
  • two 200-row evaluation splits (before/after) in the original PrimeVul row format

Integrity safeguards
────────────────────
* **No leakage** – any commit present in the training set is excluded from both
  evaluation splits.
* **Uniqueness** – every split is unique on ``commit_id``.
* **Deduplication** – the raw PrimeVul corpus contains duplicate rows (the same
  ``commit_id`` appears in train/valid/test).  We collapse them *once* during
  ingestion, keeping the first occurrence (deterministic order).
* **Fail-fast** – if any guard fails a clear ``ValueError`` is raised.

Folder layout expected:
    datasets/primevul/
        ├─ primevul_train.jsonl
        ├─ primevul_valid.jsonl
        ├─ primevul_test.jsonl
        └─ commit_dates.jsonl
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd

# ─── Config ────────────────────────────────────────────────────────────────────
DATA_DIR = Path("datasets/primevul")
OUT_TRAIN_CHAT = DATA_DIR / "primevul_before1000_chat_nofunc.jsonl"
OUT_EVAL_BEFORE = DATA_DIR / "primevul_eval_before200.jsonl"
OUT_EVAL_AFTER = DATA_DIR / "primevul_eval_after200.jsonl"

CUTOFF = pd.Timestamp("2021-09-01", tz="UTC")
N_TRAIN = 1_000
N_EVAL_SPLIT = 200
SEED = 42

SYSTEM_PROMPT = (
    "You are a commit message assistant. I will give you project+commit "
    "and the first half of the commit message. Predict the full original "
    "commit message only. No markdown or explanation."
)

# ─── Helper utilities ──────────────────────────────────────────────────────────

def _read_jsonl(path: Path) -> pd.DataFrame:
    """Read a JSONL file into a :class:`pandas.DataFrame`."""
    with path.open(encoding="utf-8") as fh:
        return pd.DataFrame(json.loads(l) for l in fh)


def load_merged() -> pd.DataFrame:
    """Return a *deduplicated* dataframe of all PrimeVul rows with dates.

    The original corpus ships with duplicated ``commit_id`` rows across
    train/valid/test splits.  For our purposes we need each commit only once,
    so we:
    1. Concatenate the splits.
    2. Attach commit dates.
    3. Drop rows with a missing commit message ("None").
    4. Sort deterministically by ``commit_date`` (latest first) and remove
       duplicates on ``commit_id`` keeping the first occurrence.
    5. Verify uniqueness.
    """

    train, valid, test = (
        DATA_DIR / f"primevul_{p}.jsonl" for p in ("train", "valid", "test")
    )
    df_all = pd.concat(map(_read_jsonl, (train, valid, test)), ignore_index=True)

    df_all = df_all[df_all["commit_message"] != "None"].copy()

    dates = _read_jsonl(DATA_DIR / "commit_dates.jsonl")
    dates["commit_date"] = pd.to_datetime(dates["commit_date"], utc=True)

    merged = df_all.merge(
        dates[["project", "commit_id", "commit_date"]],
        on="commit_id",
        how="left",
    ).dropna(subset=["commit_date"])

    # Deduplicate commit_id – keep the most recent commit_date (deterministic)
    merged = (
        merged.sort_values("commit_date", ascending=False)
        .drop_duplicates(subset=["commit_id"], keep="first")
        .reset_index(drop=True)
    )

    _assert_unique(merged, ["commit_id"], name="merged_deduped")
    return merged


def build_prompt(project: str, commit: str, partial_msg: str) -> str:
    """⚠️ NO function body included anymore."""
    return (
        f"Project: {project}\n"
        f"Commit: {commit}\n"
        f'Partial commit message: "{partial_msg}"'
    )


def normalise(txt: str) -> str:
    """Strip trailing whitespace on a per-line basis and surrounding newlines."""
    return "\n".join(l.rstrip() for l in txt.strip().splitlines())


# ─── Integrity checks ─────────────────────────────────────────────────────────

def _assert_unique(df: pd.DataFrame, subset: List[str], *, name: str) -> None:
    """Raise if *df* contains duplicates in *subset*."""
    dups = df[df.duplicated(subset=subset, keep=False)]
    if not dups.empty:
        raise ValueError(
            f"{name}: {len(dups)} duplicate rows on columns {subset}."
        )


def _assert_no_overlap(
    left: pd.DataFrame,
    right: pd.DataFrame,
    subset: Iterable[str],
    *,
    left_name: str,
    right_name: str,
) -> None:
    """Raise if any row overlaps on *subset* between *left* and *right*."""
    overlap = pd.merge(left[list(subset)], right[list(subset)], on=list(subset), how="inner")
    if not overlap.empty:
        raise ValueError(
            f"Data leakage detected: {len(overlap)} overlapping rows between "
            f"{left_name} and {right_name} on columns {list(subset)}."
        )


# ─── Output helpers ────────────────────────────────────────────────────────────

def write_jsonl(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


# ─── Main pipeline ────────────────────────────────────────────────────────────

def main() -> None:  # noqa: C901 – single entry point is fine
    random.seed(SEED)
    np.random.seed(SEED)

    df = load_merged()

    before_df = df[df["commit_date"] < CUTOFF].copy()
    after_df = df[df["commit_date"] >= CUTOFF].copy()

    # ── 1) TRAIN ― 1 000 rows (before cutoff, unique) ────────────────────────
    train_df = before_df.sample(N_TRAIN, random_state=SEED).copy()
    _assert_unique(train_df, ["commit_id"], name="train_df")

    # Build chat rows
    train_chat_rows = []
    for r in train_df.itertuples():
        full_msg = normalise(r.commit_message)
        half_hint = " ".join(full_msg.split()[: len(full_msg.split()) // 2])

        train_chat_rows.append(
            {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": build_prompt(r.project_url, r.commit_id, half_hint),
                    },
                    {"role": "assistant", "content": full_msg},
                ]
            }
        )

    write_jsonl(OUT_TRAIN_CHAT, train_chat_rows)
    print(f"✅  {OUT_TRAIN_CHAT}  ({len(train_chat_rows)} rows)")

    # ── Prune training examples from *before* pool to prevent leakage ────────
    before_df_no_train = before_df[~before_df["commit_id"].isin(train_df["commit_id"])]

    # ── 2) EVAL-BEFORE ― 200 rows ────────────────────────────────────────────
    eval_before = before_df_no_train.sample(N_EVAL_SPLIT, random_state=SEED + 1).copy()
    _assert_unique(eval_before, ["commit_id"], name="eval_before")
    _assert_no_overlap(
        train_df,
        eval_before,
        subset=["commit_id"],
        left_name="train_df",
        right_name="eval_before",
    )

    eval_before_rows = eval_before[[
        "project_url",
        "commit_id",
        "func",
        "commit_message",
    ]].to_dict(orient="records")
    write_jsonl(OUT_EVAL_BEFORE, eval_before_rows)
    print(f"✅  {OUT_EVAL_BEFORE}  ({len(eval_before_rows)} rows)")

    # ── 3) EVAL-AFTER ― 200 rows ─────────────────────────────────────────────
    eval_after = after_df.sample(N_EVAL_SPLIT, random_state=SEED).copy()
    _assert_unique(eval_after, ["commit_id"], name="eval_after")
    _assert_no_overlap(
        train_df,
        eval_after,
        subset=["commit_id"],
        left_name="train_df",
        right_name="eval_after",
    )

    eval_after_rows = eval_after[[
        "project_url",
        "commit_id",
        "func",
        "commit_message",
    ]].to_dict(orient="records")
    write_jsonl(OUT_EVAL_AFTER, eval_after_rows)
    print(f"✅  {OUT_EVAL_AFTER}  ({len(eval_after_rows)} rows)")

    print("🎉  All integrity checks passed – splits created without leakage.")


if __name__ == "__main__":
    main()
