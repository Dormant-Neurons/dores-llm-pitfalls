#!/usr/bin/env python3
"""
PrimeVul GPT-3.5-turbo evaluation script (compatible with **openai-python ≥ 1.0**).

This script
1. Merges the PrimeVul train/valid/test splits and attaches commit timestamps.
2. Draws two random 200-function samples: **before** vs **on/after** 1 Oct 2021.
3. Queries *gpt-3.5-turbo* in a binary-classification setting ("vulnerable"? → 1/0).
4. Prints Accuracy, Precision, Recall and F1 per subset **and** the class balance
   (absolute numbers and percentages) for each subset.

---
Requirements
- Python ≥ 3.9
- pandas, scikit-learn, tqdm, numpy
- **openai-python ≥ 1.0**  (`pip install --upgrade openai`)

Set the environment variable **OPENAI_API_KEY** before running.

Each full run issues 400 chat completions (2×200) → mind rate-limits/costs.
"""
from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm import tqdm

# ─────────────────────────────── OpenAI setup ────────────────────────────────
from openai import OpenAI  # works for openai-python ≥ 1.0

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
if client.api_key is None:
    raise RuntimeError("Please set the OPENAI_API_KEY environment variable.")

MODEL = "gpt-3.5-turbo"
TEMPERATURE = 0.0  # deterministic responses

# ──────────────────────────────── Paths / config ─────────────────────────────
DATA_DIR = Path("datasets/primevul")
TRAIN_PATH = DATA_DIR / "primevul_train.jsonl"
VALID_PATH = DATA_DIR / "primevul_valid.jsonl"
TEST_PATH = DATA_DIR / "primevul_test.jsonl"
COMMIT_DATES_PATH = DATA_DIR / "commit_dates.jsonl"

CUTOFF = pd.Timestamp("2021-10-01", tz="UTC")  # boundary date
N_SAMPLES = 500
RANDOM_SEED = 42

# ───────────────────────────────── Helpers ───────────────────────────────────

def _read_jsonl(path: Path) -> pd.DataFrame:
    """Load a *.jsonl* file into a DataFrame."""
    with path.open("r", encoding="utf-8") as fh:
        records = [json.loads(l) for l in fh]
    return pd.DataFrame(records)


def load_merged() -> pd.DataFrame:
    """Merge the three dataset splits with commit-date metadata."""
    df_all = pd.concat(map(_read_jsonl, (TRAIN_PATH, VALID_PATH, TEST_PATH)), ignore_index=True)

    commit_df = _read_jsonl(COMMIT_DATES_PATH)
    commit_df["commit_date"] = pd.to_datetime(commit_df["commit_date"], utc=True)

    merged = df_all.merge(
        commit_df[["project", "commit_id", "commit_date"]],
        on=["commit_id"],
        how="left",
        validate="many_to_one",
    )
    if merged["commit_date"].isna().any():
        missing = merged["commit_date"].isna().sum()
        print(f"⚠️  {missing} rows missing commit_date – ignored during sampling.")
        merged = merged.dropna(subset=["commit_date"])  # simplifies sampling logic
    return merged


def draw_samples(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (after_df, before_df) of N_SAMPLES each."""
    after_df = df[df["commit_date"] >= CUTOFF].sample(n=N_SAMPLES, random_state=RANDOM_SEED)
    before_df = df[df["commit_date"] < CUTOFF].sample(n=N_SAMPLES, random_state=RANDOM_SEED)
    return after_df.reset_index(drop=True), before_df.reset_index(drop=True)

SYSTEM_PROMPT = (
    "You are a security expert. Decide whether the given C/C++ function contains "
    "a vulnerability. Respond with **exactly one character**: '1' if the function "
    "is vulnerable or '0' if it is not. Provide no explanation."
)


def classify(code: str) -> int:
    """Send *code* to GPT-3.5 and return integer prediction 0/1."""
    chat = client.chat.completions.create(
        model=MODEL,
        temperature=TEMPERATURE,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"```\n{code}\n```"},
        ],
    )
    answer = chat.choices[0].message.content.strip()
    return 1 if answer.startswith("1") else 0


def evaluate(df: pd.DataFrame, name: str) -> dict:
    """Run model on *df* and compute metrics."""
    y_true = df["target"].astype(int).tolist()
    y_pred: List[int] = []

    for code in tqdm(df["func"], desc=f"{name:>18}"):
        try:
            y_pred.append(classify(code))
        except Exception as exc:
            # on hard failure, default to safe negative prediction
            print(f"OpenAI error – default 0: {exc}")
            y_pred.append(0)

    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    return dict(dataset=name, accuracy=acc, precision=prec, recall=rec, f1=f1)


def class_balance(df: pd.DataFrame) -> dict:
    """Return absolute and percentage class balance for a DataFrame."""
    total = len(df)
    vuln = int(df["target"].sum())
    safe = total - vuln
    return {
        "total": total,
        "vulnerable": vuln,
        "not_vulnerable": safe,
        "vuln_pct": vuln / total * 100,
        "not_pct": safe / total * 100,
    }

# ─────────────────────────────────── main ─────────────────────────────────────

def main() -> None:
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    merged = load_merged()
    after_df, before_df = draw_samples(merged)

    metrics = [
        evaluate(after_df, "after Oct-2021"),
        evaluate(before_df, "before Oct-2021"),
    ]

    print("\n===== Summary =====")
    for m in metrics:
        print(
            f"{m['dataset']:>15}: "
            f"Acc={m['accuracy']:.3f}  "
            f"Prec={m['precision']:.3f}  "
            f"Rec={m['recall']:.3f}  "
            f"F1={m['f1']:.3f}"
        )

    # ──────────────── Class balance summary ────────────────
    print("\n--- Class Balance ---")
    for name, df_ in [("after Oct-2021", after_df), ("before Oct-2021", before_df)]:
        bal = class_balance(df_)
        print(
            f"{name:>15}: "
            f"{bal['vulnerable']:>3}/{bal['total']} vulnerable "
            f"({bal['vuln_pct']:.1f}%), "
            f"{bal['not_vulnerable']:>3}/{bal['total']} not vulnerable "
            f"({bal['not_pct']:.1f}%)"
        )


if __name__ == "__main__":
    main()
