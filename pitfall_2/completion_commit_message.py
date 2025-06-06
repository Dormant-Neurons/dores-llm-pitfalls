#!/usr/bin/env python3
"""
PrimeVul *commit message* completion evaluation.
Evaluates:
  • Levenshtein distance
  • Levenshtein similarity ratio
  • BLEU (nltk smooth method 4)
  • Jaccard token similarity

Inputs:
  - Only samples with a non-"None" commit_message string.

Dependencies:
  pandas, numpy, sklearn, tqdm, openai, python-Levenshtein (optional), nltk (optional)
"""

from __future__ import annotations
import json, os, random, functools
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from openai import OpenAI

# Debug flag
PRINT_COMMIT_MESSAGES = True

try:
    import Levenshtein
    lev_distance = Levenshtein.distance
except ImportError:
    def lev_distance(a: str, b: str) -> int:
        @functools.lru_cache(None)
        def _ld(i, j):
            if i == 0: return j
            if j == 0: return i
            cost = 0 if a[i-1] == b[j-1] else 1
            return min(_ld(i-1,j)+1, _ld(i,j-1)+1, _ld(i-1,j-1)+cost)
        return _ld(len(a), len(b))

try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    _smooth = SmoothingFunction().method4
    def bleu_score(ref, hyp):
        ref_tok, hyp_tok = ref.split(), hyp.split()
        return sentence_bleu([ref_tok], hyp_tok, smoothing_function=_smooth) if hyp_tok else 0.0
except ImportError:
    def bleu_score(ref, hyp): return 0.0

def jaccard_similarity(a, b):
    s1, s2 = set(a.split()), set(b.split())
    return len(s1 & s2) / len(s1 | s2) if s1 | s2 else 0.0

# ───── OpenAI Setup ─────
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
if client.api_key is None:
    raise RuntimeError("OPENAI_API_KEY not set")
MODEL, TEMPERATURE = "gpt-3.5-turbo", 0.0

# ───── Paths and Config ─────
DATA_DIR = Path("datasets/primevul")
TRAIN_PATH, VALID_PATH, TEST_PATH = (DATA_DIR / f"primevul_{p}.jsonl" for p in ("train", "valid", "test"))
COMMIT_DATES_PATH = DATA_DIR / "commit_dates.jsonl"
CUTOFF = pd.Timestamp("2021-09-01", tz="UTC")
N_SAMPLES, RANDOM_SEED = 200, 42

# ───── I/O ─────
def _read_jsonl(path):
    with path.open(encoding="utf-8") as fh:
        return pd.DataFrame(json.loads(l) for l in fh)

def load_merged():
    df_all = pd.concat(map(_read_jsonl, (TRAIN_PATH, VALID_PATH, TEST_PATH)), ignore_index=True)
    df_all = df_all[df_all["commit_message"] != "None"]
    commit_df = _read_jsonl(COMMIT_DATES_PATH)
    commit_df["commit_date"] = pd.to_datetime(commit_df["commit_date"], utc=True)
    merged = df_all.merge(commit_df[["project", "commit_id", "commit_date"]], on="commit_id", how="left")
    merged = merged.dropna(subset=["commit_date"])
    return merged

def draw_samples(df):
    aft = df[df["commit_date"] >= CUTOFF].sample(N_SAMPLES, random_state=RANDOM_SEED)
    bef = df[df["commit_date"] < CUTOFF].sample(N_SAMPLES, random_state=RANDOM_SEED)
    return aft.reset_index(drop=True), bef.reset_index(drop=True)

# ───── Prompting ─────
SYSTEM_PROMPT = (
    "You are a commit message assistant. I will give you project+commit+partial message. "
    "Predict the full original commit message only. No markdown or explanation."
)

def build_prompt(project: str, commit: str, partial_msg: str) -> str:
    return f"Project: {project}\nCommit: {commit}\nPartial commit message: \"{partial_msg}\""

def normalise(txt: str) -> str:
    return "\n".join(l.rstrip() for l in txt.strip().splitlines())

def strip_formatting(txt: str) -> str:
    return txt.replace("\n", " ").replace("\t", " ").strip()

def complete(project: str, commit: str, partial_msg: str) -> str:
    msg = build_prompt(project, commit, partial_msg)
    chat = client.chat.completions.create(
        model=MODEL, temperature=TEMPERATURE,
        messages=[{"role": "system", "content": SYSTEM_PROMPT},
                  {"role": "user", "content": msg}]
    )
    return chat.choices[0].message.content

# ───── Evaluation ─────
def evaluate(df: pd.DataFrame, name: str) -> dict:
    exact, dists, sims, bleus, jaccs = [], [], [], [], []

    iterator = zip(df["project_url"], df["commit_id"], df["commit_message"])
    for proj, cid, true_msg in tqdm(iterator, total=len(df), desc=f"{name:>18}"):
        norm_truth = normalise(true_msg)
        hint = " ".join(norm_truth.split()[:len(norm_truth.split()) // 2])

        try:
            pred = normalise(complete(proj, cid, hint))
        except Exception as e:
            print(f"OpenAI error – treating as empty: {e}")
            pred = ""

        stripped_truth = strip_formatting(norm_truth)
        stripped_pred = strip_formatting(pred)

        if PRINT_COMMIT_MESSAGES:
            print("\n=== COMMIT MESSAGE ===")
            print(f"Hint     : {hint}\nActual   : {stripped_truth}\nPredicted: {stripped_pred}")

        match = stripped_pred == stripped_truth
        exact.append(int(match))

        dist = lev_distance(stripped_truth, stripped_pred)
        dists.append(dist)
        sims.append(1 - dist / max(len(stripped_truth), len(stripped_pred), 1))
        bleus.append(bleu_score(stripped_truth, stripped_pred))
        jaccs.append(jaccard_similarity(stripped_truth, stripped_pred))

    return {
        "dataset": name,
        "accuracy": float(np.mean(exact)),
        "avg_lev_dist": float(np.mean(dists)),
        "avg_lev_sim": float(np.mean(sims)),
        "avg_bleu": float(np.mean(bleus)),
        "avg_jaccard": float(np.mean(jaccs)),
        "correct": int(np.sum(exact)),
        "total": len(df),
    }

# ───── Main ─────
def main():
    random.seed(RANDOM_SEED); np.random.seed(RANDOM_SEED)
    merged = load_merged()
    after_df, before_df = draw_samples(merged)

    results = [evaluate(after_df, "after Oct-2021"),
               evaluate(before_df, "before Oct-2021")]

    print("\n===== Summary =====")
    for r in results:
        print(
            f"{r['dataset']:>15}: "
            f"Acc={r['accuracy']:.2%}  "
            f"LevDist={r['avg_lev_dist']:.1f}  "
            f"LevSim={r['avg_lev_sim']:.2%}  "
            f"BLEU={r['avg_bleu']:.3f}  "
            f"Jacc={r['avg_jaccard']:.2%}  "
            f"({r['correct']}/{r['total']} exact)"
        )

if __name__ == "__main__":
    main()