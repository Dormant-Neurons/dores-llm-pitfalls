#!/usr/bin/env python3
"""
PrimeVul *code-completion* evaluation — extended metrics version.

Adds:
  • Levenshtein distance               (requires python-Levenshtein, else pure-py fallback)
  • Levenshtein similarity ratio
  • BLEU (corpus average, NLTK smooth-method 4)  →  pip install nltk
  • Jaccard token similarity

Outputs per subset: Accuracy, avg Levenshtein dist/ratio, avg BLEU, avg Jaccard, plus
(# exact matches / total).

Dependencies
------------
python >= 3.9; pandas, numpy, sklearn, tqdm, openai-python >= 1.0
Optional but recommended: python-Levenshtein, nltk  (pip install python-Levenshtein nltk)
"""

from __future__ import annotations
import json, os, random, re, functools
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from openai import OpenAI

# ────────────── OpenAI setup ──────────────
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
if client.api_key is None:
    raise RuntimeError("OPENAI_API_KEY not set")
MODEL, TEMPERATURE = "gpt-3.5-turbo", 0.0

# ────────────── Paths / config ─────────────
DATA_DIR = Path("datasets/primevul")
TRAIN_PATH, VALID_PATH, TEST_PATH = (DATA_DIR / f"primevul_{p}.jsonl"
                                     for p in ("train", "valid", "test"))
COMMIT_DATES_PATH = DATA_DIR / "commit_dates.jsonl"
CUTOFF = pd.Timestamp("2021-09-01", tz="UTC")
N_SAMPLES, RANDOM_SEED = 200, 42

# ────────────── Similarity helpers ─────────
try:
    import Levenshtein   # type: ignore
    lev_distance = Levenshtein.distance
except ImportError:      # pure-Python fallback (O(n·m))
    def lev_distance(a: str, b: str) -> int:
        @functools.lru_cache(maxsize=None)
        def _ld(i: int, j: int) -> int:
            if i == 0: return j
            if j == 0: return i
            cost = 0 if a[i-1] == b[j-1] else 1
            return min(
                _ld(i-1, j) + 1,          # deletion
                _ld(i, j-1) + 1,          # insertion
                _ld(i-1, j-1) + cost      # substitution
            )
        return _ld(len(a), len(b))

try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction  # type: ignore
    _smooth = SmoothingFunction().method4
    def bleu_score(ref: str, hyp: str) -> float:
        ref_tok, hyp_tok = ref.split(), hyp.split()
        if not hyp_tok:                    # avoid math/log(0) errors
            return 0.0
        return sentence_bleu([ref_tok], hyp_tok, smoothing_function=_smooth)
except ImportError:                        # graceful degradation
    def bleu_score(ref: str, hyp: str) -> float:  # type: ignore
        return 0.0

def jaccard_similarity(a: str, b: str) -> float:
    s1, s2 = set(a.split()), set(b.split())
    return len(s1 & s2) / len(s1 | s2) if s1 | s2 else 0.0

# ────────────── I/O helpers ──────────────
def _read_jsonl(path: Path) -> pd.DataFrame:
    with path.open(encoding="utf-8") as fh:
        return pd.DataFrame(json.loads(l) for l in fh)

def load_merged() -> pd.DataFrame:
    df_all = pd.concat(map(_read_jsonl, (TRAIN_PATH, VALID_PATH, TEST_PATH)),
                       ignore_index=True)
    commit_df = _read_jsonl(COMMIT_DATES_PATH)
    commit_df["commit_date"] = pd.to_datetime(commit_df["commit_date"], utc=True)
    merged = df_all.merge(commit_df[["project", "commit_id", "commit_date"]],
                          on="commit_id", how="left", validate="many_to_one")
    if merged["commit_date"].isna().any():
        missing = merged["commit_date"].isna().sum()
        print(f"⚠️  {missing} rows missing commit_date – ignored.")
        merged = merged.dropna(subset=["commit_date"])
    return merged

def draw_samples(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    aft = df[df["commit_date"] >= CUTOFF].sample(N_SAMPLES, random_state=RANDOM_SEED)
    bef = df[df["commit_date"] < CUTOFF].sample(N_SAMPLES, random_state=RANDOM_SEED)
    return aft.reset_index(drop=True), bef.reset_index(drop=True)

# ────────────── Prompt helpers ─────────────
SYSTEM_PROMPT = (
    "You are a code-completion assistant. I will give you the first half of a "
    "C/C++ function plus project+commit. Respond with the *complete* original "
    "function *only* – no markdown, no explanation."
)
_CODE_FENCE = re.compile(r"^```[^\n]*\n(.+?)\n```$", re.S | re.M)

def first_half(code: str) -> str:
    lines = code.splitlines()
    return "\n".join(lines[:len(lines)//2])

def normalise(txt: str) -> str:
    """Strip ``` fences & trailing spaces; normalise newlines."""
    m = _CODE_FENCE.match(txt.strip())
    if m:
        txt = m.group(1)
    return "\n".join(l.rstrip() for l in txt.strip().splitlines())

def complete(code_half: str, project: str, commit: str) -> str:
    user = (f"Project: {project}\nCommit: {commit}\n"
            f"Function (first half):\n```\n{code_half}\n```")
    chat = client.chat.completions.create(
        model=MODEL, temperature=TEMPERATURE,
        messages=[{"role": "system", "content": SYSTEM_PROMPT},
                  {"role": "user", "content": user}]
    )
    return chat.choices[0].message.content

# ────────────── Evaluation ─────────────
def evaluate(df: pd.DataFrame, name: str) -> dict:
    exact, dists, sims, bleus, jaccs = [], [], [], [], []

    iterator = zip(df["func"], df["project_url"], df["commit_id"])
    for func, proj, cid in tqdm(iterator, total=len(df), desc=f"{name:>18}"):
        half = first_half(func)
        try:
            pred_raw = complete(half, proj, cid)
            pred = normalise(pred_raw)
        except Exception as exc:
            print(f"OpenAI error – treating as empty: {exc}")
            pred = ""

        truth = normalise(func)
        match = pred == truth
        exact.append(int(match))

        # Levenshtein
        dist = lev_distance(truth, pred)
        dists.append(dist)
        sims.append(1 - dist / max(len(truth), len(pred), 1))

        # BLEU & Jaccard
        bleus.append(bleu_score(truth, pred))
        jaccs.append(jaccard_similarity(truth, pred))

    return {
        "dataset":         name,
        "accuracy":        float(np.mean(exact)),
        "avg_lev_dist":    float(np.mean(dists)),
        "avg_lev_sim":     float(np.mean(sims)),
        "avg_bleu":        float(np.mean(bleus)),
        "avg_jaccard":     float(np.mean(jaccs)),
        "correct":         int(np.sum(exact)),
        "total":           len(df),
    }

# ────────────── Main entry ─────────────
def main() -> None:
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