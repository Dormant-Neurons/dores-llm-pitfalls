#!/usr/bin/env python3
"""
Evaluation of the fine-tuned PrimeVul commit-message model
———————————————————————————————————————————————
• Uses the two fixed 200-row splits created earlier
    datasets/primevul/primevul_eval_before200.jsonl
    datasets/primevul/primevul_eval_after200.jsonl
• Builds prompts **without** function bodies
• Queries the fine-tuned model (change FT_MODEL below or pass --model)
• Reports accuracy, Levenshtein metrics, BLEU-4 (smooth-4) & Jaccard
"""

from __future__ import annotations
import argparse, functools, json, os, sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from openai import OpenAI
from tqdm import tqdm

# ── Config ─────────────────────────────────────────────────────────────────────
DATA_DIR = Path("datasets/primevul")
BEFORE_PATH = DATA_DIR / "primevul_eval_before200.jsonl"
AFTER_PATH  = DATA_DIR / "primevul_eval_after200.jsonl"

FT_MODEL = "ft:gpt-3.5-turbo-1106:personal:primevul-nofunc-20210901:BfMLy1kL"  # <-- edit

TEMPERATURE = 0.0
PRINT_COMMIT_MESSAGES = True

# ── Optional deps (Levenshtein / nltk) ─────────────────────────────────────────
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
        ref_t, hyp_t = ref.split(), hyp.split()
        return sentence_bleu([ref_t], hyp_t, smoothing_function=_smooth) if hyp_t else 0.0
except ImportError:
    def bleu_score(ref, hyp): return 0.0

def jaccard_similarity(a: str, b: str) -> float:
    s1, s2 = set(a.split()), set(b.split())
    return len(s1 & s2) / len(s1 | s2) if s1 | s2 else 0.0

# ── I/O helpers ────────────────────────────────────────────────────────────────
def _read_jsonl(path: Path) -> pd.DataFrame:
    with path.open(encoding="utf-8") as fh:
        return pd.DataFrame(json.loads(l) for l in fh)

def load_eval_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    return _read_jsonl(AFTER_PATH), _read_jsonl(BEFORE_PATH)

# ── Prompt building (⚠️ no function body) ──────────────────────────────────────
SYSTEM_PROMPT = (
    "You are a commit message assistant. I will give you project+commit and the "
    "first half of the commit message. Predict the full original commit message "
    "only. No markdown or explanation."
)

def build_prompt(project: str, commit: str, partial_msg: str) -> str:
    return (
        f"Project: {project}\n"
        f"Commit: {commit}\n"
        f'Partial commit message: "{partial_msg}"'
    )

def normalise(txt: str) -> str:
    return "\n".join(l.rstrip() for l in txt.strip().splitlines())

def strip_formatting(txt: str) -> str:
    return txt.replace("\n", " ").replace("\t", " ").strip()

# ── OpenAI wrapper ─────────────────────────────────────────────────────────────
def openai_client():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        sys.exit("Set OPENAI_API_KEY env var.")
    return OpenAI(api_key=api_key)

def complete(client: OpenAI, code_unused: str, project: str,
             commit: str, partial_msg: str, model: str) -> str:
    chat = client.chat.completions.create(
        model=model,
        temperature=TEMPERATURE,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": build_prompt(project, commit, partial_msg)}
        ]
    )
    return chat.choices[0].message.content

# ── Evaluation loop ────────────────────────────────────────────────────────────
def evaluate(df: pd.DataFrame, name: str, model: str, client: OpenAI) -> dict:
    exact, dists, sims, bleus, jaccs = [], [], [], [], []

    iterator = zip(df["func"], df["project_url"], df["commit_id"], df["commit_message"])
    for func, proj, cid, true_msg in tqdm(iterator, total=len(df), desc=name):
        truth = normalise(true_msg)
        hint  = " ".join(truth.split()[: len(truth.split()) // 2])

        try:
            pred = normalise(complete(client, "", proj, cid, hint, model))
        except Exception as e:
            print(f"OpenAI error – treating as empty: {e}")
            pred = ""

        tgt = strip_formatting(truth)
        out = strip_formatting(pred)

        if PRINT_COMMIT_MESSAGES:
            print("\n=== COMMIT MESSAGE ===")
            print(f"Hint     : {hint}\nActual   : {tgt}\nPredicted: {out}")

        match = int(out == tgt)
        exact.append(match)

        dist = lev_distance(tgt, out)
        dists.append(dist)
        sims.append(1 - dist / max(len(tgt), len(out), 1))
        bleus.append(bleu_score(tgt, out))
        jaccs.append(jaccard_similarity(tgt, out))

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

# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=FT_MODEL,
                        help="Fine-tuned model name (default set in script)")
    args = parser.parse_args()

    after_df, before_df = load_eval_frames()
    client = openai_client()

    results = [
        evaluate(after_df,  "after-cutoff-200",  args.model, client),
        evaluate(before_df, "before-cutoff-200", args.model, client),
    ]

    print("\n===== Summary =====")
    for r in results:
        print(
            f"{r['dataset']:>18}: "
            f"Acc={r['accuracy']:.2%}  "
            f"LevDist={r['avg_lev_dist']:.1f}  "
            f"LevSim={r['avg_lev_sim']:.2%}  "
            f"BLEU={r['avg_bleu']:.3f}  "
            f"Jacc={r['avg_jaccard']:.2%}  "
            f"({r['correct']}/{r['total']} exact)"
        )

if __name__ == "__main__":
    main()