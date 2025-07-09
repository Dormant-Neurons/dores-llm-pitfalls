#!/usr/bin/env python3
"""primevul_multi_backend_eval.py
=================================================
Benchmark PrimeVul commit-message completion across **four** back-ends:

* **OpenAI** (GPT-4o, GPT-4o-mini, GPT-4, etc.)
* **Anthropic Claude** (claude-3.x)
* **DeepSeek** (chat / reasoner models)
* **Local OpenAI-compatible** servers (e.g. *LM Studio* hosting Llama-3-8B-Instruct)

The script samples **exactly 100 unique commits** (fixed seed ⇒ deterministic) and
reports accuracy and text-similarity metrics.

---------------  Quick start  ----------------
```bash
# 1. Install deps
pip install openai anthropic pandas numpy scikit-learn tqdm python-Levenshtein nltk

# 2. Choose a back-end (env var) and export needed keys
export BACKEND="openai"          # or anthropic | deepseek | local
export OPENAI_API_KEY="sk-..."   # for openai backend

# For anthropic
export ANTHROPIC_API_KEY="..."

# For deepseek (OpenAI-compatible)
export DEEPSEEK_API_KEY="..."

# For local (LM Studio)
export LLM_BASE_URL="http://127.0.0.1:1234"   # omit /v1
export LLM_API_KEY="lm-studio"                # anything non-empty

# 3. Run
python primevul_multi_backend_eval.py
```

CLI flags override env vars::

```bash
python primevul_multi_backend_eval.py --backend anthropic --model claude-3-opus-20240229
```
"""
from __future__ import annotations

import argparse
import functools
import json
import os
import random
import sys
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score  # noqa: F401 (imported for reference)
from tqdm import tqdm
from dotenv import load_dotenv 

# ── .env loader ──
try: # type: ignore

    load_dotenv()  # load environment variables from .env if present
except ImportError:  # pragma: no cover
    print("[INFO] python-dotenv not installed – skipping .env loading.", file=sys.stderr)

# ── Optional metric libs ──
try:
    import Levenshtein  # type: ignore

    lev_distance: Callable[[str, str], int] = Levenshtein.distance  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover – fallback pure-python

    def lev_distance(a: str, b: str) -> int:  # type: ignore
        @functools.lru_cache(None)
        def _ld(i: int, j: int) -> int:
            if i == 0:
                return j
            if j == 0:
                return i
            cost = 0 if a[i - 1] == b[j - 1] else 1
            return min(
                _ld(i - 1, j) + 1, _ld(i, j - 1) + 1, _ld(i - 1, j - 1) + cost,
            )

        return _ld(len(a), len(b))

try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction  # type: ignore

    _smooth = SmoothingFunction().method4  # type: ignore[attr-defined]

    def bleu_score(ref: str, hyp: str) -> float:  # noqa: D401
        """Sentence BLEU with smoothing (nltk method4)."""

        ref_tok, hyp_tok = ref.split(), hyp.split()
        return sentence_bleu([ref_tok], hyp_tok, smoothing_function=_smooth) if hyp_tok else 0.0

except ImportError:  # pragma: no cover – BLEU fallback

    def bleu_score(ref: str, hyp: str) -> float:  # type: ignore
        return 0.0


def jaccard_similarity(a: str, b: str) -> float:
    s1, s2 = set(a.split()), set(b.split())
    return len(s1 & s2) / len(s1 | s2) if s1 | s2 else 0.0


# ── Prompt template ──
SYSTEM_PROMPT = (
    "You are a commit message assistant. I will give you project+commit+partial message. "
    "Predict the full original commit message only. No markdown or explanation."
)


# ── Unified backend helpers ──
class CompletionBackend:
    """Wrapper that hides backend-specific API calls."""

    def __init__(self, name: str, model: str, max_tokens: int = 1024):
        self.name = name.lower()
        self.model = model
        self.max_tokens = max_tokens

        if self.name == "openai":
            from openai import OpenAI

            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise RuntimeError("OPENAI_API_KEY not set.")
            self.client = OpenAI(api_key=api_key)
            self._call = self._openai_call  # type: ignore[attr-defined]

        elif self.name == "anthropic":
            from anthropic import Anthropic  # type: ignore

            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise RuntimeError("ANTHROPIC_API_KEY not set.")
            self.client = Anthropic(api_key=api_key)
            self._call = self._anthropic_call  # type: ignore[attr-defined]

        elif self.name == "deepseek":
            from openai import OpenAI

            api_key = os.getenv("DEEPSEEK_API_KEY")
            if not api_key:
                raise RuntimeError("DEEPSEEK_API_KEY not set.")
            self.client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
            self._call = self._openai_call  # type: ignore[attr-defined]

        elif self.name == "local":
            from openai import OpenAI

            base_url = os.getenv("LLM_BASE_URL", "http://127.0.0.1:1234")
            api_key = os.getenv("LLM_API_KEY", "lm-studio")
            self.client = OpenAI(api_key=api_key, base_url=f"{base_url.rstrip('/')}/v1")
            self._call = self._openai_call  # type: ignore[attr-defined]
        else:
            raise ValueError(f"Unsupported backend: {name}")

    # ---- backend-specific call implementations ----
    def _openai_call(self, user_prompt: str) -> str:  # noqa: D401
        response = self.client.chat.completions.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=0.0,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response.choices[0].message.content

    def _anthropic_call(self, user_prompt: str) -> str:  # noqa: D401
        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=0.0,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return response.content[0].text

    # ---- public API ----
    def complete(self, project: str, commit: str, partial: str, suffix: str = "") -> str:
        user_prompt = (
            f"Project: {project}\nCommit: {commit}\nPartial commit message: \"{partial}\""
        )
        if suffix:
            user_prompt += f"\n{suffix}"
        try:
            return self._call(user_prompt)  # type: ignore[misc]
        except Exception as exc:  # pragma: no cover – network/API errors
            print(f"[WARN] Completion error ({self.name}) – returning empty string: {exc}", file=sys.stderr)
            return ""


# ── Dataset helpers ──
DATA_DIR = Path("datasets/primevul")
TRAIN_PATH, VALID_PATH, TEST_PATH = (
    DATA_DIR / f"primevul_{split}.jsonl" for split in ("train", "valid", "test")
)
COMMIT_DATES_PATH = DATA_DIR / "commit_dates.jsonl"  # still used for metadata join

RANDOM_SEED = 42
N_SAMPLES = 100  # fixed sample size


def _read_jsonl(path: Path) -> pd.DataFrame:
    with path.open(encoding="utf-8") as fh:
        return pd.DataFrame(json.loads(l) for l in fh)


def load_dataset() -> pd.DataFrame:
    """Load & merge commit metadata, return DataFrame of unique commits."""

    df_all = pd.concat((_read_jsonl(p) for p in (TRAIN_PATH, VALID_PATH, TEST_PATH)), ignore_index=True)
    df_all = df_all[df_all["commit_message"] != "None"]

    commit_df = _read_jsonl(COMMIT_DATES_PATH)[["project", "commit_id", "commit_date"]]
    df = df_all.merge(commit_df, on="commit_id", how="left").drop_duplicates("commit_id")
    return df.reset_index(drop=True)


def sample_commits(df: pd.DataFrame, n: int, rng_seed: int) -> pd.DataFrame:
    if len(df) < n:
        raise RuntimeError(f"Dataset only has {len(df)} unique commits (<{n}).")
    return df.sample(n, random_state=rng_seed).reset_index(drop=True)


# ── Text helpers ──

def normalise(txt: str) -> str:
    return "\n".join(l.rstrip() for l in txt.strip().splitlines())


def strip_formatting(txt: str) -> str:
    return txt.replace("\n", " ").replace("\t", " ").strip()


# ── Evaluation loop ──

def evaluate(df: pd.DataFrame, backend: CompletionBackend, show_commits: bool = False, suffix: str = "") -> Dict[str, float | int]:
    exact, dists, sims, bleus, jaccs = [], [], [], [], []

    iterator: List[Tuple[str, str, str]] = list(zip(df["project_url"], df["commit_id"], df["commit_message"]))

    for proj, cid, true_msg in tqdm(iterator, desc="Evaluating", total=len(df)):
        norm_truth = normalise(true_msg)
        hint_words = norm_truth.split()
        hint = " ".join(hint_words[: len(hint_words) // 2])  # first half as partial

        pred = normalise(backend.complete(proj, cid, hint, suffix=suffix))

        truth_stripped = strip_formatting(norm_truth)
        pred_stripped = strip_formatting(pred)

        if show_commits:
            print("\n=== COMMIT MESSAGE ===")
            print(f"Hint     : {hint}")
            print(f"Actual   : {truth_stripped}")
            print(f"Predicted: {pred_stripped}")

        match = pred_stripped == truth_stripped
        exact.append(int(match))

        dist = lev_distance(truth_stripped, pred_stripped)
        dists.append(dist)
        sims.append(1 - dist / max(len(truth_stripped), len(pred_stripped), 1))

        bleus.append(bleu_score(truth_stripped, pred_stripped))
        jaccs.append(jaccard_similarity(truth_stripped, pred_stripped))

    return {
        "accuracy": float(np.mean(exact)),
        "avg_lev_dist": float(np.mean(dists)),
        "avg_lev_sim": float(np.mean(sims)),
        "avg_bleu": float(np.mean(bleus)),
        "avg_jaccard": float(np.mean(jaccs)),
        "correct": int(np.sum(exact)),
        "total": len(df),
    }


# ── CLI interface ──

def parse_args() -> argparse.Namespace:  # noqa: D401
    parser = argparse.ArgumentParser(description="PrimeVul commit-message benchmark (multi-backend)")
    parser.add_argument(
        "--backend",
        choices=["openai", "anthropic", "deepseek", "local"],
        default=os.getenv("BACKEND"),
        help="Which backend to use (env BACKEND overrides).",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("MODEL_NAME", "gpt-4o-mini"),
        help="Model name (env MODEL_NAME).",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default="",
        help="Optional text to append to the end of the prompt.",
    )
    parser.add_argument("--show", action="store_true", help="Print commit messages + predictions.")
    return parser.parse_args()


# ── Main entrypoint ──

def main() -> None:  # noqa: D401
    args = parse_args()

    # Auto-detect backend if not provided
    backend_name = args.backend or (
        "anthropic" if os.getenv("ANTHROPIC_API_KEY") else ("deepseek" if os.getenv("DEEPSEEK_API_KEY") else ("openai" if os.getenv("OPENAI_API_KEY") else "local"))
    )

    backend = CompletionBackend(backend_name, args.model)

    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    df_all = load_dataset()
    sample_df = sample_commits(df_all, N_SAMPLES, RANDOM_SEED)

    metrics = evaluate(sample_df, backend, show_commits=args.show, suffix=args.suffix)

    print("\n===== Summary =====")
    print(
        f"Backend={backend.name}  Model={backend.model}\n"
        f"Acc        : {metrics['accuracy']:.2%} ({metrics['correct']}/{metrics['total']})\n"
        f"Levenshtein: dist={metrics['avg_lev_dist']:.2f}  sim={metrics['avg_lev_sim']:.2%}\n"
        f"BLEU       : {metrics['avg_bleu']:.3f}\n"
        f"Jaccard    : {metrics['avg_jaccard']:.2%}"
    )


if __name__ == "__main__":
    main()
