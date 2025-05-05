#!/usr/bin/env python
"""
Token‑length statistics for four vulnerability datasets
+ LaTeX table output.

Datasets  | code field | label field | positive label
----------|------------|-------------|----------------
bstee615/bigvul                         func_before   vul            1
google/code_x_glue_cc_defect_detection  func          target         True
bstee615/diversevul                     func          target         1
colin/PrimeVul                          func          target         1

Tokenizer  : Salesforce/codet5-small (fast)
Cut‑offs   : 512 / 1 024 / 2 048 tokens

Author : <you>
Date   : 2025‑05‑05
"""

import multiprocessing as mp
from functools import partial
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer

# ────────── config ───────────────────────────────────────────────────────── #

CUT_OFFS = (512, 1024, 2048)

DATASETS = {
    "BigVul": {
        "hf_id":     "bstee615/bigvul",
        "code_key":  "func_before",
        "label_key": "vul",
        "positive":  1,
    },
    "Devign": {
        "hf_id":     "google/code_x_glue_cc_defect_detection",
        "code_key":  "func",
        "label_key": "target",
        "positive":  True,
    },
    "DiverseVul": {
        "hf_id":     "bstee615/diversevul",
        "code_key":  "func",
        "label_key": "target",
        "positive":  1,
    },
    "PrimeVul": {
        "hf_id":     "colin/PrimeVul",
        "code_key":  "func",
        "label_key": "target",
        "positive":  1,
    },
}

# ────────── helpers ──────────────────────────────────────────────────────── #

def add_len_flags(example, code_key, cutoffs, tokenizer):
    """Compute token length and flags for 'longer than' each cut‑off."""
    length = len(tokenizer(example[code_key]).input_ids)
    flags  = {f"gt_{c}": length > c for c in cutoffs}
    return {"tok_len": length, **flags}


def process_dataset(name, spec, tokenizer, cutoffs):
    """Return total vulnerable and counts > each cut‑off."""
    print(f"\n▶  Loading {name} …")
    ds = load_dataset(spec["hf_id"], split="train+validation+test")

    # keep only vulnerable functions
    ds = ds.filter(lambda x: x[spec["label_key"]] == spec["positive"])

    fn = partial(add_len_flags,
                 code_key=spec["code_key"],
                 cutoffs=cutoffs,
                 tokenizer=tokenizer)
    ds = ds.map(fn,
                num_proc=mp.cpu_count(),
                desc=f"Tokenising ({len(ds):,} rows)")

    total  = len(ds)
    counts = {c: int(sum(ds[f"gt_{c}"])) for c in cutoffs}   # ← fixed
    return total, counts


def make_latex(results, cutoffs):
    """
    Build a LaTeX `table` float with booktabs rules, a caption, and a label.

    The table shows the total number of vulnerable functions in each corpus
    and the percentage whose CodeT5‑tokenised length exceeds 512 / 1024 / 2048
    tokens (one decimal digit).
    """
    # lrrrr  – one left‑aligned dataset name + 1 total + N % columns
    cols_spec = "l" + "r" * (1 + len(cutoffs))

    header = (
        "Dataset & Total "
        + " ".join(f"& $>{c}$\\,\\% " for c in cutoffs)
        + r"\\"
    )

    lines = [
        r"\begin{table}[ht]",
        r"  \centering",
        r"  \caption{Proportion of \emph{vulnerable} functions whose "
        r"CodeT5‑tokenised length exceeds common transformer context‑window "
        r"sizes (512, 1\,024, and 2\,048 tokens) in four public "
        r"vulnerability‑detection datasets.\label{tab:token-cutoffs}}",
        f"  \\begin{{tabular}}{{{cols_spec}}}",
        r"    \toprule",
        "    " + header,
        r"    \midrule",
    ]

    for name, (total, counts) in results.items():
        row = [name, f"{total:,}"]
        for c in cutoffs:
            pct = counts[c] / total * 100
            row.append(f"{pct:.1f}")
        lines.append("    " + " & ".join(row) + r" \\")
    lines.extend([
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table}",
    ])
    return "\n".join(lines)


# ────────── main ─────────────────────────────────────────────────────────── #

def main():
    tokenizer = AutoTokenizer.from_pretrained(
        "Salesforce/codet5-small", use_fast=True
    )

    results = {}
    for name, spec in DATASETS.items():
        total, counts = process_dataset(name, spec, tokenizer, CUT_OFFS)
        results[name] = (total, counts)

        # Console summary
        print(f"\n{name}  – vulnerable functions: {total:,}")
        for c in CUT_OFFS:
            pct = counts[c] / total * 100
            print(f"   > {c:4d} tokens : {counts[c]:8,}  ({pct:6.2f} %)")
        print()

    # Build LaTeX table
    latex = make_latex(results, CUT_OFFS)
    print("\n────────── LaTeX table ──────────\n")
    print(latex)
    print("\n────────── end LaTeX ────────────")

    # Write to file
    Path("pitfall_5/generated_latex/token_cutoff_table.tex").write_text(latex)
    print("\nLaTeX table saved to token_cutoff_table.tex")


if __name__ == "__main__":
    main()