#!/usr/bin/env python3
"""
Leakage‑ratio experiment for vulnerability‑detection datasets with
CodeT5p‑220M.

Supports the following ready‑made mappings (edit DATASETS to add more):

    BigVul     →  bstee615/bigvul                    (func_before, vul)
    Devign     →  google/code_x_glue_cc_defect_detection (func, target)
    DiverseVul →  bstee615/diversevul                (func, target)
    PrimeVul   →  colin/PrimeVul                     (func, target)

If you pass an arbitrary 🤗 Hub path that isn’t in DATASETS the script
assumes the columns are named ``func`` and ``target`` with positive
label 1.

For each leakage ratio r ∈ {0,0.2,…,1.0} the script:
  1. Copies ⌊r·|test|⌋ random test rows into the training set.
  2. Fine‑tunes CodeT5p for the chosen number of epochs (default 1).
  3. Reports **overall macro‑F1 on the full test set**.

Example:
    python leak_ratio_experiment.py --dataset BigVul --epochs 1

Author: <you>   2025‑05‑05
"""

import argparse
from typing import Any, Dict
import os

import numpy as np
import torch
from datasets import load_dataset, concatenate_datasets, DatasetDict, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
)
from sklearn.metrics import f1_score

# ───────────────────────── configuration ──────────────────────────
MODEL_ID = "Salesforce/codet5p-220m"
CTX_LIMIT = 512
SPLIT_SEED = 42
LEAK_RATIOS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

DATASETS: Dict[str, Dict[str, Any]] = {
    "BigVul": {
        "hf_id": "bstee615/bigvul",
        "code_key": "func_before",
        "label_key": "vul",
        "positive": 1,
    },
    "Devign": {
        "hf_id": "google/code_x_glue_cc_defect_detection",
        "code_key": "func",
        "label_key": "target",
        "positive": True,
    },
    "DiverseVul": {
        "hf_id": "bstee615/diversevul",
        "code_key": "func",
        "label_key": "target",
        "positive": 1,
    },
    "PrimeVul": {
        "hf_id": "colin/PrimeVul",
        "code_key": "func",
        "label_key": "target",
        "positive": 1,
    },
}

# ───────────────────────── helper functions ───────────────────────

def standardise_columns(ds: Dataset, code_key: str, label_key: str, positive_val: Any) -> Dataset:
    """Rename columns to `func`, `target` and make labels 0/1 ints."""

    def convert(ex):
        ex["func"] = ex.pop(code_key)
        lbl = ex.pop(label_key)
        ex["target"] = int(lbl == positive_val) if isinstance(lbl, (int, bool)) else 0
        return ex

    keep_cols = [code_key, label_key]
    drop_cols = [c for c in ds.column_names if c not in keep_cols]
    ds = ds.remove_columns(drop_cols)
    ds = ds.map(convert)
    return ds


def add_toklen_and_split(ds_all: Dataset, tokenizer) -> DatasetDict:
    """Add token count and produce 60/20/20 random split."""

    def add_tok(ex):
        ex["n_tok"] = len(tokenizer(ex["func"]).input_ids)
        return ex

    ds_all = ds_all.map(add_tok, num_proc=1)
    ds_all = ds_all.shuffle(seed=SPLIT_SEED)

    n = len(ds_all)
    train_end, val_end = int(0.6 * n), int(0.8 * n)
    return DatasetDict({
        "train": ds_all.select(range(train_end)),
        "validation": ds_all.select(range(train_end, val_end)),
        "test": ds_all.select(range(val_end, n)),
    })


def tokenize_func(batch, tokenizer):
    codes = batch["func"]
    labels = batch["target"]
    enc = tokenizer(codes, truncation=True, max_length=CTX_LIMIT)
    enc["labels"] = [tokenizer(str(int(l))).input_ids for l in labels]
    return enc


@torch.no_grad()
def eval_f1(ds: Dataset, tokenizer, model):
    model.eval()
    B = 32
    y_true, y_pred = [], []
    for i in range(0, len(ds), B):
        batch = ds.select(range(i, min(i + B, len(ds))))
        enc = tokenizer(batch["func"], truncation=True, max_length=CTX_LIMIT,
                        padding=True, return_tensors="pt").to(model.device)
        outs = model.generate(**enc, max_length=3, num_beams=1)
        dec = tokenizer.batch_decode(outs, skip_special_tokens=True)
        y_pred.extend([1 if t.strip().startswith("1") else 0 for t in dec])
        y_true.extend(batch["target"])
    return f1_score(y_true, y_pred, average="macro")


# ───────────────────────── core routine ───────────────────────────

def run_for_ratio(ratio: float, splits: DatasetDict, tokenizer, epochs: int, tag: str):
    rng = np.random.default_rng(SPLIT_SEED)
    leak_n = int(ratio * len(splits["test"]))
    leak_idx = rng.choice(len(splits["test"]), size=leak_n, replace=False) if leak_n else []
    leak_ds = splits["test"].select(leak_idx) if leak_n else None

    train_ds = concatenate_datasets([splits["train"], leak_ds]).shuffle(seed=SPLIT_SEED) if leak_n else splits["train"]

    # Tokenise
    cols_strip = [c for c in ("func", "target", "n_tok") if c in train_ds.column_names]
    tokenised_train = train_ds.map(lambda ex: tokenize_func(ex, tokenizer),
                                   remove_columns=cols_strip, batched=True, num_proc=1)
    tokenised_val = splits["validation"].map(lambda ex: tokenize_func(ex, tokenizer),
                                             remove_columns=cols_strip, batched=True, num_proc=1)

    if len(tokenised_train) == 0:
        print(f"{tag:10s} | Leak {int(ratio*100):3d}% | *SKIPPED* (empty train set)")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID, low_cpu_mem_usage=True).to(device)

    args = Seq2SeqTrainingArguments(
        output_dir=f"./codet5p_{tag}_leak_{int(ratio*100)}",
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        learning_rate=3e-5,
        num_train_epochs=epochs,
        save_strategy="no",
        logging_strategy="no",
        fp16=(device.type == "cuda"),
        seed=SPLIT_SEED,
    )

    trainer = Seq2SeqTrainer(
        model=model, args=args,
        train_dataset=tokenised_train,
        eval_dataset=tokenised_val,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model),
        tokenizer=tokenizer,
    )
    trainer.train()

    f1 = eval_f1(splits["test"], tokenizer, model)
    print(f"{tag:10s} | Leak {int(ratio*100):3d}% | F1(full test) = {f1:.3f}")

    del model; torch.cuda.empty_cache()


# ──────────────────────────── main ───────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Leakage dose‑response experiment")
    parser.add_argument("--dataset", required=True,
                        help="Dataset key (e.g. BigVul) or HF path (owner/name)")
    parser.add_argument("--epochs", type=int, default=1,
                        help="Fine‑tuning epochs per leakage ratio (default 1)")
    args = parser.parse_args()

    # Resolve mapping or fall back to defaults
    if args.dataset in DATASETS:
        cfg = DATASETS[args.dataset]
        hf_id, code_key, label_key, positive = cfg.values()
        tag = args.dataset
    else:
        hf_id, code_key, label_key, positive = args.dataset, "func", "target", 1
        tag = hf_id.split("/")[-1]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f">> device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True, use_fast=True)

    HF_TOKEN = os.getenv("HF_TOKEN")  # set in sbatch: export HF_TOKEN=hf_xxx
    ds_all = load_dataset(hf_id, split="train+validation+test",
                      use_auth_token=HF_TOKEN)

    ds_all = standardise_columns(ds_all, code_key, label_key, positive)

    splits = add_toklen_and_split(ds_all, tokenizer)

    for r in LEAK_RATIOS:
        run_for_ratio(r, splits, tokenizer, args.epochs, tag)


if __name__ == "__main__":
    main()
