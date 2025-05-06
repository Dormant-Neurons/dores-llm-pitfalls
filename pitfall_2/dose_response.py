#!/usr/bin/env python3
"""
Leakage‑ratio experiment for vulnerability‑detection corpora with
CodeT5p‑220M.

Supported datasets (pre‑defined mappings)
----------------------------------------
* BigVul      → ``bstee615/bigvul``                (columns: ``func_before``, ``vul``)
* Devign      → ``google/code_x_glue_cc_defect_detection``  (``func``, ``target``)
* DiverseVul  → ``bstee615/diversevul``            (``func``, ``target``)
* PrimeVul    → ``colin/PrimeVul``                (``func``, ``target``)

For each leak ratio in {0 %,20 %,…,100 %} the script leaks the chosen
fraction of *test* into *training*, fine‑tunes (default 1 epoch), then
prints **overall F1 on the full test set**.

CLI examples
~~~~~~~~~~~~
```
python leak_ratio_experiment.py --dataset BigVul     --epochs 1
python leak_ratio_experiment.py --dataset Devign     --epochs 2
python leak_ratio_experiment.py --dataset colin/PrimeVul --epochs 1  # custom path still works
```
"""

import argparse
import os
import sys
from typing import Any, Dict

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

# ───────────────────────── constants ─────────────────────────────
MODEL_ID    = "Salesforce/codet5p-220m"
CTX_LIMIT   = 512
SPLIT_SEED  = 42
LEAK_RATIOS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

DATASETS: Dict[str, Dict[str, Any]] = {
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

# ───────────────────────── helpers ───────────────────────────────

def standardise_columns(ds: Dataset, code_key: str, label_key: str, positive_val: Any) -> Dataset:
    """Rename columns to `func`, `target` and convert labels to 0/1 ints."""

    def convert(example):
        example["func"] = example[code_key]
        lbl = example[label_key]
        if isinstance(lbl, bool):
            example["target"] = int(lbl == positive_val)
        else:
            example["target"] = 1 if lbl == positive_val else 0
        return example

    cols_to_remove = [c for c in ds.column_names if c not in {code_key, label_key}]
    # run map in batches to keep it fast
    ds = ds.map(convert, remove_columns=cols_to_remove, batched=False)
    # finally drop the original cols we relabelled
    ds = ds.remove_columns([code_key, label_key])
    return ds


def make_random_splits(ds_all: Dataset, tokenizer) -> DatasetDict:
    """Return 60/20/20 random split with pre‑computed token counts."""

    def add_toklen(ex):
        ex["n_tok"] = len(tokenizer(ex["func"]).input_ids)
        return ex

    ds_all = ds_all.map(add_toklen, num_proc=os.cpu_count())
    ds_all = ds_all.shuffle(seed=SPLIT_SEED)

    n = len(ds_all)
    train_end = int(0.6 * n)
    val_end   = int(0.8 * n)
    return DatasetDict({
        "train":      ds_all.select(range(train_end)),
        "validation": ds_all.select(range(train_end, val_end)),
        "test":       ds_all.select(range(val_end, n)),
    })


def tokenize_func(batch, tokenizer):
    codes = batch["func"]
    labels = batch["target"]
    model_inputs = tokenizer(codes, truncation=True, max_length=CTX_LIMIT)
    model_inputs["labels"] = [tokenizer(str(int(x))).input_ids for x in labels]
    return model_inputs


@torch.no_grad()
def eval_f1(ds: Dataset, tokenizer, model) -> float:
    model.eval()
    BATCH = 32
    y_true, y_pred = [], []
    for i in range(0, len(ds), BATCH):
        batch = ds.select(range(i, min(i + BATCH, len(ds))))
        enc = tokenizer(batch["func"], truncation=True, max_length=CTX_LIMIT,
                        padding=True, return_tensors="pt").to(model.device)
        outs = model.generate(**enc, max_length=3, num_beams=1)
        preds = tokenizer.batch_decode(outs, skip_special_tokens=True)
        y_pred.extend([1 if p.strip().startswith("1") else 0 for p in preds])
        y_true.extend(batch["target"])
    return f1_score(y_true, y_pred, average="macro")


# ───────────────────────── core experiment ───────────────────────

def run_for_ratio(ratio: float, splits: DatasetDict, tokenizer, epochs: int, tag: str):
    """Fine‑tune at a given leakage ratio and print overall test F1."""
    rng = np.random.default_rng(SPLIT_SEED)
    leak_size = int(ratio * len(splits["test"]))
    leak_idx  = rng.choice(len(splits["test"]), size=leak_size, replace=False) if leak_size else []
    leak_ds   = splits["test"].select(leak_idx) if leak_size else None

    train_ds = (concatenate_datasets([splits["train"], leak_ds]).shuffle(seed=SPLIT_SEED)
                if leak_size else splits["train"])

    # Tokenise → we *explicitly* remove only the known raw columns to avoid the
    # rare edge case where remove_columns=train_ds.column_names would wipe out
    # everything when train_ds already lacks those names (causing len=0).
    cols_to_strip = [c for c in ["func", "target", "n_tok"] if c in train_ds.column_names]

    tokenised_train = train_ds.map(
        lambda ex: tokenize_func(ex, tokenizer),
        remove_columns=cols_to_strip,
        batched=True,
        num_proc=1,
    )
    tokenised_val = splits["validation"].map(
        lambda ex: tokenize_func(ex, tokenizer),
        remove_columns=cols_to_strip,
        batched=True,
        num_proc=1,
    )

    if len(tokenised_train) == 0:
        print(f"{tag:10s} | Leak {int(ratio*100):3d}% | *SKIPPED* (train set empty)")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID, low_cpu_mem_usage=True).to(device)

    out_dir = f"./codet5p_{tag}_leak_{int(ratio*100)}"
    args = Seq2SeqTrainingArguments(
        output_dir=out_dir,
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

    f1_full = eval_f1(splits["test"], tokenizer, model)
    print(f"{tag:10s} | Leak {int(ratio*100):3d}% | F1(full test) = {f1_full:.3f}")

    del model; torch.cuda.empty_cache()

# ───────────────────────── main ────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Leakage ratio experiment")
    parser.add_argument("--dataset", required=True,
                        help="Dataset key (BigVul, Devign, DiverseVul, PrimeVul) or 🤗 hub path")
    parser.add_argument("--epochs", type=int, default=1, help="Fine‑tuning epochs per ratio")
    args = parser.parse_args()
    ...():
    parser = argparse.ArgumentParser(description="Leakage ratio experiment")
    parser.add_argument("--dataset", required=True,
                        help="Dataset key (BigVul, Devign, DiverseVul, PrimeVul) or 🤗 hub path")
    parser.add_argument("--epochs", type=int, default=1, help="Fine‑tuning epochs per ratio")
    args = parser.parse_args()

    # Resolve dataset
    if args.dataset in DATASETS:
        cfg = DATASETS[args.dataset]
        hf_path   = cfg["hf_id"]
        code_key  = cfg["code_key"]
        label_key = cfg["label_key"]
        positive  = cfg["positive"]
        tag       = args.dataset
    else:
        # Assume direct HF path with default column names
        hf_path   = args.dataset
        code_key  = "func"
        label_key = "target"
        positive  = 1  # assume 1 is positive
        tag       = hf_path.split("/")[-1]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f">> device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True, use_fast=True)

    ds_all = load_dataset(hf_path, split="train+validation+test")
    ds_all = standardise_columns(ds_all, code_key, label_key, positive)

    splits = make_random_splits(ds_all, tokenizer)

    for ratio in LEAK_RATIOS:
        run_for_ratio(ratio, splits, tokenizer, args.epochs, tag)


if __name__ == "__main__":
    main()
