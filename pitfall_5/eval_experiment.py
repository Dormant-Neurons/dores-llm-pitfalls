#!/usr/bin/env python3
"""
Fine‑tune CodeT5p‑220M on PrimeVul with a 512‑token limit
using a CUDA‑enabled GPU, then evaluate on two 1 000‑sample
subsets (≤512 and >512 tokens).

Author: <you>   Date: 2025‑05‑05
"""

import os
import sys
from typing import List, Dict

import numpy as np
import torch
from datasets import load_dataset, DatasetDict, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
)
from sklearn.metrics import f1_score
from tqdm import tqdm

MODEL_ID = "Salesforce/codet5p-220m"
CTX_LIMIT = 512
SPLIT_SEED = 42
EVAL_N = 1_000  # per bucket
OUTPUT_DIR = "./codet5p_primevul_512_cuda"

# ───────────────────────── helpers ───────────────────────── #

def make_splits(tokenizer) -> DatasetDict:
    """60/20/20 split of PrimeVul with token counts pre‑computed."""
    ds_all = load_dataset("colin/PrimeVul", split="train+validation+test")

    def add_toklen(ex):
        ex["n_tok"] = len(tokenizer(ex["func"]).input_ids)
        return ex

    ds_all = ds_all.map(add_toklen, num_proc=os.cpu_count())

    # shuffle once, then train / val / test proportional split
    ds_all = ds_all.shuffle(seed=SPLIT_SEED)
    n = len(ds_all)
    train_end = int(0.6 * n)
    val_end = int(0.8 * n)
    return DatasetDict(
        {
            "train": ds_all.select(range(train_end)),
            "validation": ds_all.select(range(train_end, val_end)),
            "test": ds_all.select(range(val_end, n)),
        }
    )


def tokenize_func(batch, tokenizer):
    """
    Batched tokenisation:
      • batch["func"]   : list[str]
      • batch["target"] : list[[0] or [1]]
    Returns dict of equal‑length lists ready for Arrow.
    """
    codes = batch["func"]
    raw_labels = [
        t[0] if isinstance(t, list) else t  # unwrap [0]/[1] → 0/1
        for t in batch["target"]
    ]
    # ---- tokenise code (trunc 512) ----
    model_inputs = tokenizer(
        codes,
        truncation=True,
        max_length=CTX_LIMIT,
    )

    # ---- tokenise labels ("0" / "1") ----
    label_texts = [str(int(x)) for x in raw_labels]
    # use same tokenizer; tiny sequences, so loop is fine
    model_inputs["labels"] = [tokenizer(t).input_ids for t in label_texts]
    return model_inputs


def build_eval_subsets(test_ds: Dataset, tokenizer):
    short_idx = [i for i, n in enumerate(test_ds["n_tok"]) if n <= CTX_LIMIT]
    long_idx = [i for i, n in enumerate(test_ds["n_tok"]) if n > CTX_LIMIT]

    rng = np.random.default_rng(SPLIT_SEED)
    short_sample = rng.choice(short_idx, size=EVAL_N, replace=False)
    long_sample = rng.choice(long_idx, size=EVAL_N, replace=False)

    short_ds = test_ds.select(short_sample)
    long_ds = test_ds.select(long_sample)
    return short_ds, long_ds


@torch.no_grad()
def eval_f1(ds: Dataset, tokenizer, model, desc: str) -> float:
    model.eval()
    BATCH = 32  # adjust if VRAM is limited
    y_true, y_pred = [], []

    for i in tqdm(range(0, len(ds), BATCH), desc=desc, ncols=80):
        batch = ds.select(range(i, min(i + BATCH, len(ds))))
        enc = tokenizer(
            batch["func"],
            truncation=True,
            max_length=CTX_LIMIT,
            padding=True,
            return_tensors="pt",
        ).to(model.device)

        outs = model.generate(
            **enc,
            max_length=3,
            num_beams=1,
            do_sample=False,
        )
        preds = tokenizer.batch_decode(outs, skip_special_tokens=True)
        y_pred.extend([1 if p.strip().startswith("1") else 0 for p in preds])
        y_true.extend([int(x) for x in batch["target"]])

    return f1_score(y_true, y_pred, average="macro")


# ───────────────────────── main ─────────────────────────── #

def main():
    import transformers

    print("Transformers version :", transformers.__version__)
    print("Loaded from          :", transformers.__file__)
    print(
        "Python path entry    :",
        next(p for p in sys.path if transformers.__file__.startswith(p)),
    )

    # Prefer CUDA if available, otherwise CPU fallback
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f">> device: {device}")

    # Set seeds for reproducibility
    torch.manual_seed(SPLIT_SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(SPLIT_SEED)

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, trust_remote_code=True, use_fast=True
    )
    splits = make_splits(tokenizer)

    # tokenise train/val
    tokenised = splits.map(
        lambda ex: tokenize_func(ex, tokenizer),
        remove_columns=splits["train"].column_names,
        batched=True,
        num_proc=1,  # simple & portable
    )

    dtype = torch.float16 if device.type == "cuda" else torch.float32
    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(device)

    args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        learning_rate=3e-5,
        num_train_epochs=3,
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=100,
        predict_with_generate=True,
        fp16=(device.type == "cuda" and dtype == torch.float16),
        save_total_limit=2,
        seed=SPLIT_SEED,
        dataloader_pin_memory=(device.type == "cuda"),
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=tokenised["train"],
        eval_dataset=tokenised["validation"],
        tokenizer=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model),
    )

    trainer.train()
    trainer.save_model(f"{OUTPUT_DIR}/final")

    # ─── evaluation on two 1 k subsets ─────────────────── #
    short_ds, long_ds = build_eval_subsets(splits["test"], tokenizer)

    f1_short = eval_f1(short_ds, tokenizer, model, "Eval ≤512")
    f1_long = eval_f1(long_ds, tokenizer, model, "Eval >512 (trunc)")

    print("\n=== Fine‑tuned CodeT5p‑220M (512‑tok limit) ===")
    print(f"F1 ≤512 tokens  : {f1_short:.3f}")
    print(f"F1 >512 tokens  : {f1_long:.3f}")


if __name__ == "__main__":
    main()
