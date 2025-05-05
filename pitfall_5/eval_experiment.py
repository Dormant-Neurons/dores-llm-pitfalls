#!/usr/bin/env python3
"""
Fine‑tune CodeT5p‑220M on PrimeVul with a 512‑token limit
using automatic mixed precision (AMP) on a CUDA GPU. After
training, the script evaluates on two 1 000‑sample subsets
(≤512 and >512 tokens).

Author: <you>   Date: 2025‑05‑05

Key change vs. the previous version
----------------------------------
The model is **loaded in full‑precision (FP32)** and AMP is
handled by the Trainer (`fp16=True`). This avoids the
"Attempting to unscale FP16 gradients" runtime error that
occurs when the model itself is instantiated in FP16.
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
    """Tokenise code + labels for Seq2Seq fine‑tuning."""
    codes = batch["func"]
    raw_labels = [t[0] if isinstance(t, list) else t for t in batch["target"]]

    model_inputs = tokenizer(
        codes,
        truncation=True,
        max_length=CTX_LIMIT,
    )

    label_texts = [str(int(x)) for x in raw_labels]
    model_inputs["labels"] = [tokenizer(t).input_ids for t in label_texts]
    return model_inputs


def build_eval_subsets(test_ds: Dataset):
    short_idx = [i for i, n in enumerate(test_ds["n_tok"]) if n <= CTX_LIMIT]
    long_idx = [i for i, n in enumerate(test_ds["n_tok"]) if n > CTX_LIMIT]

    rng = np.random.default_rng(SPLIT_SEED)
    short_sample = rng.choice(short_idx, size=EVAL_N, replace=False)
    long_sample = rng.choice(long_idx, size=EVAL_N, replace=False)

    return test_ds.select(short_sample), test_ds.select(long_sample)


@torch.no_grad()
def eval_f1(ds: Dataset, tokenizer, model, desc: str) -> float:
    model.eval()
    BATCH = 32
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
    print("CUDA available       :", torch.cuda.is_available())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f">> device: {device}")

    torch.manual_seed(SPLIT_SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(SPLIT_SEED)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True, use_fast=True)
    splits = make_splits(tokenizer)

    tokenised = splits.map(
        lambda ex: tokenize_func(ex, tokenizer),
        remove_columns=splits["train"].column_names,
        batched=True,
        num_proc=1,
    )

    # Load model **without** forcing FP16 weights; AMP will cast on‑the‑fly
    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_ID,
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
        fp16=(device.type == "cuda"),  # safe AMP
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
    short_ds, long_ds = build_eval_subsets(splits["test"])

    f1_short = eval_f1(short_ds, tokenizer, model, "Eval ≤512")
    f1_long = eval_f1(long_ds, tokenizer, model, "Eval >512 (trunc)")

    print("\n=== Fine‑tuned CodeT5p‑220M (512‑tok limit) ===")
    print(f"F1 ≤512 tokens  : {f1_short:.3f}")
    print(f"F1 >512 tokens  : {f1_long:.3f}")


if __name__ == "__main__":
    main()
