"""main hook to start the pitfall 1 fine-tuning"""
# -*- coding: utf-8 -*-
# !/usr/bin/env python3

import os
import psutil
import getpass
import datetime
import argparse

import torch
from unsloth import FastLanguageModel, is_bfloat16_supported
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset

from utils.colors import TColors

MODEL_SPECIFIER: str = "unsloth/Qwen2.5-Coder-0.5B-Instruct"
MAX_SEQ_LENGTH: int = 4096
NUM_TRAINING: int = 5
MODEL_PATH: str = "./model_outputs/"
DATASET_PATH: str = "./generated_datasets/"
EOS_TOKEN: str = None # will be overwritten by the tokenizer


def format_prompt(examples) -> dict:
    """format the dataset inputs for the trainer"""
    return {
        "messages": [
            {
                "role": "system",
                "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
            },
            {"role": "user", "content": f"{examples["prompt"]}"},
            {"role": "assistant", "content": f"{examples["response"]}"},
        ]
    }


def main(device: str = "cpu") -> None:
    """Main function to start the pitfall 1 fine-tuning"""

    # set the devices correctly
    if device == "cpu":
        device = torch.device("cpu")
    elif device == "cuda" and torch.cuda.is_available():
        device = torch.device(device)
    elif device == "mps" and torch.backends.mps.is_available():
        device = torch.device(device)
    else:
        print(
            f"{TColors.WARNING}Warning{TColors.ENDC}: Device {TColors.OKCYAN}{device} "
            f"{TColors.ENDC}is not available. Setting device to CPU instead."
        )
        device = torch.device("cpu")

    # have a nice system status print
    print(
        "\n"
        + f"## {TColors.BOLD}{TColors.HEADER}{TColors.UNDERLINE}System Information"
        + f"{TColors.ENDC} "
        + "#" * (os.get_terminal_size().columns - 23)
    )
    print(
        f"## {TColors.OKBLUE}{TColors.BOLD}Date{TColors.ENDC}: "
        + str(datetime.datetime.now().strftime("%A, %d. %B %Y %I:%M%p"))
    )
    print(
        f"## {TColors.OKBLUE}{TColors.BOLD}System{TColors.ENDC}: "
        f"{torch.get_num_threads()} CPU cores with {os.cpu_count()} threads and "
        f"{torch.cuda.device_count()} GPUs on user: {getpass.getuser()}"
    )
    print(f"## {TColors.OKBLUE}{TColors.BOLD}Device{TColors.ENDC}: {device}")
    if (device == "cuda" or torch.device("cuda")) and torch.cuda.is_available():
        print(
            f"## {TColors.OKBLUE}{TColors.BOLD}GPU Memory{TColors.ENDC}: "
            f"{torch.cuda.mem_get_info()[1] // 1024**2} MB"
        )
    elif (device == "mps" or torch.device("mps")) and torch.backends.mps.is_available():
        print(
            f"## {TColors.OKBLUE}{TColors.BOLD}Shared Memory{TColors.ENDC}: "
            f"{psutil.virtual_memory()[0] // 1024**2} MB"
        )
    else:
        print(
            f"## {TColors.OKBLUE}{TColors.BOLD}CPU Memory{TColors.ENDC}: "
            f"{psutil.virtual_memory()[0] // 1024**2} MB"
        )
    print(
        f"## {TColors.BOLD}{TColors.HEADER}{TColors.UNDERLINE}Parameters"
        + f"{TColors.ENDC} "
        + "#" * (os.get_terminal_size().columns - 14)
    )
    print(
        f"## {TColors.OKBLUE}{TColors.BOLD}Number of Training Loops{TColors.ENDC}: {NUM_TRAINING}"
    )
    print(
        f"## {TColors.OKBLUE}{TColors.BOLD}MAX_SEQ_LENGTH{TColors.ENDC}: {MAX_SEQ_LENGTH}"
    )
    print(
        f"## {TColors.OKBLUE}{TColors.BOLD}Model Saving Path{TColors.ENDC}: {MODEL_PATH}"
    )
    print(
        f"## {TColors.OKBLUE}{TColors.BOLD}Gerenated Datasets Path{TColors.ENDC}: {DATASET_PATH}"
    )
    print("#" * os.get_terminal_size().columns + "\n")


    # iterte over two loops: first the model training and then the dataset generation
    # the model is trained for N times and after each training the dataset
    # is generated from the new model

    for i in range(NUM_TRAINING):
        # load the model
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=MODEL_SPECIFIER,
            max_seq_length=MAX_SEQ_LENGTH,
            dtype=None,
            load_in_4bit=True,
        )
        global EOS_TOKEN
        EOS_TOKEN = tokenizer.eos_token

        # add LoRA adapters
        model = FastLanguageModel.get_peft_model(
            model,
            r=16,  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            lora_alpha=16,
            lora_dropout=0,  # Supports any, but = 0 is optimized
            bias="none",  # Supports any, but = "none" is optimized
            use_gradient_checkpointing="unsloth",  # True or "unsloth" for very long context
            random_state=1337,
            use_rslora=False,  # We support rank stabilized LoRA
            loftq_config=None,  # And LoftQ
        )

        # load the dataset
        # for the first model the original dataset is used, then the generated dataset
        # is used for the next models
        dataset = load_dataset("bigcode/self-oss-instruct-sc2-exec-filter-50k", split="train")
        dataset.save_to_disk(DATASET_PATH)
        # TODO: implement the dataset loading
        dataset = None

        # for some stats
        gpu_stats = torch.cuda.get_device_properties(0)
        start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
        max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)

        # create a trainer to train the model
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset,
            formatting_func=format_prompt,
            #dataset_text_field="text",
            max_seq_length=MAX_SEQ_LENGTH,
            dataset_num_proc=2,
            packing=True,  # Can make training 5x faster for short sequences.
            args=TrainingArguments(
                per_device_train_batch_size=2,
                gradient_accumulation_steps=4,
                warmup_steps=5,
                # num_train_epochs = 1, # Set this for 1 full training run.
                max_steps=60,
                learning_rate=2e-4,
                fp16=not is_bfloat16_supported(),
                bf16=is_bfloat16_supported(),
                logging_steps=1,
                optim="adamw_8bit",
                weight_decay=0.01,
                lr_scheduler_type="linear",
                seed=1337,
                output_dir="outputs",
                report_to="none",  # Use this for WandB etc
            ),
        )

        # train the model
        trainer_stats = trainer.train()

        # print some fancy stats
        used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
        used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
        used_percentage = round(used_memory / max_memory * 100, 3)
        lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
        print(f"{trainer_stats.metrics["train_runtime"]} seconds used for training.")
        print(
            f"{round(trainer_stats.metrics["train_runtime"]/60, 2)} minutes used for training."
        )
        print(f"Peak reserved memory = {used_memory} GB.")
        print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
        print(f"Peak reserved memory % of max memory = {used_percentage} %.")
        print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")

        # save the model (both full precision and quantized)
        model.save_pretrained_gguf(
            f"{MODEL_PATH}/model_{i}_q4_k_m",
            tokenizer,
            quantization_method="q4_k_m"
        )
        trainer.model.save_pretrained(
            f"{MODEL_PATH}/model_{i}_fp16",
            safe_serialization=True,
            save_adapter=True,
            save_config=True,
        )
        trainer.tokenizer.save_pretrained(f"{MODEL_PATH}/model_{i}_q4_k_m")

        # use the model to generate the new dataset
        FastLanguageModel.for_inference(model)

        # TODO: Implement the dataset generation

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="pitfall_1")
    parser.add_argument(
        "--device",
        "-dx",
        type=str,
        default="cpu",
        help="specifies the device to run the computations on (cpu, cuda, mps)",
    )
    args = parser.parse_args()
    main(**vars(args))
