"""main hook to start the pitfall 1 fine-tuning"""
# -*- coding: utf-8 -*-
# !/usr/bin/env python3

import os
import psutil
import getpass
import datetime
import argparse

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from unsloth import FastLanguageModel, is_bfloat16_supported
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset, Dataset

from utils.colors import TColors

MODEL_SPECIFIER: str = "unsloth/Qwen2.5-Coder-0.5B-Instruct"
DATASET_SPECIFIER: str = "bigcode/self-oss-instruct-sc2-exec-filter-50k"
MAX_SEQ_LENGTH: int = 4096
NUM_TRAINING: int = 5
MODEL_PATH: str = "./model_outputs/"
DATASET_PATH: str = "./generated_datasets/"
EOS_TOKEN: str = None # will be overwritten by the tokenizer


def format_prompt(examples) -> dict:
    """format the dataset inputs for the trainer"""

    user_inputs = examples["prompt"]
    responses = examples["response"]

    prompts = []

    for user_input, response in zip(user_inputs, responses):
        prompts.append(
            f""""You are Qwen, created by Alibaba. You are a helpful assistant.

            ### Instruction:
            {user_input}

            ### Response:
            {response}""" + EOS_TOKEN
        )

    return {"text": prompts}


def make_splits(dataset) -> Dataset:
    """Splits the dataset into training and validation sets"""

    # shuffle the dataset
    dataset = dataset.shuffle(seed=42)

    # split the dataset into training and validation sets
    train_size = int(0.8 * len(dataset))
    train_dataset = dataset.select(range(train_size))
    val_dataset = dataset.select(range(train_size, len(dataset)))

    return train_dataset, val_dataset


def main(device: str = "cpu", training_steps: int = 100, dataset_batch_size: int = 10) -> None:
    """Main function to start the pitfall 1 fine-tuning"""

    # ──────────────────────────── set devices and print informations ─────────────────────────
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
        f"## {TColors.OKBLUE}{TColors.BOLD}Training Steps{TColors.ENDC}: {training_steps}"
    )
    print(
        f"## {TColors.OKBLUE}{TColors.BOLD}Dataset Batch Size{TColors.ENDC}: {dataset_batch_size}"
    )
    print(
        f"## {TColors.OKBLUE}{TColors.BOLD}Model Saving Path{TColors.ENDC}: {MODEL_PATH}"
    )
    print(
        f"## {TColors.OKBLUE}{TColors.BOLD}Gerenated Datasets Path{TColors.ENDC}: {DATASET_PATH}"
    )
    print("#" * os.get_terminal_size().columns + "\n")

    # ───────────────────────── start the actual finetuning ──────────────────────────────
    # iterte over two loops: first the model training and then the dataset generation
    # the model is trained for N times and after each training the dataset
    # is generated from the new model

    # load the dataset
    original_dataset = load_dataset(
        DATASET_SPECIFIER,
        split="train"
    )
    original_dataset = original_dataset.select_columns(["prompt", "response"])
    original_dataset.save_to_disk(DATASET_PATH + "original_dataset")
    # the dataloader is later used for the generation of the new dataset
    original_dataloader = DataLoader(
        original_dataset.with_format("torch"),
        batch_size=10,
    )
    print(f"Original dataset length: {len(original_dataset)}")

    for i in range(NUM_TRAINING):
        # load the model
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=MODEL_SPECIFIER if i == 0 else f"{MODEL_PATH}/model_{i-1}_fp16",
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
        if i > 0:
            # if the first training iteration is done, load the generated dataset from the disk
            dataset = Dataset.load_from_disk(DATASET_PATH+f"generated_dataset_{i-1}")
        else:
            dataset = original_dataset

        # for the first model the original dataset is used, then the generated dataset
        # is used for the next models
        dataset_train, dataset_val = make_splits(dataset)
        dataset_train = dataset_train.map(format_prompt, batched=True)
        dataset_val = dataset_val.map(format_prompt, batched=True)

        # for some stats
        gpu_stats = torch.cuda.get_device_properties(0)
        start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
        max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)

        # create a trainer to train the model
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset_train,
            eval_dataset=dataset_val,
            # formatting_func=format_prompt,
            dataset_text_field="text",
            max_seq_length=MAX_SEQ_LENGTH,
            dataset_num_proc=8,
            packing=True,  # Can make training 5x faster for short sequences.
            args=TrainingArguments(
                per_device_train_batch_size=2,
                gradient_accumulation_steps=4,
                warmup_steps=5,
                # num_train_epochs = 1, # Set this for 1 full training run.
                max_steps=training_steps,
                learning_rate=2e-4,
                fp16=not is_bfloat16_supported(),
                bf16=is_bfloat16_supported(),
                logging_steps=1,
                optim="adamw_8bit",
                weight_decay=0.01,
                lr_scheduler_type="linear",
                seed=1337,
                output_dir="outputs",
                report_to="none",
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
        trainer.model.save_pretrained(
            f"{MODEL_PATH}/model_{i}_fp16",
            safe_serialization=True,
            save_adapter=True,
            save_config=True,
        )
        trainer.tokenizer.save_pretrained(f"{MODEL_PATH}/model_{i}_q4_k_m")

        # use the model to generate the new dataset
        FastLanguageModel.for_inference(model)

        # ────────────────────────────── generate the new datasets ────────────────────────────
        print(f"## {TColors.OKBLUE}{TColors.BOLD}Generate Dataset{TColors.ENDC}")
        new_data = []
        for gen_iter, data_batch in tqdm(
            enumerate(original_dataloader), total=len(original_dataloader)
        ):
            # generate a dataset with the same length as the original dataset
            if gen_iter >= len(original_dataloader):
                break

            # tokenize the data batch
            inputs = []
            for data in data_batch["prompt"]:
                inputs.append(
                    f""""You are Qwen, created by Alibaba. You are a helpful assistant.

                    ### Instruction:
                    {data}"""
                )
            # generate the answer using the model
            inputs = tokenizer(
                inputs,
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).to("cuda")

            generated_answers = model.generate(
                **inputs,
                max_new_tokens=MAX_SEQ_LENGTH,
                use_cache=True,
                temperature=0.01,

            )
            generated_answers = tokenizer.batch_decode(
                generated_answers, skip_special_tokens=True
            )[0]
            print("Generated Answers: ", generated_answers)

            # add the generated answer to the dataset
            for generated_answer, data in zip(generated_answers, data_batch):
                generated_answer = generated_answer.split("### Instruction:")[-1].strip()
                question = data_batch["prompt"]
                # add the generated answer to the dataset
                if len(generated_answer) == 0:
                    continue
                if len(question) == 0:
                    continue
                print("Question: ", question)
                print("Generated Answer: ", generated_answer)
                # add the generated answer to the dataset
                new_data.append(
                    {
                        "prompt": question,
                        "response": generated_answer,
                    }
                )
        # save the new dataset to disk
        new_dataset = Dataset.from_list(new_data)
        new_dataset.save_to_disk(DATASET_PATH + f"generated_dataset_{i}")

    # ────────────────── evaluate the models' perplexity and other metrics ─────────────────────────
    # TODO: implement the evaluation of the models' perplexity and other metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="pitfall_1")
    parser.add_argument(
        "--device",
        "-dx",
        type=str,
        default="cpu",
        help="specifies the device to run the computations on (cpu, cuda, mps)",
    )
    parser.add_argument(
        "--training_steps",
        "-ts",
        type=int,
        default=100,
        help="specifies the number of training steps to run",
    )
    parser.add_argument(
        "--dataset_batch_size",
        "-dbs",
        type=int,
        default=10,
        help="specifies the batch size for the dataset",
    )
    args = parser.parse_args()
    main(**vars(args))
