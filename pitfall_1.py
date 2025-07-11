"""main hook to start the pitfall 1 fine-tuning"""
# -*- coding: utf-8 -*-
# !/usr/bin/env python3

import os
import time
from datetime import timedelta
import psutil
import getpass
import datetime
import argparse

import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
from unsloth import FastLanguageModel, is_bfloat16_supported
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset, Dataset

from utils.colors import TColors
from human_eval.data import write_jsonl, read_problems

MODEL_SPECIFIER: str = "unsloth/Qwen2.5-Coder-0.5B"
DATASET_SPECIFIER: str = "bigcode/self-oss-instruct-sc2-exec-filter-50k"
MODEL_PATH: str = "./model_outputs/"
DATASET_PATH: str = "./generated_datasets/"
EOS_TOKEN: str = None  # will be overwritten by the tokenizer


def min_max_normalize(
    p_dict: dict, all_perplexities: list, new_min: int = 0, new_max: int = 1
):
    """Min-max normalize a dictionary of values to a new range [new_min, new_max]"""
    old_min = min(all_perplexities)
    old_max = max(all_perplexities)

    if old_max == old_min:
        print(
            f"{TColors.WARNING}Warning{TColors.ENDC}: Normalization does not work if the old "
            "max and min are equal. Returning the original dictionary."
        )
        return p_dict

    normalized_dict = {}

    for key, values in p_dict.items():
        temp_normalized_values = []
        # normalize each value in the list
        for p_value in values:
            temp_normalized_values.append(
                new_min
                + ((p_value - old_min) * (new_max - new_min)) / (old_max - old_min)
            )

        normalized_dict[key] = temp_normalized_values

    return normalized_dict


def preprocess_dataset(dataset: Dataset, block_size: int, tokenizer) -> Dataset:
    """Preprocess the dataset: drop out unnecessary columns and batch the dataset in
    a predetermined block_size
    """

    def tokenize_func(examples: dict) -> dict:
        """Tokenize the dataset examples"""
        # tokenize the data
        if "response" not in examples.keys():
            # if the dataset does not have a "response" column, we assume it has a "text" column
            return tokenizer(examples["text"])
        return tokenizer(examples["response"])

    # check if the dataset has the "response" column
    if "response" not in dataset.column_names:
        dataset = dataset.select_columns(["text"])
    else:
        dataset = dataset.select_columns(["response"])
    dataset = dataset.map(tokenize_func, batched=True, num_proc=8, keep_in_memory=True)

    # concatenate all data into a list
    concatenated_data = []
    for data in dataset:
        concatenated_data.append(data["input_ids"])

    total_length = len(concatenated_data)

    if total_length >= block_size:
        total_length = (total_length // block_size) * block_size

    # split the data into chunks of block_size
    chunked_data = []
    for entry in concatenated_data:
        for i in range(0, len(entry), block_size):
            if len(entry[i : i + block_size]) < block_size:
                # if the last chunk is smaller than block_size, we skip it
                continue
            chunked_data.append(entry[i : i + block_size])

    # now the list contains tokenized chunks of the dataset, each of size block_size
    # we decode it not back into text
    chunked_data = [tokenizer.decode(chunk) for chunk in chunked_data]

    # convert the chunked data into a Dataset
    return Dataset.from_dict({"text": chunked_data})


def format_prompt(examples: dict) -> dict:
    """format the dataset inputs for the trainer"""

    completion_data = examples["response"]  # we only want the code completion part
    # user_inputs = examples["instruction"]
    # responses = examples["response"]

    prompts = []

    # for user_input, response in zip(user_inputs, responses):
    #     prompts.append(
    #         f"""<|imstart|>system
    #         You are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>
    #         <|im_start|>user
    #         {user_input}<|im_end|>
    #         <|im_start|>assistant
    #         {response}<|im_end|>"""
    #         + EOS_TOKEN
    #     )

    for data in completion_data:
        prompts.append(data + EOS_TOKEN)

    return {"text": prompts}


def make_splits(dataset: Dataset) -> Dataset:
    """Splits the dataset into training and validation sets"""
    # split the dataset into training and validation sets
    train_size = int(0.9 * len(dataset))
    train_dataset = dataset.select(range(train_size))
    val_dataset = dataset.select(range(train_size, len(dataset)))

    return train_dataset, val_dataset


def main(
    device: str = "cpu",
    training_epochs: int = 5,
    dataset_batch_size: int = 10,
    training_batch_size: int = 8,
    skip_training: bool = False,
    num_generations: int = 5,
    block_size: int = 64,
    histogram_only: bool = False,
    human_eval_only: bool = False,
    data_path: str = "",
    use_original_dataset: bool = False,
    model_specifier: str = "",
) -> None:
    """
    Main function to start the pitfall 1 fine-tuning

    Args:
        device (str): device to run the computations on (cpu, cuda, mps)
        training_epochs (int): number of training epochs to run
        dataset_batch_size (int): batch size for the dataset
        training_batch_size (int): batch size for the training/eval
        skip_training (bool): if True, skip the training and only evaluate the models
        num_generations (int): number of generations to run (default: 5)
        block_size (int): size of the blocks to split the dataset into (default: 64)
        histogram_only (bool): if True, only generate the histogram and skip the rest
        human_eval_only (bool): if True, only generate human eval samples and skip the rest
        data_path (str): path to save the generated datasets and models
        use_original_dataset (bool): if True, use the original datainstead of the synthetic dataset
        model_specifier (str): model specifier to use for the training

    Returns:
        None
    """
    start_time = time.time()

    # ──────────────────────────── set devices and print informations ─────────────────────────
    # set the devices correctly
    if "cpu" in device:
        device = torch.device("cpu", 0)
    elif "cuda" in device and torch.cuda.is_available():
        if "cuda" not in device.split(":")[-1]:
            device = torch.device("cuda", int(device.split(":")[-1]))
        else:
            device = torch.device("cuda", 0)
    elif "mps" in device and torch.backends.mps.is_available():
        if "mps" not in device.split(":")[-1]:
            device = torch.device("mps", int(device.split(":")[-1]))
        else:
            device = torch.device("mps", 0)
    else:
        print(
            f"{TColors.WARNING}Warning{TColors.ENDC}: Device {TColors.OKCYAN}{device} "
            f"{TColors.ENDC}is not available. Setting device to CPU instead."
        )
        device = torch.device("cpu", 0)

    # set data paths
    if data_path != "":
        global DATASET_PATH
        DATASET_PATH = os.path.join(data_path, "generated_datasets/")
        global MODEL_PATH
        MODEL_PATH = os.path.join(data_path, "model_outputs/")
        # create the directories if they do not exist
        os.makedirs(DATASET_PATH, exist_ok=True)
        os.makedirs(MODEL_PATH, exist_ok=True)

    # set the model specifier
    if model_specifier != "":
        global MODEL_SPECIFIER
        MODEL_SPECIFIER = model_specifier
    specifier_name = MODEL_SPECIFIER.split("/")[-1]

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
    if (device == "cuda" or torch.device("cuda", 0)) and torch.cuda.is_available():
        print(
            f"## {TColors.OKBLUE}{TColors.BOLD}GPU Memory{TColors.ENDC}: "
            f"{torch.cuda.mem_get_info()[1] // 1024**2} MB"
        )
    elif (
        device == "mps" or torch.device("mps", 0)
    ) and torch.backends.mps.is_available():
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
        f"## {TColors.OKBLUE}{TColors.BOLD}Model Specifier{TColors.ENDC}: {MODEL_SPECIFIER}"
    )
    print(
        f"## {TColors.OKBLUE}{TColors.BOLD}Dataset Specifier{TColors.ENDC}: {DATASET_SPECIFIER}"
    )
    print(
        f"## {TColors.OKBLUE}{TColors.BOLD}Use Original Data{TColors.ENDC}: {use_original_dataset}"
    )
    print(
        f"## {TColors.OKBLUE}{TColors.BOLD}Number of Generations{TColors.ENDC}: {num_generations}"
    )
    print(f"## {TColors.OKBLUE}{TColors.BOLD}Block size{TColors.ENDC}: {block_size}")
    print(
        f"## {TColors.OKBLUE}{TColors.BOLD}Training Steps{TColors.ENDC}: {training_epochs}"
    )
    print(
        f"## {TColors.OKBLUE}{TColors.BOLD}Dataset Batch Size{TColors.ENDC}: {dataset_batch_size}"
    )
    print(
        f"## {TColors.OKBLUE}{TColors.BOLD}Training Batch Size{TColors.ENDC}: {training_batch_size}"
    )
    print(
        f"## {TColors.OKBLUE}{TColors.BOLD}Skip Training{TColors.ENDC}: {skip_training}"
    )
    print(
        f"## {TColors.OKBLUE}{TColors.BOLD}Model Saving Path{TColors.ENDC}: {MODEL_PATH}"
    )
    print(
        f"## {TColors.OKBLUE}{TColors.BOLD}Gerenated Datasets Path{TColors.ENDC}: {DATASET_PATH}"
    )
    print("#" * os.get_terminal_size().columns + "\n")

    # load the tokenizer to count to tokens of the dataset
    _, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_SPECIFIER,
        max_seq_length=block_size,
        dtype=None,
        load_in_4bit=True,
    )
    global EOS_TOKEN
    EOS_TOKEN = tokenizer.eos_token

    # load the dataset
    original_dataset = load_dataset(DATASET_SPECIFIER, split="train")
    original_dataset = original_dataset.select_columns(["response"])

    # print some information about the dataset
    token_counts = []
    for data in tqdm(original_dataset, desc="Calculating token counts"):
        inputs = tokenizer(
            data["response"],
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        # count the tokens
        token_count = inputs["input_ids"].shape[1]
        token_counts.append(token_count)

    print(f"Max token count: {max(token_counts)}")
    print(f"Avg token count: {sum(token_counts) / len(token_counts)}")
    print(f"Min token count: {min(token_counts)}")
    print(f"Original dataset length: {len(original_dataset)}\n")
    original_dataset = original_dataset.map(format_prompt, batched=True)
    original_dataset.save_to_disk(DATASET_PATH + f"original_dataset_bs{block_size}")

    assert block_size > min(token_counts), (
        f"{TColors.FAIL}Block size must be larger than "
        f"the minimum token count of the dataset.{TColors.ENDC}"
    )

    # preprocess the dataset
    chunked_dataset = preprocess_dataset(original_dataset, block_size, tokenizer)
    chunked_dataset.save_to_disk(
        DATASET_PATH + f"chunked_dataset_bs{block_size}_{specifier_name}"
    )
    # the dataloader is later used for the generation of the new dataset
    chunked_dataloader = DataLoader(
        chunked_dataset.with_format("torch"),
        batch_size=dataset_batch_size,
    )

    if not skip_training:
        # ───────────────────────── start the actual finetuning ──────────────────────────────
        # iterte over two loops: first the model training and then the dataset generation
        # the model is trained for N times and after each training the dataset
        # is generated from the new model
        for i in range(num_generations):
            # load the model
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=MODEL_SPECIFIER,  # if i == 0 else f"{MODEL_PATH}model_{i-1}_fp16",
                max_seq_length=block_size,
                dtype=None,
                load_in_4bit=True,
            )

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
            if i > 0 and not use_original_dataset:
                # if the first training iteration is done, load the generated dataset from the disk
                dataset = Dataset.load_from_disk(
                    DATASET_PATH + f"generated_dataset_{i - 1}_bs{block_size}_{specifier_name}"
                )
            else:
                dataset = chunked_dataset

            # for the first model the original dataset is used, then the generated dataset
            # is used for the next models
            dataset_train, dataset_val = make_splits(dataset)

            # for some stats
            gpu_stats = torch.cuda.get_device_properties(0)
            start_gpu_memory = round(
                torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3
            )
            max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)

            # create a trainer to train the model
            trainer = SFTTrainer(
                model=model,
                tokenizer=tokenizer,
                train_dataset=dataset_train,
                eval_dataset=dataset_val,
                # formatting_func=format_prompt,
                dataset_text_field="text",
                max_seq_length=block_size,
                dataset_num_proc=8,
                packing=True,  # Can make training 5x faster for short sequences.
                args=TrainingArguments(
                    gradient_accumulation_steps=4,
                    warmup_steps=5,
                    num_train_epochs=training_epochs,
                    per_device_train_batch_size=training_batch_size,
                    per_device_eval_batch_size=training_batch_size,
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
            metrics = trainer.evaluate()
            print(
                f"## {TColors.OKBLUE}{TColors.BOLD}Loss: {TColors.ENDC}{metrics["eval_loss"]:.4f}"
            )

            # print some fancy stats
            used_memory = round(
                torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3
            )
            used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
            used_percentage = round(used_memory / max_memory * 100, 3)
            lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
            print(
                f"{trainer_stats.metrics["train_runtime"]} seconds used for training."
            )
            print(
                f"{round(trainer_stats.metrics["train_runtime"] / 60, 2)} min. used for training."
            )
            print(f"Peak reserved memory = {used_memory} GB.")
            print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
            print(f"Peak reserved memory % of max memory = {used_percentage} %.")
            print(
                f"Peak reserved memory for training % of max memory = {lora_percentage} %."
            )

            # save the model
            trainer.model.save_pretrained(
                f"{MODEL_PATH}model_{i}_bs{block_size}_{specifier_name}",
                safe_serialization=True,
                save_adapter=True,
                save_config=True,
            )
            trainer.tokenizer.save_pretrained(
                f"{MODEL_PATH}model_{i}_bs{block_size}_{specifier_name}"
            )

            del trainer
            del model
            del tokenizer
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            # use the model to generate the new dataset
            # for this the model is loaded again with the quantized weights
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=f"{MODEL_PATH}model_{i}_bs{block_size}_{specifier_name}",
                max_seq_length=block_size,
                dtype=None,
                load_in_4bit=True,
            )
            FastLanguageModel.for_inference(model)
            # ────────────────────────────── generate the new datasets ────────────────────────────
            if not use_original_dataset:
                print(
                    f"## {TColors.OKBLUE}{TColors.BOLD}Generate Dataset {i}{TColors.ENDC}"
                )
                new_data = []
                for _, data_batch in tqdm(
                    enumerate(chunked_dataloader), total=len(chunked_dataloader)
                ):
                    # tokenize the data batch
                    inputs = list(data_batch["text"])

                    # generate the answer using the model
                    inputs = tokenizer(
                        inputs,
                        padding=True,
                        truncation=True,
                        return_tensors="pt",
                    ).to("cuda")

                    generated_answers = model.generate(
                        **inputs,
                        # num_beams=5,
                        repetition_penalty=3.0,
                        min_new_tokens=block_size,
                        max_new_tokens=block_size,
                        use_cache=True,
                    )

                    # only keep and decode the last 64 tokens of the generated answer
                    generated_answers = generated_answers[:, 64:]
                    generated_answers = tokenizer.batch_decode(generated_answers)

                    new_data += list(generated_answers)

                # save the new dataset to disk
                new_dataset = Dataset.from_dict({"text": new_data})
                new_dataset = preprocess_dataset(new_dataset, block_size, tokenizer)
                new_dataset.save_to_disk(
                    DATASET_PATH + f"generated_dataset_{i}_bs{block_size}_{specifier_name}"
                )

    # ────────────────── evaluate the models' perplexity and other metrics ─────────────────────────
    # iterate over every model and the generated dataset and calculate the perplexity
    # for the perplexity, every datapoint i.e., the generated answer for every question
    # is evaluated to get the probability for a given perplexity over the whole dataset
    if not human_eval_only:
        if not histogram_only:
            print(
                f"## {TColors.OKBLUE}{TColors.BOLD}Calculate Perplexity{TColors.ENDC}"
            )
            perplexity_dict = {}
            all_perplexities = []


            for i in range(num_generations):
                # load the model
                model_name = (
                    f"{MODEL_PATH}model_{i}_bs{block_size}_{specifier_name}"
                    if i > 0 and not use_original_dataset
                    else MODEL_SPECIFIER
                )
                model, tokenizer = FastLanguageModel.from_pretrained(
                    model_name=model_name,
                    max_seq_length=block_size,
                    dtype=None,
                    load_in_4bit=True,
                )
                FastLanguageModel.for_inference(model)

                # load the dataset
                if i == 0 or use_original_dataset:
                    # for the first generation, use the original dataset
                    ppl_dataset = Dataset.load_from_disk(
                        DATASET_PATH + f"/chunked_dataset_bs{block_size}_{specifier_name}"
                    )
                else:
                    ppl_dataset = Dataset.load_from_disk(
                        DATASET_PATH
                        + f"/generated_dataset_{i - 1}_bs{block_size}_{specifier_name}"
                    )

                ppl_dataloader = DataLoader(
                    ppl_dataset.with_format("torch"),
                    batch_size=1,  # batch size for the perplexity calculation
                )

                # add new entry to the dict
                perplexity_dict[f"Generation {i}"] = []

                # calculate the perplexity for every datapoint in the dataset (eval)
                for data_batch in tqdm(
                    ppl_dataloader, desc=f"Calculating perplexity for Generation {i}"
                ):
                    inputs = tokenizer(
                        data_batch["text"],
                        padding=True,
                        truncation=True,
                        return_tensors="pt",
                    ).to("cuda")

                    # calculate the perplexity for every datapoint in the dataset
                    with torch.no_grad():
                        outputs = model(**inputs, labels=inputs["input_ids"])
                        loss = outputs.loss
                        perplexity = torch.exp(loss)
                        perplexity_dict[f"Generation {i}"].append(perplexity.item())

            # get all single values from the dict and flatten them into a list
            all_perplexities = [
                perplexity
                for values in perplexity_dict.values()
                for perplexity in values
            ]
            # # normalize the perplexity values
            # perplexity_dict = min_max_normalize(
            #     perplexity_dict, all_perplexities, new_min=0, new_max=100
            # )
            # # update the all_perplexities list with the normalized values
            # all_perplexities = [
            #     perplexity for values in perplexity_dict.values() for perplexity in values
            # ]

            # save the perplexity dict to a file
            torch.save(
                perplexity_dict,
                DATASET_PATH + f"perplexity_dict_bs{block_size}_{specifier_name}.pt",
            )  # save the dict to a file
            print(
                f"## {TColors.OKBLUE}{TColors.BOLD}Saved the perplexity dict under: "
                f"{TColors.HEADER}{DATASET_PATH}perplexity_dict_bs{block_size}_{specifier_name}" \
                f".pt{TColors.ENDC}"
            )
            # save the all_perplexities list to a file
            torch.save(
                all_perplexities,
                DATASET_PATH + f"all_perplexities_bs{block_size}_{specifier_name}.pt",
            )  # save the list to a file
            print(
                f"## {TColors.OKBLUE}{TColors.BOLD}Saved the all_perplexities list under: "
                f"{TColors.HEADER}{DATASET_PATH}all_perplexities_bs{block_size}_{specifier_name}" \
                f".pt{TColors.ENDC}"
            )
        else:
            # load the perplexity dict and all_perplexities list from the files
            perplexity_dict = torch.load(
                DATASET_PATH + f"perplexity_dict_bs{block_size}_{specifier_name}.pt"
            )
            all_perplexities = torch.load(
                DATASET_PATH + f"all_perplexities_bs{block_size}_{specifier_name}.pt"
            )

        # ────────────────── plot the perplexity histogram ─────────────────────────
        print(
            f"## {TColors.OKBLUE}{TColors.BOLD}Plotting Perplexity Histogram{TColors.ENDC}"
        )

        min_perplexity = min(all_perplexities)
        max_perplexity = max(all_perplexities)
        bins = torch.linspace(min_perplexity, max_perplexity, len(all_perplexities) + 1)

        plt.figure(figsize=(14, 8))
        # plot the perplexity for every model as a histogram
        for name, perplexities in perplexity_dict.items():
            plt.hist(perplexities, bins=bins, density=True, alpha=0.35, label=name)

        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("Perplexity")
        plt.ylabel("Probability")
        plt.title(
            f"Perplexity of generated datapoints for blocksize of {block_size}_{specifier_name}"
        )
        plt.legend(loc="upper right")
        plt.tight_layout()
        plt.savefig(f"perplexity_histogram_bs{block_size}_{specifier_name}.png")

        print(
            f"## {TColors.OKBLUE}{TColors.BOLD}Saved the histogram under: "
            f"{TColors.HEADER}./perplexity_histogram_bs{block_size}_{specifier_name}" \
            f".png{TColors.ENDC}"
        )

    # # ────────────────── test the models outputs ─────────────────────────
    # this generates samples based on the human eval dataset from OpenAI
    # the samples are then evaluated using the evaluate library
    # https://github.com/openai/human-eval

    problems = read_problems()

    samples = []
    for model_idx in range(num_generations):
        # load the model
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=f"{MODEL_PATH}model_{model_idx}_bs{block_size}_{specifier_name}",
            max_seq_length=2048,
            dtype=None,
            load_in_4bit=True,
        )
        FastLanguageModel.for_inference(model)
        print(
            f"## {TColors.OKBLUE}{TColors.BOLD}Generating samples for model {model_idx} "\
            f"{TColors.ENDC}"
        )
        for task_id in tqdm(
            problems,
            desc="Generating samples from the models",
            total=len(problems)
        ):
            # create x samples for each problem (how often to generate the answer) -> pass@k
            for _ in range(5):
                # generate the answer for the test question
                inputs = tokenizer(
                    problems[task_id]["prompt"],
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                ).to("cuda")

                generated_answer = model.generate(
                    **inputs,
                    repetition_penalty=3.0,
                    max_new_tokens=2048,
                    use_cache=True,
                )

                # decode the generated answer
                generated_answer = tokenizer.batch_decode(
                    generated_answer, skip_special_tokens=True
                )[0]
                # remove the prompt from the generated answer
                generated_answer = generated_answer[len(problems[task_id]["prompt"]):]
                # add to the list of samples
                samples.append({"task_id": task_id, "completion": generated_answer})

        write_jsonl(
            f"{DATASET_PATH}eval_samples_gen{model_idx}_bs{block_size}_{specifier_name}.jsonl",
            samples,
        )

    # ────────────────── print the elapsed time ─────────────────────────
    # End the timer
    end_time = time.time()

    # Calculate elapsed time
    elapsed_time = end_time - start_time
    delta = timedelta(seconds=int(elapsed_time))

    days = delta.days
    hours, remainder = divmod(delta.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    print(f"## {TColors.OKBLUE}{TColors.BOLD}Execution time: ")
    if days:
        print(f"{TColors.HEADER}{days} days, {hours:02}:{minutes:02}:{seconds:02}")
    else:
        print(f"{TColors.HEADER}{hours:02}:{minutes:02}:{seconds:02}")
    print(f"{TColors.ENDC}")


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
        "--training_epochs",
        "-te",
        type=int,
        default=5,
        help="specifies the number of training epochs to run",
    )
    parser.add_argument(
        "--dataset_batch_size",
        "-dbs",
        type=int,
        default=100,
        help="specifies the batch size for the dataset",
    )
    parser.add_argument(
        "--training_batch_size",
        "-tbs",
        type=int,
        default=16,
        help="specifies the batch size for the training/eval",
    )
    parser.add_argument(
        "--skip_training",
        "-st",
        action="store_true",
        help="if set, skip the training and only evaluate the models",
    )
    parser.add_argument(
        "--num_generations",
        "-ng",
        type=int,
        default=10,
        help="specifies the number of generations to run (default: 5)",
    )
    parser.add_argument(
        "--block_size",
        "-bs",
        type=int,
        default=64,
        help="specifies the size of the blocks to split the dataset into (default: 64)",
    )
    parser.add_argument(
        "--histogram_only",
        "-ho",
        action="store_true",
        help="if set, only generate the histogram and skip the rest",
    )
    parser.add_argument(
        "--data_path",
        "-dp",
        type=str,
        default="",
        help="path to save the generated datasets and models (default: current directory)",
    )
    parser.add_argument(
        "--use_original_dataset",
        "-uod",
        action="store_true",
        help="if set, use the original dataset instead of the synthetic dataset",
    )
    parser.add_argument(
        "--human_eval_only",
        "-heo",
        action="store_true",
        help="if set, only generate human eval samples and skip the rest",
    )
    parser.add_argument(
        "--model_specifier",
        "-ms",
        type=str,
        default="",
        help="model specifier to use for the training (default: unsloth/Qwen2.5-Coder-0.5B)",
    )
    args = parser.parse_args()
    main(**vars(args))
