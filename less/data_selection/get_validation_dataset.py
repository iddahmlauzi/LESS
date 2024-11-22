import json
import os
from typing import List, Tuple

import pandas as pd
import torch
from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader
from transformers import DataCollatorForSeq2Seq, PreTrainedTokenizerBase

# llama-chat model's instruction format
B_INST, E_INST = "[INST]", "[/INST]"


def tokenize(tokenizer: PreTrainedTokenizerBase,
             query: str,
             completion: str,
             max_length: int,
             print_ex: bool = False) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
    """
    Formats a chat conversation into input tensors for a transformer model.

    Args:
        tokenizer (PreTrainedTokenizerBase): The tokenizer used to encode the input.
        query (str): The question part of the chat conversation.
        completion (str): The answer part of the chat conversation.
        max_length (int): The maximum length of the input tensors.
        print_ex (bool, optional): Whether to print the example. Defaults to False.

    Returns:
        tuple: A tuple containing the full input IDs, labels, and attention mask tensors.
    """
    full_prompt = query + completion

    if print_ex:
        print("******** Example starts ********")
        print(full_prompt)
        print("******** Example ends ********")

    prompt_input_ids = torch.tensor(
        tokenizer.encode(query, max_length=max_length))
    full_input_ids = torch.tensor(
        tokenizer.encode(full_prompt, max_length=max_length))
    labels = torch.tensor(tokenizer.encode(full_prompt, max_length=max_length))
    labels[:len(prompt_input_ids)] = -100
    attention_mask = [1] * len(full_input_ids)

    return full_input_ids, labels, attention_mask

def get_humaneval_dataset(data_dir: str,
                          tokenizer: PreTrainedTokenizerBase,
                          max_length: int,
                          **kwargs):
    """
    Get the HumanEval dataset formatted for code generation tasks.

    Each example consists of:
      - Query: the 'prompt' (function signature and docstring).
      - Completion: the 'canonical_solution' (code solution).

    Args:
        data_dir (str): The main data directory (unused as the dataset is loaded directly).
        tokenizer (PreTrainedTokenizerBase): The tokenizer used to tokenize the input text.
        max_length (int): The maximum length of the input sequence.

    Returns:
        Dataset: The tokenized HumanEval dataset containing input_ids, attention_mask, and labels.
    """

    # Load the HumanEval dataset
    humaneval_dataset = load_dataset("openai/openai_humaneval", split="test")
    
    # Select the first 83 examples
    humaneval_dataset = humaneval_dataset.select(range(83))

    # Initialize tokenized dataset
    dataset = {"input_ids": [], "attention_mask": [], "labels": []}

    # Tokenize each example
    for i, example in enumerate(humaneval_dataset):
        query = example["prompt"]  # Function signature and docstring
        completion = example["canonical_solution"]  # Solution code

        full_input_ids, labels, attention_mask = tokenize(
            tokenizer=tokenizer,
            query=query,
            completion=completion,
            max_length=max_length,
            print_ex=(i == 0)
        )

        # Append tokenized data to the dataset
        dataset["input_ids"].append(full_input_ids)
        dataset["labels"].append(labels)
        dataset["attention_mask"].append(attention_mask)

    # Convert the dictionary to a Hugging Face Dataset object
    return Dataset.from_dict(dataset)


def get_dataset(task, **kwargs):
    """
    Get the dataset for the given task.

    Args:
        task_name (str): The name of the task.

    Raises:
        ValueError: If the task name is not valid.

    Returns:
        Dataset: The dataset.
    """
    
    if task == "humaneval":
        return get_humaneval_dataset(**kwargs)
    else:
        raise ValueError("Invalid task name")


def get_dataloader(dataset, tokenizer, batch_size=1):
    data_collator = DataCollatorForSeq2Seq(
            tokenizer=tokenizer, padding="longest") 
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,  # When getting gradients, we only do this single batch process
                            collate_fn=data_collator)
    print("There are {} examples in the dataset".format(len(dataset)))
    return dataloader
