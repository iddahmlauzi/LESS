import contextlib
from functools import partial
from typing import List, Union

import numpy as np
import torch
from datasets import load_dataset, Features, Value


@contextlib.contextmanager
def temp_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


def get_training_dataset(dataset_name, tokenizer, max_seq_length, sample_percentage=1.0, seed=0):
    """ get training dataset with a specified seed """
    
    print("We are getting the training dataset")

    raw_dataset = load_raw_dataset(
        dataset_name, sample_percentage=sample_percentage, seed=seed)
    lm_datasets = encode_data(
        raw_dataset, tokenizer, max_seq_length)
    return lm_datasets

def load_raw_dataset(dataset_name: str, sample_size=None, sample_percentage=1.0, seed=0):
    """ load raw dataset from Hugging Face Datasets """
    
    # Load dataset from Hugging Face directly
    features = Features({
    "text": Value("string"),
    "source": Value("string") 
    })

    # Load the dataset with the expected features (schema)
    processed_dataset = load_dataset("UDACA/Code-Mixed-Dataset", split="train", features=features)
    
    # Calculate sample size if not provided
    if sample_size is None:
        sample_size = int(len(processed_dataset) * sample_percentage)

    # If no sampling is needed, return the full dataset
    if sample_size == len(processed_dataset):
        return processed_dataset  # no shuffle

    # Shuffle and sample the dataset with a fixed seed
    with temp_seed(seed):
        index = np.random.permutation(len(processed_dataset))[:sample_size]

    sampled_dataset = processed_dataset.select(index)

    return sampled_dataset


def encode_data(raw_datasets, tokenizer, max_seq_length, processing_num_workers=10, overwrite_cache=False, func_name="encode_with_messages_format"):
    """ encode data with the specified tokenizer and format. """
    # if already encoded, return
    if "input_ids" in raw_datasets.features:
        return raw_datasets
    encode_function = get_encode_function(
        raw_datasets, tokenizer, max_seq_length, func_name)
    # To speed up this part, we use multiprocessing.
    lm_datasets = raw_datasets.map(
        encode_function,
        batched=False,
        num_proc=processing_num_workers,
        load_from_cache_file=not overwrite_cache,
        desc="Tokenizing and reformatting data",
    )
    lm_datasets.set_format(type="pt")
    return lm_datasets


def get_encode_function(raw_datasets, tokenizer, max_seq_length, func="encode_with_messages_format"):
    """ get encode function based on the dataset. """
    if "text" in raw_datasets.column_names:
        encode_function = partial(
            encode_with_text_format,
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
        )
    else:
        raise ValueError("Dataset must contain a 'text' column.")
    return encode_function

def encode_with_text_format(example, tokenizer, max_seq_length):
    """
    Custom encoding function for text data. 
    Assumes each example has a 'text' field which contains the raw text.
    """
    # Tokenize the text column
    example_text = example['text']
    tokenized_example = tokenizer(
        example_text, return_tensors='pt', max_length=max_seq_length, truncation=True)
    
    input_ids = tokenized_example.input_ids
    labels = input_ids.clone()  # We use the input_ids as the labels for language modeling
    
    # Create attention mask 
    attention_mask = torch.ones_like(input_ids) # No padding, so mask is all 1s
    
    return {
        'input_ids': input_ids.flatten(),
        'labels': labels.flatten(),
        'attention_mask': attention_mask.flatten(),
    }