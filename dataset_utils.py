import os
from itertools import chain


def fixed_seq_length_of_datasets(
    dataset,
    fixed_seq_length,
    load_from_cache_file=False,
):
    block_size = fixed_seq_length

    # Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])

        # Padding in front of tokens to align it with the group size.
        if total_length % block_size != 0:
            total_length = (total_length // block_size) * block_size

        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        return result

    lm_datasets = dataset.map(
        group_texts,
        batched=True,
        load_from_cache_file=load_from_cache_file,
        desc=f"Grouping texts in chunks of {block_size}",
    )

    return lm_datasets


def prepare_dataset(
    raw_dataset,
    tokenizer,
    seq_length=512,
    overwrite_cache=False,
):
    column_names = raw_dataset["train"].column_names
    text_column_name = "text"

    tokenized_dataset = raw_dataset.map(
        lambda examples: tokenizer(examples[text_column_name]),
        batched=True,
        remove_columns=column_names,
        load_from_cache_file=not overwrite_cache,
        desc="Running tokenizer on dataset",
    )

    lm_dataset = fixed_seq_length_of_datasets(
        tokenized_dataset,
        seq_length,
        load_from_cache_file=not overwrite_cache,
    )

    return lm_dataset
