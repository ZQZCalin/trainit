from datasets import load_dataset, disable_caching
from transformers import DataCollatorForLanguageModeling
from torch.utils.data import DataLoader
import torch.nn.functional as F
from einops import rearrange
import random
import functools
from pathlib import Path


def shift_labels(batch):
    new_batch = {k: v for (k,v) in batch.items()}
    new_batch["labels"] = F.pad(batch["labels"][:, 1:], (0, 1), value=-100)
    return new_batch

    # Updated on 09/17: the current implementation modifies the original copy of batch.
    # Instead, we want to keep it unchanged. The above implementation returns a new copy.
    # However, we need to test whether the new implementation causes memory overflow.
    batch["labels"] = F.pad(batch["labels"][:, 1:], (0, 1), value=-100)
    return dict(batch)


def split_sequences(batch, max_length):
    # assumes that batch has shape [B, L] where L is a multiple of max_length
    batch["size"] = batch["input_ids"].size()
    batch["input_ids"] = rearrange(batch["input_ids"], "b (l c) -> (b l) c")
    batch["labels"] = rearrange(batch["labels"], "b (l c) -> (b l) c")
    return batch


def postprocess_collate_fn(collate_fn, post_fn):
    old_torch_call = collate_fn.torch_call

    def new_torch_call(self, *args, **kwargs):
        batch = old_torch_call(self, *args, **kwargs)
        return post_fn(batch)

    collate_fn.torch_call = new_torch_call
    return collate_fn


def get_lm_loader_next_token(
    tokenizer,
    split,
    batch_size,
    max_length=None,
    shuffle_buffer_size=0,
    pad_to_multiple_of=None,
    mlm=False,
    mlm_probability=0,
    random_start=False,
    dataset="c4",
    ds_path=None,
    num_workers=2,
    output_format="torch",
    **collator_args,
):
    """
    Produces a pytorch dataloader object to train C4 on a "next token prediction" task
    in which each example is some length L tokenized text, and the model must
    predict the nth token using the first n-1 tokens for each n from 1 to L+1.

    For a batch size of B, the each entry in the dataloader will have columns:
        'size': a list of B integers describing the number of tokens in each example.
        'input_ids': a BxL matrix of token ids where L is the maximum sequence length.
        'labels': a BxL matrix of target token ids. Each row of 'labels' is usually
            simply a shifted version of 'input_ids', so that actually only the
            last entry in the row is not computable from 'input_ids'. Despite this
            redundancy, it is simpler in downstream code to have the labels available
            in this format.

    Arguments:
        tokenizer: the tokenizer to use (should be Huggingface tokenizer).
        split: 'train' or 'test'.
        batch_size: integer, the batch size.
        max_length: restrict sequences to this max length (if None, then no restriction).
        shuffle_buffer_size: if >0, will "shuffle" the data using a lookahead buffer of this size.
            Larger values require more memory but provide a better shuffle.
        pad_to_multiple_of: pad sequences to a multiple of this value, if provided.
        mlm: if true, blank out some tokens at random in the input.
        mlm_probability: probability to blank out any given token.
        random_start: start each sequence in a random point in the original sequence.
        dataset: name of dataset to use - can be "pile" or "c4"
        ds_path: local path to the data. Set to "None" to use defaults.
        num_worker: number of CPU threads to use for this.
        **collator args: extra arguments to pass to DataCollatorForLanguageModeling


    Returns:
        pytorch DataLoader for the C4 dataset.
    """

    if dataset == "c4":
        hf_dataset = get_c4_dataset(split=split, ds_path=ds_path)
    elif dataset == "pile":
        hf_dataset = get_pile_dataset(split=split, ds_path=ds_path)
    else:
        raise ValueError(f"requested unknown dataset: {dataset}")
    collate_fn = DataCollatorForLanguageModeling(
        tokenizer,
        mlm=mlm,
        mlm_probability=mlm_probability,
        pad_to_multiple_of=pad_to_multiple_of,
        **collator_args,
    )

    collate_fn = postprocess_collate_fn(collate_fn, shift_labels)
    return get_lm_loader_from_collate_fn(
        tokenizer=tokenizer,
        hf_dataset=hf_dataset,
        batch_size=batch_size,
        shuffle_buffer_size=shuffle_buffer_size,
        max_length=max_length,
        random_start=random_start,
        collate_fn=collate_fn,
        num_workers=num_workers,
        output_format=output_format,
    )


def get_c4_dataset(
    split,
    ds_path=None,
):
    if ds_path is None:
        ds_path = "/projectnb/aclab/datasets/c4/en/"
    if split in ["val", "valid", "validation"]:
        # normalize different possible ways to ask for validation set.
        split = "validation"

    disable_caching()
    c4 = load_dataset("c4", "en", data_dir=ds_path, streaming=True, split=split)
    return c4


def get_pile_dataset(
    split,
    ds_path=None,
):
    if ds_path is None:
        ds_path = "/projectnb/aclab/datasets/pile/raw_data/"

    disable_caching()
    ds_path = Path(ds_path)

    if split in ["val", "valid", "validation"]:
        # normalize different possible ways to ask for validation set.
        split = "validation"

    # some fun syntatic magic with Path objects: look in the directory given by 
    # '{ds_path}/{split}'
    ds_path = ds_path / split  

    json_files = ds_path.glob("*.jsonl")  # grab all jsonl files in the directory.

    json_files = [str(x) for x in json_files]

    pile = load_dataset("json", data_files=json_files, streaming=True)

    # for some reason, when we load this way the dataset is always
    # placed in the  "train" key:
    pile = pile["train"]

    return pile


def get_lm_loader_from_collate_fn(
    tokenizer,
    hf_dataset,
    batch_size,
    max_length,
    shuffle_buffer_size,
    random_start,
    collate_fn,
    num_workers=2,
    output_format="torch",
):
    disable_caching()
    # c4 = load_dataset("c4", "en", data_dir=ds_path, streaming=True, split=split)
    # c4 = c4.filter(lambda x: len(x["text"]) > 1)
    if shuffle_buffer_size > 0:
        hf_dataset = hf_dataset.shuffle(buffer_size=shuffle_buffer_size)
    if random_start:
        hf_dataset = hf_dataset.map(
            lambda examples: {
                "text": examples["text"][random.randint(0, len(examples["text"])) :]
            }
        )
    # this is a huge hack... I'd like to believe there is a better way to find
    # out the columns, but honestly it's possible there isn't.
    columns = list(next(iter(hf_dataset)).keys())

    hf_dataset = hf_dataset.map(
        lambda examples: tokenizer(
            examples["text"], padding=True, truncation=True, max_length=max_length
        ),
        remove_columns=columns,
        batched=True,
        batch_size=batch_size,
    )
    hf_dataset = hf_dataset.with_format(output_format)

    dataloader = DataLoader(
        hf_dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=num_workers,
    )
    return dataloader