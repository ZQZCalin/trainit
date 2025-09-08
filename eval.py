# All-in-one (and mostly self-contained) code to load checkpoint and eval.
# 
# Usage:
#   cd /projectnb/aclab/qinziz/trainit
#   module load python3/3.10.12 cuda/12.2
#   source /projectnb/aclab/qinziz/trainit/env/bin/activate
#   dir=ABS/PATH/TO/CKPT/FOLDER
#   ckpt=CHECKPOINT_model.ckpt
#   python eval.py --dir $dir --ckpt $ckpt --batch_size 16 2>&1 | tee .log; cat .log


import torch.nn.functional as F
from torch.utils.data import DataLoader
import transformers
from transformers import DataCollatorForLanguageModeling
from datasets import load_dataset, disable_caching

from jax import numpy as jnp
from jax import random as jr
from jax import tree_util as jtu

import equinox as eqx

import argparse
from omegaconf import OmegaConf, DictConfig

import os
import subprocess
from pathlib import Path

from tqdm import tqdm
import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s: %(message)s",
)
logger: logging.Logger = logging.getLogger("trainit_eval")

# Reuse model and loss from the pipeline
from _src.model import init_model, init_tokenizer
from _src.loss import init_loss_fn
from serialize import serializer


# ==============================================================================
# Quick implementation of eval dataloader
# ==============================================================================

def get_raw_datasets(raw_files):
    """
    Returns:
        a dictionary of raw datasets (IterableDataset), with keys "train", "val", "eval".
    """
    # Check raw data files.
    for split, raw_path in raw_files.items():
        result = subprocess.run(
            ["ls", "-l", raw_path],
            capture_output=True,
            text=True
        )
        logger.debug(f"{split} files:\n{result.stdout}")
    
    # Convert to list of jsonl under "train/".
    raw_files["train"] = [
        str(file) for file in Path(raw_files["train"]).glob("*.jsonl")
    ]

    # Construct raw dataset from jsonl files.
    raw_datasets = load_dataset(
        "json", data_files=raw_files, cache_dir=".cache", streaming=True
    )

    # Peak into raw datasets.
    for split, dataset in raw_datasets.items():
        example = next(iter(dataset))
        logger.debug(f"Example from '{split}' dataset:\n{example}\n")
    return raw_datasets


# Data processing help functions.
def shift_labels(batch):
    """Shift labels to the left and pad the last with -100."""
    new_batch = {k: v for (k,v) in batch.items()}
    new_batch["labels"] = F.pad(batch["labels"][:, 1:], (0, 1), value=-100)
    return new_batch


def postprocess_collate_fn(collate_fn, post_fn):
    old_torch_call = collate_fn.torch_call

    def new_torch_call(self, *args, **kwargs):
        batch = old_torch_call(self, *args, **kwargs)
        return post_fn(batch)

    collate_fn.torch_call = new_torch_call
    return collate_fn


def get_dataloader(
        raw_dataset,
        tokenizer,
        max_length,
        pad_to_multiple_of,
        batch_size,
        num_workers,
        use_shift_labels,
):
    collate_fn = DataCollatorForLanguageModeling(
        tokenizer,
        mlm=False,
        mlm_probability=0.0,
        pad_to_multiple_of=pad_to_multiple_of,
    )
    if use_shift_labels:
        collate_fn = postprocess_collate_fn(collate_fn, shift_labels)

    disable_caching()
    columns = list(next(iter(raw_dataset)).keys())

    raw_dataset = raw_dataset.map(
        lambda examples: tokenizer(
            examples["text"], padding=True, truncation=True, max_length=max_length
        ),
        remove_columns=columns,
        batched=True,
        batch_size=batch_size,
    )
    raw_dataset = raw_dataset.with_format("torch")

    dataloader = DataLoader(
        raw_dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=num_workers,
    )
    return dataloader


def init_eval_dataloader(config, eval_batch_size):
    raw_files = {
        "train": "/projectnb/aclab/datasets/pile/raw_data/train/",
        "val": "/projectnb/aclab/datasets/pile/raw_data/val.jsonl",
        "eval": "/projectnb/aclab/datasets/pile/raw_data/test.jsonl",
    }
    raw_datasets = get_raw_datasets(raw_files)
    tokenizer = init_tokenizer(config.model)
    eval_dataloader = get_dataloader(
        raw_dataset=raw_datasets["eval"],
        tokenizer=tokenizer,
        max_length=config.model.context_length,
        pad_to_multiple_of=config.model.context_length,
        batch_size=eval_batch_size,
        num_workers=config.dataset.dataloader_workers,
        use_shift_labels=config.dataset.shift_labels,
    )

    return eval_dataloader


# python eval.py \
#     --ckpt "/projectnb/aclab/qinziz/trainit/scheduler_outputs/2025-05-04/v5_4seg_peak2e-3_eps0.24const_grid10_0a940a/checkpoint/1400-2000/lr2:7.00e-03" \
#     2>&1 | tee .log; cat .log

# ==============================================================================
# Main eval function
# ==============================================================================

def eval_step(
        model,
        batch,
        loss_fn,
):
    # NOTE@ZQZCalin: the model is deterministic, so we use a dummy key.
    dummy_key = jr.PRNGKey(42)
    loss, logits = loss_fn(model, batch, key=dummy_key)

    # Get accuracy
    _, target = batch
    predictions = jnp.argmax(logits, axis=2)
    correct_tokens = jnp.sum(predictions == target)
    total_tokens = jnp.sum(target != -100)
    return loss, correct_tokens, total_tokens

def eval(
        model,
        dataloader,
        loss_fn,
):
    eval_step_jit = eqx.filter_jit(
        jtu.Partial(eval_step),
    )
    eval_loss = 0
    correct_tokens = 0
    total_tokens = 0
    pbar = tqdm(
        enumerate(dataloader)
    )
    for idx, batch in pbar:
        input_ids = jnp.asarray(batch["input_ids"])
        labels = jnp.asarray(batch["labels"])
        batch = (input_ids, labels)
        loss, correct, total = eval_step_jit(model, batch, loss_fn)

        if jnp.isnan(loss):
            pbar.set_description(f"batch {idx+1}: eval_loss = {loss}")
            continue

        eval_loss += loss
        correct_tokens += correct
        total_tokens += total

        pbar.set_description(
            f"batch {idx+1}: \
            eval_loss = {eval_loss/(idx+1)}, \
            eval_acc = {correct_tokens/total_tokens}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, help="absolute path of checkpoint folder")
    parser.add_argument("--ckpt", type=str, help="checkpoint name")
    parser.add_argument("--cfg", type=str, default="config.yaml")
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    config = OmegaConf.load(
        os.path.join(args.dir, args.cfg)
    )
    logger.info(
        f"""
        Eval from checkpoint:
        - path: {args.dir}
        - checkpoint: {args.ckpt}
        - config: {args.cfg}
        """
    )
    logger.info(
        f"Loaded from checkpoint config: \n{OmegaConf.to_yaml(config)}"
    )
    
    dataloader = init_eval_dataloader(config, args.batch_size)
    model = init_model(config, key=jr.PRNGKey(42)) # NOTE@ZQZCalin: dummy random seed
    model = serializer.load(
        os.path.join(args.dir, args.ckpt), model
    )
    loss_fn = init_loss_fn(config)

    eval(model, dataloader, loss_fn)



if __name__ == "__main__":
    main()