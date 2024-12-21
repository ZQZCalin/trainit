import hydra
from omegaconf import OmegaConf, DictConfig
import jax
from jax import tree_util as jtu
from jax import numpy as jnp
from jax import random as jr
import torch
from torch import nn
import torch.nn.functional as F
import equinox as eqx
from models.mingpt import GPT
from models.extend_nn import Linear, LayerNorm
from types import SimpleNamespace
from models.utils import parse_state_dict
import re
from typing import Any, Tuple
from jaxtyping import Array
from loader.lm_loader import get_lm_loader_next_token
import transformers
from accelerate import Accelerator
from utils import softmax_cross_entropy
from loadit import LoadIt

import sys
sys.path.append('./minGPT')
from mingpt.model import GPT as minGPT


"""
We need some systematic tests to explain the different behavior between neurips submission vs our current setup.
A few factors that might affect the performance:
- randomness
- optimizer and corresponding hyperparameters
- model structure
    - hugging face models use attention_masks from dataset?
- tokenizer
- dataset
"""


def get_mingpt(tokenizer):
    context_length = 1024
    model_config = minGPT.get_default_config()
    model_config.model_type = "gpt2"
    model_config.vocab_size = len(tokenizer)
    model_config.block_size = context_length  # openai's model block_size (i.e. input context length)
    model_config.embd_pdrop = 0
    model_config.resid_pdrop = 0
    model_config.attn_pdrop = 0
    return minGPT(model_config)


def get_hf_gpt(tokenizer):
    model_conf = transformers.AutoConfig.from_pretrained("gpt2", trust_remote_code=False)
    model_conf.attn_pdrop = 0.0
    model_conf.resid_pdrop = 0.0
    model_conf.embd_pdrop = 0.0
    model = transformers.AutoModelForCausalLM.from_config(model_conf)
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))
    return model


def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def test1():
    print("\n>>>Testing on hugging face tokenizer without <|pad|> token",
          "with LoadIt dataset (batch_size=2)...")

    tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2", use_fast=True, trust_remote_code=False)
    train_loader = LoadIt(root_dir="/projectnb/aclab/tranhp/trainDataloader_pile/", max_workers=1)
    device = get_device()

    mingpt = get_mingpt(tokenizer).to(device)
    hf_gpt = get_hf_gpt(tokenizer).to(device)

    loss_mingpt = 0
    loss_hf = 0   
    loss_hf_no_mask = 0
    N = 10
    for i, batch in enumerate(train_loader):
        if i > N:
            break
        batch = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
        idx = batch["input_ids"]
        targets = batch["labels"]
        if i == 0:
            print(f"Batch keys: {batch.keys()}",
                  f"Input shape: {idx.shape}")

        with torch.no_grad():
            # mingpt
            logits, _ = mingpt(idx)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            loss_mingpt += loss.detach().float()

            # hugging face
            loss = hf_gpt(**batch).loss
            loss_hf += loss.detach().float()

            # hugging face no mask
            loss = hf_gpt(input_ids=idx, labels=targets).loss
            loss_hf_no_mask += loss.detach().float()

    print(f"mingpt loss: {loss_mingpt/N}",
        f"\nhugging face loss: {loss_hf/N}",
        f"\nhugging face no mask loss: {loss_hf_no_mask/N}")


def test2():
    print("\n>>>Testing loadit dataset vs streaming dataset...")
    tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2", use_fast=True, trust_remote_code=False)
    tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

    context_length = 1024
    batch_size = 2
    trainloader_streaming = get_lm_loader_next_token(
        tokenizer,
        split="train",
        batch_size=batch_size,
        max_length=context_length,
        shuffle_buffer_size=0,
        pad_to_multiple_of=context_length,
        num_workers=2,
        dataset="pile",
    )
    trainloader_loadit = LoadIt(root_dir="/projectnb/aclab/tranhp/trainDataloader_pile/", max_workers=1)

    N = 5
    for i, (batch_streaming, batch_loadit) in enumerate(zip(trainloader_streaming, trainloader_loadit)):
        if i > N:
            break
        print("streaming:", batch_streaming)
        print("loadit:", batch_loadit)
        # san check: it looks like in loadit, inputs = labels???
        # print("loadit input = labels:", batch_loadit["input_ids"] == batch_loadit["labels"])


def test3():
    print("\n>>>Checking what happens for loadit data if texts have different lengths...")
    train_loader = LoadIt(root_dir="/projectnb/aclab/tranhp/trainDataloader_pile/", max_workers=1)
    # N = 10000
    for i, batch in enumerate(train_loader):
        # if i > N:
        #     break
        contains_zero = batch["attention_mask"].eq(0).any().item()
        # print(batch)
        # print(f"Is the batch not filled: {contains_zero}")
        if contains_zero:
            break
        if i % 1000 == 0:
            print(f"num data: {i}")


def test4():
    print("\n>>>Testing loadit dataset vs streaming dataset...")
    tokenizer_streaming = transformers.AutoTokenizer.from_pretrained("gpt2", use_fast=True, trust_remote_code=False)
    tokenizer_streaming.add_special_tokens({"pad_token": "<|pad|>"})
    tokenizer_loadit = transformers.AutoTokenizer.from_pretrained("gpt2", use_fast=True, trust_remote_code=False)

    context_length = 1024
    batch_size = 2
    trainloader_streaming = get_lm_loader_next_token(
        tokenizer_streaming,
        split="train",
        batch_size=batch_size,
        max_length=context_length,
        shuffle_buffer_size=0,
        pad_to_multiple_of=context_length,
        num_workers=2,
        dataset="pile",
    )
    trainloader_loadit = LoadIt(root_dir="/projectnb/aclab/tranhp/trainDataloader_pile/", max_workers=1)

    N = 1
    for i, batch_streaming in enumerate(trainloader_streaming):
        if i > N:
            break
        batch_loadit = trainloader_loadit[i]

        tokens_streaming = batch_streaming["input_ids"][0]
        print("streaming:", batch_streaming)
        print(tokenizer_streaming.decode(tokens_streaming))

        tokens_loadit = batch_loadit["input_ids"][0]
        print("loadit:", batch_loadit)
        print(tokenizer_loadit.decode(tokens_loadit))


if __name__ == "__main__":
    # test1()
    # test2()
    # test3()
    test4()