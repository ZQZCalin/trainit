"""Initialize models."""

import transformers
import models
import equinox as eqx
from omegaconf import DictConfig
from typing import Any, Tuple
from jaxtyping import PRNGKeyArray


def init_tokenizer(config: DictConfig, pad_token: bool = True):
    """Initializes tokenizer. If `pad_token` is true, adds pad_token as a special token. Defaults to true."""
    model_name = config.model.name
    if model_name == "gpt":
        tokenizer = transformers.GPT2TokenizerFast.from_pretrained("gpt2")
        if pad_token:
            tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
    else:
        raise ValueError(f"model {model_name} is not supported.")
    return tokenizer


def init_model(vocab_size: int, config: DictConfig, *, key: PRNGKeyArray) -> eqx.Module:
    """Initializes model. config: global_config.model"""
    if not config.load_pytorch:
        model = models.GPT(vocab_size, config, key=key)
    else:
        model_config = torch_GPT.get_default_config()
        model_config.model_type = 'gpt2'
        model_config.vocab_size = vocab_size                    # openai's model vocabulary
        model_config.block_size = config.context_length         # openai's model block_size (i.e. input context length)
        model_config.embd_pdrop = config.transformer_dropout
        model_config.resid_pdrop = config.attn_linear_dropout
        model_config.attn_pdrop = config.attn_dropout
        torch_model = torch_GPT(model_config)
        model = models.GPT(vocab_size, config, state_dict=torch_model.state_dict())
    return model


def init_tokenizer_and_model(config: DictConfig) -> Tuple[Any, eqx.Module]:
    return 