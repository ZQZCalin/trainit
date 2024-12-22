"""Initialize models."""

import transformers
import models
import equinox as eqx
from omegaconf import DictConfig
from typing import Any, Tuple
from jaxtyping import PRNGKeyArray
import jax.random as jr


def init_tokenizer(config: DictConfig):
    """Initializes tokenizer. 
    
    Args:
        config: global_config.model. 
            if config.pad_token is true, adds pad_token as a special token.
    """
    if config.name == "gpt":
        tokenizer = transformers.GPT2TokenizerFast.from_pretrained("gpt2")
        if config.pad_token:
            tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
    else:
        raise ValueError(f"invalid config: model.name '{config.name}' is not supported.")
    return tokenizer


def init_model(vocab_size: int, config: DictConfig, *, key: PRNGKeyArray) -> eqx.Module:
    """Initializes model. 
    
    Args:
        config: global_config.model.
    """
    # NOTE: we disable loading torch gpt for simplicity.
    if config.name == "gpt":
        model = models.GPT(vocab_size, config, key=key)
    else:
        raise ValueError(f"invalid config: model.name '{config.name}' is not supported.")
    return model


def init_model_and_tokenizer(config: DictConfig) -> Tuple[eqx.Module, Any]:
    """Wraps init_tokenizer and init_model for language models.
    
    Args:
        config: global_config.
    """
    if config.name == "gpt":
        tokenizer = init_tokenizer(config.model)
        vocab_size = len(tokenizer)
        seed = config.model.seed
        model = init_model(
            vocab_size=vocab_size,
            config=config.model,
            key=jr.PRNGKey(seed),
        )
        return model, tokenizer
    else:
        raise ValueError(f"invalid config: model.name '{config.name}' is not supported.")