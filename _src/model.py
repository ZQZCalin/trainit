"""Initialize models."""

import transformers
import models
import equinox as eqx
from omegaconf import DictConfig
from typing import Any, Tuple
from jaxtyping import PRNGKeyArray


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


def init_language_model(
        config: DictConfig, 
        *, 
        key: PRNGKeyArray
    ) -> Tuple[eqx.Module, Any]:
    """Initializes language models.
    
    Args:
        config: global_config.

    Returns:
        A tuple of model and tokenizer.
    """
    tokenizer = init_tokenizer(config.model)
    if config.name == "gpt":
        vocab_size = len(tokenizer)
        model = models.GPT(vocab_size, config, key=key)
        return model, tokenizer
    else:
        raise ValueError(f"invalid config: model.name '{config.name}' is not supported.")