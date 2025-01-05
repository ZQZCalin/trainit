"""Initialize models."""

import transformers
import models
import equinox as eqx
from omegaconf import DictConfig
from typing import Any, Tuple, List
from jaxtyping import PRNGKeyArray
from _src.base import language_models, vision_models, basic_models


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
    ) -> eqx.Module:
    """Initializes language models.
    
    Args:
        config: global_config.model.
        key: random key in model construction.

    Returns:
        An `eqx.Module` object.
    """
    tokenizer = init_tokenizer(config)
    if config.name == "gpt":
        vocab_size = len(tokenizer)
        model = models.GPT(vocab_size, config, key=key)
        return model
    else:
        raise ValueError(f"invalid config: model.name '{config.name}' is not supported.")
    

def init_vision_model():
    """Initializes vision models."""
    raise NotImplementedError


def init_basic_model():
    """Initializes basic models"""
    raise NotImplementedError
    

def init_model(
		config: DictConfig,
		*,
		key: PRNGKeyArray,
) -> List[eqx.Module,]:
	"""Initializes the training model.
	
	Args:
		config: global_config.
		key: a PRNGKey for model initialization.
		
	Return:
		An equinox model, and optinally additional variables.
		language models: the model and its associated tokenizer.
		vision models: only the model.
	"""
	name = config.model.name
	if name in language_models:
		return init_language_model(
            config.model, 
            key=key,
        )
	if name in vision_models:
		return init_vision_model()
	if name in basic_models:
		return init_basic_model()
	raise ValueError(f"invalid config: model.name == '{name}' is not supported.")