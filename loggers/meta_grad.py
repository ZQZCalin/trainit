"""Logger for meta-gradient method."""

import jax
from jax import numpy as jnp
from jax import random as jr
from jax import tree_util as jtu
import optax
from typing import Union, Optional, NamedTuple
from jaxtyping import Array, PyTree
from omegaconf import DictConfig
from utils import tree_utils
from loggers import base


class MetaGradLoggerState(NamedTuple):
    """empty"""


def meta_grad_logger(config: DictConfig = None) -> base.Logger:
    """A very minimal log function.
    
    Examples:
        >>> from loggings import minimal_logger
        >>> logger = minimal_logger()
        >>> log_state = logger.init(params=...)
        >>> log_state, log_metrics = logger.update(log_state, loss_val=..., params=..., grads=...)
    """
    del config

    def init_fn(params: optax.Params, **kwargs):
        state = MetaGradLoggerState()
        metrics = {}
        return state, metrics
    
    def update_fn(state, **kwargs):
        metric = {}
        return state, metric
    
    return base.Logger(init_fn, update_fn)
