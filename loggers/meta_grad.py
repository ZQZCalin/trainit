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


def _safe_update_log_dict(target: dict, new_dict: dict) -> dict:
    unknown = new_dict.keys() - target.keys()
    if unknown:
        raise KeyError(f"keys not in target dict: {unknown}")
    return {**target, **new_dict}


class MetaGradLoggerState(NamedTuple):
    """empty"""
    update_prev: Optional[optax.Updates]
    update_prev_ema: Optional[optax.Updates]


def meta_grad_logger(config: DictConfig = None) -> base.Logger:
    """A very minimal log function.
    
    Examples:
        >>> from loggings import minimal_logger
        >>> logger = minimal_logger()
        >>> log_state = logger.init(params=...)
        >>> log_state, log_metrics = logger.update(log_state, loss_val=..., params=..., grads=...)
    """

    log_update_grad_corr: bool = config.log_update_grad_corr
    log_ema_update_grad_corr: bool = config.log_ema_update_grad_corr
    beta: float = config.update_ema_constant

    store_update_prev = log_update_grad_corr or log_ema_update_grad_corr
    store_update_prev_ema = log_ema_update_grad_corr

    # metric keys
    CORR = "optim/<u_{t-1}, g_t>"
    EMA_CORR = f"optim/<ema(u_[t-1], {beta}), g_t>"
    default_metrics = {
        CORR: jnp.zeros([]),
        EMA_CORR: jnp.zeros([]),
    }

    def _filter_metrics(metrics: dict):
        res = {}
        if log_update_grad_corr:
            res[CORR] = metrics[CORR]
        if log_ema_update_grad_corr:
            res[EMA_CORR] = metrics[EMA_CORR]
        return res

    def init_fn(params: optax.Params, **kwargs):
        state = MetaGradLoggerState(
            update_prev=tree_utils.zeros_like(params) if store_update_prev else None,
            update_prev_ema=tree_utils.zeros_like(params) if store_update_prev_ema else None,
        )
        metrics = default_metrics
        return state, _filter_metrics(metrics)
    
    def update_fn(
            state: MetaGradLoggerState, 
            grads: optax.Updates,
            updates: optax.Updates, 
            **kwargs
        ):
        update_prev = state.update_prev
        update_prev_ema = state.update_prev_ema
        metrics = {}
        if store_update_prev:
            metrics.update({
                CORR: tree_utils.inner(update_prev, grads),
            })
        if store_update_prev_ema:
            update_prev_ema = jtu.tree_map(
                lambda m, u: beta * m + (1-beta) * u, update_prev_ema, update_prev,
            )
            metrics.update({
                EMA_CORR: tree_utils.inner(update_prev_ema, grads),
            })
        metrics = _safe_update_log_dict(default_metrics, metrics)
        state = state._replace(
            update_prev=updates,
            update_prev_ema=update_prev_ema,
        )
        return state, _filter_metrics(metrics)
    
    return base.Logger(init_fn, update_fn)