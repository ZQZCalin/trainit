"""A stateless log function"""

import jax
from jax import numpy as jnp
from jax import random as jr
from jax import tree_util as jtu
import chex
import optax
from typing import Any, Tuple, NamedTuple, Optional, Union, Callable, Protocol, Literal
from omegaconf import DictConfig
from utils import tree_utils


LogState = chex.ArrayTree
LogMetric = chex.ArrayTree


class LogInitFn(Protocol):
    def __call__(self, **extra_args: Any) -> LogState:
        """The `init` function."""


class LogUpdateFn(Protocol):
    def __call__(self, state: LogState, **extra_args: Any) -> Tuple[LogState, LogMetric]:
        """The `update` function."""


class LogFn(NamedTuple):
    """Loss function implemented by value and gradient function."""
    init: LogInitFn
    update: LogUpdateFn


class StandardLogState(NamedTuple):
    grad_prev: optax.Updates
    params_prev: optax.Params
    cumulatives: chex.ArrayTree


def standard_log() -> LogFn:
    """An example of a log function."""
    def init_fn(params: optax.Params):
        state = StandardLogState(
            grad_prev = jnp.zeros_like(params),
            params_prev = params,
            cumulatives = {
                "loss_min": jnp.array(float("inf")),
                "grad/inner_sum": jnp.zeros([]),
                "grad/cosine_sum": jnp.zeros([]),
            },
        )
        return state
    
    def update_fn(state, loss_val: chex.Array, params: optax.Params, grad: optax.Updates):
        grad_prev = state.grad_prev
        params_prev = state.params_prev
        cumulatives = state.cumulatives

        cumulatives["loss_min"] = jnp.minimum(cumulatives["loss_min"], loss_val)
        cumulatives["grad/inner_sum"] += _tree.inner(grad, grad_prev)
        cumulatives["grad/cosine_sum"] += _tree.cosine(grad, grad_prev)

        metric = {
            "loss": loss_val,
            "grad/norm": _tree.norm(grad),
            "grad/inner": _tree.inner(grad, grad_prev),
            "grad/cosine": _tree.cosine(grad, grad_prev),
        }
        metric.update(cumulatives)

        state = StandardLogState(
            grad_prev = grad,
            params_prev = params,
            cumulatives = cumulatives,
        )
        return state, metric
    
    return LogFn(init_fn, update_fn)




def init_log(config: DictConfig) -> LogFn:
    return standard_log()