"""A demo of different instances of LoggingFn."""

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


class SimpleLogState(NamedTuple):
    grads_prev: optax.Updates
    params_prev: optax.Params
    cumulatives: PyTree


def simple_log() -> base.Logger:
    """A very minimal log function.
    
    Examples:
        >>> from loggings import simple_log
        >>> logger = simple_log()
        >>> log_state = logger.init(params=...)
        >>> log_state, log_metrics = logger.update(log_state, loss_val=..., params=..., grads=...)
    """
    def init_fn(params: optax.Params):
        state = SimpleLogState(
            grads_prev = jnp.zeros_like(params),
            params_prev = params,
            cumulatives = {
                "loss_min": jnp.array(float("inf")),
                "grad/inner_sum": jnp.zeros([]),
                "grad/cosine_sum": jnp.zeros([]),
            },
        )
        metrics = {
            "loss": jnp.zeros([]),
            "grad/norm": jnp.zeros([]),
            "grad/inner": jnp.zeros([]),
            "grad/cosine": jnp.zeros([]),
        }
        metrics.update(state.cumulatives)
        return state, metrics
    
    def update_fn(
            state, 
            loss_val: Array, 
            params: optax.Params, 
            grads: optax.Updates
        ):
        grads_prev = state.grads_prev
        params_prev = state.params_prev
        cumulatives = state.cumulatives

        cumulatives["loss_min"] = jnp.minimum(cumulatives["loss_min"], loss_val)
        cumulatives["grad/inner_sum"] += tree_utils.inner(grads, grads_prev)
        cumulatives["grad/cosine_sum"] += tree_utils.cosine(grads, grads_prev)

        metric = {
            "loss": loss_val,
            "grad/norm": tree_utils.norm(grads),
            "grad/inner": tree_utils.inner(grads, grads_prev),
            "grad/cosine": tree_utils.cosine(grads, grads_prev),
        }
        metric.update(cumulatives)

        state = SimpleLogState(
            grads_prev = grads,
            params_prev = params,
            cumulatives = cumulatives,
        )
        return state, metric
    
    return base.Logger(init_fn, update_fn)


class FullLogState(NamedTuple):
    """full_log state."""
    params_prev: Optional[optax.Updates]        # x(n-1)
    grads_prev: Optional[optax.Updates]         # g(n-1)
    grads_hist: Optional[optax.Updates]         # g(1:n-1)
    num_inf_grads: Array
    metrics_prev: base.LogMetrics


def full_log(
        config: DictConfig,
) -> base.Logger:
    """A more comprehensive log function that tracks advanced statistics
    related to gradients and parameters.

    This incurs additional memory cost to store extra variables. However,
    you can configure which metrics to track and reduce certain costs.

    Args:
        config: global_config.logging 

    Examples:
        >>> from loggings import full_log
        >>> logger = full_log(config.logging)
        >>> log_state = logger.init(params=...)
        >>> log_state, log_metrics = logger.update(log_state, ...)
    """

    log_callback_data = config.log_callback_data

    has_params_prev = config.store_last_params  # stores x(n-1)
    has_grads_prev = config.store_last_grads    # stores g'(n-1) = \nabla f(x(n-1), z(n))
    has_grads_hist = config.store_past_grads    # stores g(1:n-1), where g(n) = \nabla f(x(n), z(n))

    DEFAULT_METRICS = {
        "params/norm": jnp.zeros([]),
        "params/norm_l1": jnp.zeros([]),
        "params/norm_inf": jnp.zeros([]),

        "params/cos(xn, x(n-1))": jnp.zeros([]),

        "grads/norm": jnp.zeros([]),
        "grads/norm_l1": jnp.zeros([]),
        "grads/norm_inf": jnp.zeros([]),

        "grads/<gn, g(n-1)>": jnp.zeros([]),
        "grads/cos(gn, g(n-1))": jnp.zeros([]),

        "grads/<gn, g(1:n-1)>": jnp.zeros([]),
        "grads/cos(gn, g(1:n-1))": jnp.zeros([]),

        "grads/inf_grads": jnp.zeros([], jnp.int32),

        "update/norm": jnp.zeros([]),
        "update/norm_l1": jnp.zeros([]),
        "update/norm_inf": jnp.zeros([]),

        "update/<gn, xn-x(n-1)>": jnp.zeros([]),
        "update/<gn, xn-x(n-1)>_sum": jnp.zeros([]),

        "update/<gn, Delta(n)>": jnp.zeros([]),
        "update/<gn, Delta(n)>_sum": jnp.zeros([]),
        "update/cos(gn, Delta(n))": jnp.zeros([]),

        "update/<g(n-1), Delta(n)>": jnp.zeros([]),
        "update/<g(n-1), Delta(n)>_sum": jnp.zeros([]),
        "update/cos(g(n-1), Delta(n))": jnp.zeros([]),

        "update/cos(Delta(n+1), Delta(n))": jnp.zeros([]),

        "loss/fn-f(n-1)": jnp.zeros([]),
        "loss/fn-f(n-1)_sum": jnp.zeros([]),

        # TODO: add these metrics
        # "precond/cos(-gn, x*-xn)": jnp.zeros([]),
        # "precond/cos(Delta(n), x*-xn)": jnp.zeros([]),
    }

    def init_fn(params: optax.Params):
        """Initializes aux_state from confg."""
        if not log_callback_data:
            return None, {}
        return FullLogState(
            params_prev = params if has_params_prev else None,
            grads_prev = tree_utils.zeros_like(params) if has_grads_prev else None,
            grads_hist = tree_utils.zeros_like(params) if has_grads_hist else None,
            num_inf_grads = jnp.zeros([], jnp.int32),
            metrics_prev = DEFAULT_METRICS,
        ), DEFAULT_METRICS

    def update_fn(
            state: base.LogState,
            loss: Array,
            loss_prev: Array,
            params: optax.Params,
            grads: optax.Updates,
            updates: optax.Updates,
            random_scaling: Union[float, Array] = 1.0,
    ):
        """Updates log_state and compute log_metrics.
        
        Args:
            loss: f(x_n, z_n)
            loss_prev: f(x_(n-1), z_n)
            params: x_n
            grads: g(x_n, z_n)
            updates: Delta_(n+1), dependent of g(x_n, z_n)
        """
        if not log_callback_data:
            return None, {}
        
        # `apply_if_finite` wrapper.
        def reject_update(state):
            state = state._replace(
                num_inf_grads = optax.safe_int32_increment(state.num_inf_grads)
            )
            return state, state.metrics_prev
        
        def apply_update(state):
            # NOTE: please be aware that `updates` corresponds to Delta(n+1)
            # and `updates_prev` corresponds to Delta(n)
            
            zeros = tree_utils.zeros_like(params)
            params_prev = state.params_prev if has_params_prev else zeros
            grads_prev = state.grads_prev if has_grads_prev else zeros
            grads_hist = state.grads_hist if has_grads_hist else zeros
            num_inf_grads = state.num_inf_grads
            metrics_prev = state.metrics_prev

            params_diff = tree_utils.subtract(params, params_prev)
            updates_prev = tree_utils.scalar_dot(params_diff, 1/random_scaling)

            metrics = {
                "params/norm": tree_utils.norm(params, p=2),
                "params/norm_l1": tree_utils.norm(params, p=1),
                "params/norm_inf": tree_utils.norm(params, p="inf"),

                "params/cos(xn, x(n-1))": tree_utils.cosine(params, params_prev),

                "grads/norm": tree_utils.norm(grads, p=2),
                "grads/norm_l1": tree_utils.norm(grads, p=1),
                "grads/norm_inf": tree_utils.norm(grads, p="inf"),

                "grads/<gn, g(n-1)>": tree_utils.inner(grads, grads_prev),
                "grads/cos(gn, g(n-1))": tree_utils.cosine(grads, grads_prev),

                "grads/<gn, g(1:n-1)>": tree_utils.inner(grads, grads_hist),
                "grads/cos(gn, g(1:n-1))": tree_utils.cosine(grads, grads_hist),

                "grads/inf_grads": num_inf_grads,

                "update/norm": tree_utils.norm(updates_prev, p=2),              # norm of Delta(n), thus uses updates_prev
                "update/norm_l1": tree_utils.norm(updates_prev, p=1),
                "update/norm_inf": tree_utils.norm(updates_prev, p="inf"),

                "update/<gn, xn-x(n-1)>": tree_utils.inner(grads, params_diff),
                "update/<gn, xn-x(n-1)>_sum": metrics_prev["update/<gn, xn-x(n-1)>_sum"],

                "update/<gn, Delta(n)>": tree_utils.inner(grads, updates_prev),
                "update/<gn, Delta(n)>_sum": metrics_prev["update/<gn, Delta(n)>_sum"],
                "update/cos(gn, Delta(n))": tree_utils.cosine(grads, updates_prev),

                "update/<g(n-1), Delta(n)>": tree_utils.inner(grads_prev, updates_prev),
                "update/<g(n-1), Delta(n)>_sum": metrics_prev["update/<g(n-1), Delta(n)>_sum"],
                "update/cos(g(n-1), Delta(n))": tree_utils.cosine(grads_prev, updates_prev),

                "update/cos(Delta(n+1), Delta(n))": tree_utils.cosine(updates, updates_prev),

                "loss/fn-f(n-1)": loss-loss_prev,
                "loss/fn-f(n-1)_sum": metrics_prev["loss/fn-f(n-1)_sum"],

                # TODO: add these metrics
                # "precond/cos(-gn, x*-xn)": jnp.zeros([]),
                # "precond/cos(Delta(n), x*-xn)": jnp.zeros([]),
            }
            metrics["update/<gn, xn-x(n-1)>_sum"] += metrics["update/<gn, xn-x(n-1)>"]
            metrics["update/<gn, Delta(n)>_sum"] += metrics["update/<gn, Delta(n)>"]
            metrics["update/<g(n-1), Delta(n)>_sum"] += metrics["update/<g(n-1), Delta(n)>"]
            metrics["loss/fn-f(n-1)_sum"] += metrics["loss/fn-f(n-1)"]

            state = state._replace(
                params_prev = params if has_params_prev else None,
                grads_prev = grads if has_grads_prev else None,
                grads_hist = tree_utils.add(grads_hist, grads) if has_grads_hist else None,
                metrics_prev = metrics,
            )
            return state, metrics
        return jax.lax.cond(
            tree_utils.isfinite(grads), apply_update, reject_update, state,
        )
    return base.Logger(init_fn, update_fn)