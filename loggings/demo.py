"""A demo of different instances of LoggingFn."""

import jax
from jax import numpy as jnp
from jax import random as jr
from jax import tree_util as jtu
import optax
from typing import Optional, NamedTuple
from jaxtyping import Array, PyTree
from omegaconf import DictConfig
from utils import tree_utils
from loggings import base


class SimpleLogState(NamedTuple):
    grads_prev: optax.Updates
    params_prev: optax.Params
    cumulatives: PyTree


def simple_log() -> base.LogFn:
    """A very minimal log function.
    
    Examples:
        >>> from loggings import simple_log
        >>> log_fn = simple_log()
        >>> log_state = log_fn.init(params=...)
        >>> log_state, log_metrics = log_fn.update(log_state, loss_val=..., params=..., grads=...)
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
        return state
    
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
    
    return base.LogFn(init_fn, update_fn)


class FullLogState(NamedTuple):
    """full_log state."""
    params_prev: Optional[optax.Updates]        # x(n-1)
    grads_prev: Optional[optax.Updates]         # g(n-1)
    grads_hist: Optional[optax.Updates]         # g(1:n-1)
    random_scalar: Optional[Array]              # s_n
    importance_sampling: Optional[Array]        # w_n = [1-P(s)] / p(s)


def full_log(
        config: DictConfig,
) -> base.LogFn:
    """A more comprehensive log function that tracks advanced statistics
    related to gradients and parameters.

    This incurs additional memory cost to store extra variables. However,
    you can configure which metrics to track and reduce certain costs.

    Args:
        config: global_config.logging 

    Examples:
        >>> from loggings import full_log
        >>> log_fn = full_log(config.logging)
        >>> log_state = log_fn.init(params=...)
        >>> log_state, log_metrics = log_fn.update(log_state, ...)
    """

    has_params_prev = config.store_last_params  # stores x(n-1)
    has_grads_prev = config.store_last_grads    # stores g'(n-1) = \nabla f(x(n-1), z(n))
    has_grads_hist = config.store_past_grads    # stores g(1:n-1), where g(n) = \nabla f(x(n), z(n))

    def init_fn(params: optax.Params):
        """Initializes aux_state from confg."""
        if not config.log_callback_data:
            return None
        return FullLogState(
            params_prev = params if has_params_prev else None,
            grads_prev = tree_utils.zeros_like(params) if has_grads_prev else None,
            grads_hist = tree_utils.zeros_like(params) if has_grads_hist else None,
            random_scalar = jnp.ones([]),
            importance_sampling = jnp.ones([]),
        )
    
        opt_loggings = base.get_internal_logs(opt_state)
        if "update/random_scaling" not in opt_loggings:
            warnings.warn("Optimizer has no key named 'update/random_scaling,",
                        "and random scaling is default to one.",
                        "Wrap your optimizer with o2nc.wrap_random_scaling for correct logging.")
            random_scalar = jnp.ones([])
        else:
            random_scalar = opt_loggings["update/random_scaling"]
        if "update/importance_sampling" not in opt_loggings:
            warnings.warn("Optimizer has no key named 'update/importance_sampling,",
                        "and importance sampling is default to one.",
                        "Wrap your optimizer with o2nc.wrap_random_scaling for correct logging.")
            importance_sampling = jnp.ones([])
        else:
            importance_sampling = opt_loggings["update/importance_sampling"]
        loggings = {
            "grads/norm": jnp.zeros([]),
            "grads/l1-norm": jnp.zeros([]),
            "update/<gn, Delta(n)>": jnp.zeros([]),
            "update/<gn, Delta(n)>_sum": jnp.zeros([]),
            "update/<g(n-1), Delta(n)>": jnp.zeros([]),
            "update/<g(n-1), Delta(n)>_sum": jnp.zeros([]),
            "update/cos(g(n-1), Delta(n))": jnp.zeros([]),
            "update/wn*<gn, Delta(n)>": jnp.zeros([]),
            "update/wn*<gn, Delta(n)>_sum": jnp.zeros([]),
            "update/fn-f(n-1)": jnp.zeros([]),
            "update/fn-f(n-1)_sum": jnp.zeros([]),
            "update/<gn, xn-x(n-1)>": jnp.zeros([]),
            "update/<gn, xn-x(n-1)>_sum": jnp.zeros([]),
            "grads/<gn, g(n-1)>": jnp.zeros([]),
            "grads/<gn, g(1:n-1)>": jnp.zeros([]),
            "grads/cos(gn, g(n-1))": jnp.zeros([]),
            "grads/cos(gn, g(1:n-1))": jnp.zeros([]),
            "grads/inf_grads": jnp.zeros([], jnp.int32),
        }
        loggings.update(opt_loggings)
        zeros = tree_utils.zeros_like(eqx.filter(model, eqx.is_array))
        return FullLogState(
            params_diff = zeros if config.store_last_params else None,
            last_grads = zeros if config.store_last_grads else None,
            past_grads = zeros if config.store_past_grads else None,
            random_scalar = random_scalar,
            importance_sampling = importance_sampling,
            loggings = loggings,
        )

    def update_fn(
            state,
            loss_val: Array,
            loss_prev: Array,
            grads: PyTree,
            grads_prev: PyTree,
            updates: PyTree,
    ):
        """Updates aux_state. config: global config.
        Note: train_state.model uses new_model, i.e., x_(n+1).
        """
        if not config.log_callback_data:
            return None
        
        model = eqx.apply_updates(
            train_state.model, tree_utils.negative(updates))    # x_n
        opt_state = train_state.opt_state
        aux_state = train_state.aux_state
        dynamic_scaler_state = train_state.dynamic_scaler_state
        key, new_key = jr.split(train_state.train_key)
        batches = jnp.array(batches)
        
        base_loggings = {
            "grads/norm": tree_utils.norm(grads, p=2),
            "grads/l1-norm": tree_utils.norm(grads, p=1),
        }
        opt_loggings = utils.merge_dicts(*logstate.list_of_logs(opt_state))
        base_loggings.update(opt_loggings)

        def update_nan(state, base_loggings, dynamic_scaler_state):
            loggings = state.loggings
            loggings.update({
                "grads/inf_grads": optax.safe_int32_increment(loggings["grads/inf_grads"])
            })
            loggings.update(base_loggings)
            return state._replace(loggings=loggings), dynamic_scaler_state
        
        def update_finite(state, base_loggings, dynamic_scaler_state):
            loggings = state.loggings
            if config.store_last_params:
                inner_g_dx = utils.tree_inner_product(grads, state.params_diff)
                inner_g_Delta = inner_g_dx / state.random_scalar
                inner_g_wDelta = inner_g_Delta * state.importance_sampling
                loggings.update({
                    "update/<gn, xn-x(n-1)>": inner_g_dx,
                    "update/<gn, xn-x(n-1)>_sum": loggings["update/<gn, xn-x(n-1)>_sum"]+inner_g_dx,
                    "update/<gn, Delta(n)>": inner_g_Delta,
                    "update/<gn, Delta(n)>_sum": loggings["update/<gn, Delta(n)>_sum"]+inner_g_Delta,
                    "update/wn*<gn, Delta(n)>": inner_g_wDelta,
                    "update/wn*<gn, Delta(n)>_sum": loggings["update/wn*<gn, Delta(n)>_sum"]+inner_g_wDelta,
                })
            if config.store_last_params and config.store_last_grads:
                inner_g_last_Delta = utils.tree_inner_product(state.last_grads, state.params_diff)
                loggings.update({
                    "update/<g(n-1), Delta(n)>": inner_g_last_Delta,
                    "update/<g(n-1), Delta(n)>_sum": loggings["update/<g(n-1), Delta(n)>_sum"]+inner_g_last_Delta,
                    "update/cos(g(n-1), Delta(n))": utils.tree_cosine_similarity(state.last_grads, state.params_diff),
                })
            if config.store_last_params and config.compute_last_loss:
                last_model = eqx.apply_updates(
                    model, utils.negative_tree(state.params_diff))
                def compute_last_loss(i, val):
                    batch = batches[i]
                    if global_config.train.use_amp:
                        amp_loss_fn = amp(loss_fn, compute_dtype=get_dtype(global_config.train.precision))
                        last_loss_, _ = amp_loss_fn(last_model, batch, key=key)
                    else:
                        last_loss_, _ = loss_fn(last_model, batch, key=key)
                    return val + last_loss_
                last_loss = jax.lax.fori_loop(
                    0, len(batches), compute_last_loss, init_val=0
                )
                last_loss /= len(batches)   # average last loss over all batches
                df = loss - last_loss
                loggings.update({
                    "update/fn-f(n-1)": df,
                    "update/fn-f(n-1)_sum": loggings["update/fn-f(n-1)_sum"]+df,
                })
            if config.store_last_grads:
                loggings.update({
                    "grads/<gn, g(n-1)>": utils.tree_inner_product(grads, state.last_grads),
                    "grads/cos(gn, g(n-1))": utils.tree_cosine_similarity(grads, state.last_grads),
                })
            if config.store_past_grads:
                loggings.update({
                    "grads/<gn, g(1:n-1)>": utils.tree_inner_product(grads, state.past_grads),
                    "grads/cos(gn, g(1:n-1))": utils.tree_cosine_similarity(grads, state.past_grads),
                })
            loggings.update(base_loggings)
            if "update/random_scaling" in opt_loggings:
                random_scalar = opt_loggings["update/random_scaling"]
            else:
                random_scalar = state.random_scalar
            if "update/importance_sampling" in opt_loggings:
                importance_sampling = opt_loggings["update/importance_sampling"]
            else:
                importance_sampling = state.importance_sampling
            return state._replace(
                params_diff = updates if config.store_last_params else None,
                last_grads = grads if config.store_last_grads else None,
                past_grads = utils.tree_add(state.past_grads, grads) if config.store_past_grads else None,
                random_scalar = random_scalar,
                importance_sampling = importance_sampling,
                loggings = loggings,
            ), dynamic_scaler_state
        
        aux_state, dynamic_scaler_state = jax.lax.cond(
            tree_utils.isfinite(grads), update_finite, update_nan, aux_state, base_loggings, dynamic_scaler_state)
        
        return train_state._replace(
            dynamic_scaler_state = dynamic_scaler_state,
            train_key = new_key,
            aux_state = aux_state
        )
    return base.LogFn(init_fn, update_fn)