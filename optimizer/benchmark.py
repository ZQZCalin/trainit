"""Benchmark optimizers."""

import jax
from jax import numpy as jnp
from jax import random as jr
from jax import tree_util as jtu
import chex
import optax
from optax import Updates, Params, OptState, ScalarOrSchedule, GradientTransformation
from typing import Any, Tuple, NamedTuple, Optional, Callable
from online_learners import OnlineLearner, unconstrained_ogd
import sys
sys.path.append('../trainit')
import utils
from logger import RateLimitedWandbLog
import logstate
import online_learners as ol
import scheduler


class AdamWState(NamedTuple):
    """AdamW State."""
    count: chex.Array
    mu: Updates
    nu: Updates


def adamw(
    learning_rate: ScalarOrSchedule = 1e-4,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    weight_decay: float = 0.0,
    debias_beta1: bool = True,
    debias_beta2: bool = True,
) -> GradientTransformation:
    """AdamW for benchmark.

    Args:
        learning_rate (ScalarOrSchedule): _description_. Defaults to 1e-4.
        beta1 (float): _description_. Defaults to 0.9.
        beta2 (float): _description_. Defaults to 0.999.
        eps (float): _description_. Defaults to 1e-8.
        weight_decay (float): _description_. Defaults to 0.0.
        debias_beta1 (bool): Defaults to True.
        debias_beta2 (bool): Defaults to True.

    Returns:
        A `GradientTransformation` object.
    """

    def init_fn(params):
        return AdamWState(
            count=jnp.zeros([], jnp.int32),
            mu=jtu.tree_map(jnp.zeros_like, params),
            nu=jtu.tree_map(jnp.zeros_like, params)
        )
    
    def update_fn(updates, state, params):
        count_inc = optax.safe_int32_increment(state.count)
        mu = jtu.tree_map(
            lambda m, g: beta1*m + (1-beta1)*g, state.mu, updates)
        nu = jtu.tree_map(
            lambda v, g: beta2*v + (1-beta2)*g**2, state.nu, updates)
        # Debias to get the true weighted average.
        if debias_beta1:
            mu_hat = utils.tree_scalar_multiply(mu, 1/(1-beta1**count_inc))
        else:
            mu_hat = mu
        if debias_beta2:
            nu_hat = utils.tree_scalar_multiply(nu, 1/(1-beta2**count_inc))
        else:
            nu_hat = nu
        # Unpack learning rate schedule.
        eta = scheduler.get_current_lr(learning_rate, state.count)
        # Compute one-step update: -eta * [mu / (eps+sqrt(nu)) + lam * params]
        new_updates = jtu.tree_map(
            lambda m, v, p: -eta * (m/(eps+jnp.sqrt(v)) + weight_decay*p),
            mu_hat, nu_hat, params
        )
        return new_updates, AdamWState(
            count=count_inc, mu=mu, nu=nu)
    
    return GradientTransformation(init_fn, update_fn)


class SgdmState(NamedTuple):
    count: chex.Array
    momentum: optax.Updates


def sgdm(
    learning_rate: ScalarOrSchedule,
    beta: float=0.0,
    weight_decay: float=0.0,
) -> GradientTransformation:
    """SGD with momentum.
    
    Updates m_{t+1} = beta * m_t - (1-beta) * (g_t + mu*x_t)
        and x_{t+1} = x_t - eta_t * m_{t+1}, 
    where beta is the momentum constant and mu is the weight decay constant.

    Args:
        learning_rate: The learning rate scheduler.
        beta: The momentum constant in [0, 1]. Defaults to 0.
        weight_decay (float): The weight decay constant. Defaults to 0.

    Returns:
        A `GradientTransformation` object.
    """
    
    def init_fn(params):
        return SgdmState(
            count = jnp.zeros([], jnp.int32),
            momentum = jtu.tree_map(jnp.zeros_like, params),
        )
    
    def update_fn(updates, state, params):
        # TODO: which one to implement weight decay?
        # grads = jtu.tree_map(
        #     lambda g, x: g + mu*x, updates, params)
        eta = scheduler.get_current_lr(learning_rate, state.count)
        new_momentum = jtu.tree_map(
            lambda m, g: beta*m + (1-beta)*g, state.momentum, updates)
        new_updates = jtu.tree_map(
            lambda m, x: -eta * (m + weight_decay*x), new_momentum, params)
        return new_updates, SgdmState(
            count = optax.safe_int32_increment(state.count),
            momentum = new_momentum
        )
    
    return GradientTransformation(init_fn, update_fn)