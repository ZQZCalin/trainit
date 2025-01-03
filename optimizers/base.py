"""Base optimizers."""

import jax
from jax import numpy as jnp
from jax import random as jr
from jax import tree_util as jtu
import optax
from typing import Any, Tuple, NamedTuple, Optional, Callable, Union
from jaxtyping import Array, PyTree
from utils import tree_utils, log_utils
import optimizers.schedule as schedule


ScalarOrPytree = Union[float, PyTree]


class AdamWState(NamedTuple):
    """AdamW State."""
    count: Array
    mu: optax.Updates
    nu: optax.Updates
    logging: log_utils.Log


def adamw(
    learning_rate: optax.ScalarOrSchedule = 1e-4,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    weight_decay: ScalarOrPytree = 0.0,
    debias_beta1: bool = True,
    debias_beta2: bool = True,
    use_momentum: bool = True,
    use_preconditioning: bool = True,
    decouple_weight_decay: bool = False,
) -> optax.GradientTransformation:
    """AdamW for benchmark.

    Args:
        learning_rate (ScalarOrSchedule): _description_. Defaults to 1e-4.
        beta1 (float): _description_. Defaults to 0.9.
        beta2 (float): _description_. Defaults to 0.999.
        eps (float): _description_. Defaults to 1e-8.
        weight_decay (float): _description_. Defaults to 0.0.
        debias_beta1 (bool): Defaults to True.
        debias_beta2 (bool): Defaults to True.
        use_momentum (bool): Defaults to True. If false, replace \hat m_t with the gradients.
            However, m_t will still be compated based on beta1 and stored in the opt_state.
        use_preconditioning (bool): Defaults to True. If false, use \hat m_t as the update (without dividing by v_t).
            However, v_t will still be computed based on beta2 and stored in the opt_state.
        decouple_weight_decay (bool): Defaults to False. If true, learning rate eta will not be applied to weight_decay regularization.

    Returns:
        A `GradientTransformation` object.
    """

    use_pytree_wd = type(weight_decay) != float

    def init_fn(params):
        # Checks weight_decay structure during initialization.
        if use_pytree_wd and jtu.tree_structure(weight_decay)!=jtu.tree_structure(params):
            raise ValueError("structure of weight_decay must match model structure.")
        logging = {
            "optimizer/cos(g,m)": jnp.zeros([]),
            "optimizer/cos(g,m/sqrt(v))": jnp.zeros([]),
            "optimizer/cos(g,g/sqrt(v))": jnp.zeros([]),
        }
        return AdamWState(
            count=jnp.zeros([], jnp.int32),
            mu=jtu.tree_map(jnp.zeros_like, params),
            nu=jtu.tree_map(jnp.zeros_like, params),
            logging=log_utils.Log(logging),
        )
    
    def update_fn(updates, state, params):
        count_inc = optax.safe_int32_increment(state.count)
        mu = jtu.tree_map(
            lambda m, g: beta1*m + (1-beta1)*g, state.mu, updates)
        nu = jtu.tree_map(
            lambda v, g: beta2*v + (1-beta2)*g**2, state.nu, updates)
        
        # Debias to get the true weighted average.
        if debias_beta1:
            mu_hat = tree_utils.scalar_dot(mu, 1/(1-beta1**count_inc))
        else:
            mu_hat = mu
        if debias_beta2:
            nu_hat = tree_utils.scalar_dot(nu, 1/(1-beta2**count_inc))
        else:
            nu_hat = nu

        # Other optional features: turn off momentum and/or pre-conditioning.
        if not use_momentum:
            mu_hat = updates
        if not use_preconditioning:
            nu_hat = jtu.tree_map(jnp.ones_like, nu_hat)

        # Unpack learning rate schedule.
        eta = schedule.get_current_lr(learning_rate, state.count)

        # Weight decay regularization.
        if not use_pytree_wd:
            regularization = tree_utils.scalar_dot(params, weight_decay)
        else:
            regularization = tree_utils.multiply(params, weight_decay)
        if not decouple_weight_decay:
            regularization = tree_utils.scalar_dot(regularization, eta)

        # Compute one-step update: -eta * [mu / (eps+sqrt(nu)) + lam * params]
        new_updates = jtu.tree_map(
            lambda m, v, r: -(eta * m / (eps+jnp.sqrt(v)) + r),
            mu_hat, nu_hat, regularization 
        )

        # Additional logs.
        mu_hat = tree_utils.scalar_dot(mu, 1/(1-beta1**count_inc))
        nu_hat = tree_utils.scalar_dot(nu, 1/(1-beta2**count_inc))
        precond_mu = jtu.tree_map(lambda m, v: m / (eps+jnp.sqrt(v)), mu_hat, nu_hat)
        precond_g = jtu.tree_map(lambda g, v: g / (eps+jnp.sqrt(v)), updates, nu_hat)
        logging = {
            "optimizer/cos(g,m)": tree_utils.cosine(updates, mu),
            "optimizer/cos(g,m/sqrt(v))": tree_utils.cosine(updates, precond_mu),
            "optimizer/cos(g,g/sqrt(v))": tree_utils.cosine(updates, precond_g),
        }
        return new_updates, AdamWState(
            count=count_inc, mu=mu, nu=nu, logging=log_utils.Log(logging))
    
    return optax.GradientTransformation(init_fn, update_fn)


class SgdmState(NamedTuple):
    count: Array
    momentum: optax.Updates


def sgdm(
    learning_rate: optax.ScalarOrSchedule,
    beta: float=0.0,
    weight_decay: ScalarOrPytree=0.0,
) -> optax.GradientTransformation:
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
    
    # use_pytree_wd = type(weight_decay) != float
    use_pytree_wd = not isinstance(weight_decay, float)

    def init_fn(params):
        if use_pytree_wd and jtu.tree_structure(weight_decay)!=jtu.tree_structure(params):
            raise ValueError("structure of weight_decay must match model structure.")
        return SgdmState(
            count = jnp.zeros([], jnp.int32),
            momentum = jtu.tree_map(jnp.zeros_like, params),
        )
    
    def update_fn(updates, state, params):
        eta = schedule.get_current_lr(learning_rate, state.count)
        new_momentum = jtu.tree_map(
            lambda m, g: beta*m + (1-beta)*g, state.momentum, updates)
        if not use_pytree_wd:
            new_updates = jtu.tree_map(
                lambda m, x: -eta * (m + weight_decay*x), new_momentum, params)
        else:
            new_updates = jtu.tree_map(
                lambda m, x, wd: -eta * (m + wd*x), new_momentum, params, weight_decay)
        return new_updates, SgdmState(
            count = optax.safe_int32_increment(state.count),
            momentum = new_momentum
        )
    
    return optax.GradientTransformation(init_fn, update_fn)