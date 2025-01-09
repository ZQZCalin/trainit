"""Base optimizers."""

import jax
from jax import numpy as jnp
from jax import random as jr
from jax import tree_util as jtu
import optax
from typing import Any, Tuple, NamedTuple, Optional, Callable, Union
from jaxtyping import Array, PyTree
from utils import tree_utils, log_utils
from loggers import Logger, LogState
import optimizers.schedule as schedule


ScalarOrPytree = Union[float, PyTree]


class AdamState(NamedTuple):
    """Adam State."""
    count: Array
    mu: Optional[optax.Updates]
    nu: Optional[optax.Updates]
    log_state: Optional[LogState]
    logging: Optional[log_utils.Log]


def adam(
    learning_rate: optax.ScalarOrSchedule = 1e-4,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    weight_decay: float = 0.0,
    decouple_weight_decay: bool = True,
    debias_beta1: bool = True,
    debias_beta2: bool = True,
    use_momentum: bool = True,
    use_momentum_state: bool = True,
    use_precond: bool = True,
    use_precond_state: bool = True,
    use_constant_wd: bool = False,
) -> optax.GradientTransformation:
    """The Adam optimizer.

    Implements the Adam optimizer from:

    https://arxiv.org/pdf/1412.6980

    This is the base implementation that will be derived into variants
    including SGD, SGDM, and RMSProp.

    `decouple_weight_decay` controls whether to update as Adam (wd before momentum) 
    or AdamW (wd after momentum), where AdamW is implemented from:

    https://arxiv.org/pdf/1711.05101

    If users need to apply different lr or wd to different subgroups of the model,
    please use `optax.multi_transform`.

    Args:
        learning_rate: learning rate schedule.
        beta1: momentum constant.
        beta2: momentum constant of adaptive gradients.
        eps: stability constant.
        weight_decay: weight decay.
        decouple_weight_decay: Defaults to False. If true, learning rate eta will not be applied to weight_decay regularization.
        debias_beta1: if true, uses unbiased EMA of momentum. Defaults to True.
        debias_beta2: if true, uses unbiased EMA of gradient preconditioner. Defaults to True.
        use_momentum: If false, replace \hat m_t with g_t.
        use_momentum_state: If false, does not store m_t in the opt_state.
        use_precond: If false, does not scale \hat m_t by the adaptive gradient pre-conditioner v_t.
        use_precond_state: if false, does not store v_t in the opt_state.
        use_constant_wd: an experimental parameter. if true, weight decay does not scale with learning rate schedule. Defaults to False.

    Returns:
        A `GradientTransformation` object.
    """

    use_pytree_wd = type(weight_decay) != float
    if use_momentum:
        use_momentum_state = True
    if use_precond:
        use_precond_state = True

    def init_fn(params):
        # Checks weight_decay structure during initialization.
        if use_pytree_wd and jtu.tree_structure(weight_decay)!=jtu.tree_structure(params):
            raise ValueError("structure of weight_decay must match model structure.")
        logging = {
            "optimizer/cos(g,m)": jnp.zeros([]),
            "optimizer/cos(g,m/sqrt(v))": jnp.zeros([]),
            "optimizer/cos(g,g/sqrt(v))": jnp.zeros([]),
        }
        return AdamState(
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
        return new_updates, AdamState(
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