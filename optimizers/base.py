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


class AdamBaseState(NamedTuple):
    """adam_base state."""
    count: Array
    mu: Optional[optax.Updates]
    nu: Optional[optax.Updates]
    log_state: Optional[LogState]
    log_metrics: Optional[log_utils.Log]


def adam_base(
        learning_rate: optax.ScalarOrSchedule = 1e-4,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        decouple_weight_decay: bool = False,
        use_nesterov: bool = False,
        debias_beta1: bool = True,
        debias_beta2: bool = True,
        use_momentum: bool = True,
        use_momentum_state: bool = True,
        use_precond: bool = True,
        use_precond_state: bool = True,
        use_constant_wd: bool = False,
        logger: Optional[Logger] = None,
) -> optax.GradientTransformation:
    """The base Adam optimizer.

    Implements the Adam optimizer from:

    https://arxiv.org/pdf/1412.6980

    This is the base implementation that will be derived into variants
    including SGD, SGDM, and RMSProp.

    `decouple_weight_decay` controls whether to update as Adam (wd before momentum) 
    or AdamW (wd after momentum), where AdamW is adapted from:

    https://arxiv.org/pdf/1711.05101

    `use_nesterov` controls whether to apply nesterov's accelerated gradient to update,
    which is adapted from this report:

    https://openreview.net/pdf/OM0jvwB8jIp57ZJjtNEZ.pdf
    https://cs229.stanford.edu/proj2015/054_report.pdf

    If users need to apply different lr or wd to different subgroups of the model,
    please use `optax.multi_transform`.

    Args:
        learning_rate: learning rate schedule.
        beta1: momentum constant.
        beta2: momentum constant of adaptive gradients.
        eps: stability constant.
        weight_decay: weight decay.
        decouple_weight_decay: if true, uses adamw update, i.e., apply weight decay after momentum aggregation.
        use_nesterov: if true, uses NAG update, i.e., slighly overshoots the momentum update with current gradient.
        debias_beta1 (optional): if true, uses unbiased EMA of momentum. Defaults to True.
        debias_beta2 (optional): if true, uses unbiased EMA of gradient preconditioner. Defaults to True.
        use_momentum (optional): If false, replace \hat m_t with g_t.
        use_momentum_state (optional): If false, does not store m_t in the opt_state.
        use_precond (optional): If false, does not scale \hat m_t by the adaptive gradient pre-conditioner v_t.
        use_precond_state (optional): if false, does not store v_t in the opt_state.
        use_constant_wd (optional): an experimental parameter. if true, weight decay does not scale with learning rate schedule. Defaults to False.
        logger: logging callback, a `loggers.Logger` object that initializes and updates log_state and log_metrics.
        
    Returns:
        A `GradientTransformation` object.
    """

    wd = weight_decay
    if use_momentum:
        use_momentum_state = True
    if use_precond:
        use_precond_state = True

    def init_fn(params):
        if logger is not None:
            log_state, log_metrics = logger.init(params=params)
            log_metrics = log_utils.Log(log_metrics)
        else:
            log_state, log_metrics = (None, None)
        return AdamBaseState(
            count=jnp.zeros([], jnp.int32),
            mu=jtu.tree_map(jnp.zeros_like, params) if use_momentum_state else None,
            nu=jtu.tree_map(jnp.zeros_like, params) if use_precond_state else None,
            log_state=log_state,
            log_metrics=log_metrics,
        )
    
    def update_fn(updates, state, params):
        count = state.count
        mu = state.mu
        nu = state.nu

        count_inc = optax.safe_int32_increment(count)

        # Apply coupled weight decay
        if not decouple_weight_decay:
            updates = jtu.tree_map(
                lambda g, p: g + wd * p, updates, params)

        if use_momentum_state:
            mu = jtu.tree_map(
                lambda m, g: beta1*m + (1-beta1)*g, mu, updates)
            if use_momentum:
                if use_nesterov:
                    # Apply nesterov's momentum.
                    count_inc2 = optax.safe_int32_increment(count_inc)
                    mu_hat = jtu.tree_map(
                        lambda m, g: beta1*m/(1-beta1**count_inc2) + (1-beta1)*g/(1-beta1**count_inc),
                        mu, updates)
                else:
                    mu_hat = tree_utils.scalar_dot(mu, 1/(1-beta1**count_inc))
            else:
                mu_hat = updates
        else:
            mu = None
            mu_hat = updates

        if use_precond_state:
            nu = jtu.tree_map(
                lambda v, g: beta2*v + (1-beta2)*g**2, nu, updates)
            if use_precond:
                nu_hat = tree_utils.scalar_dot(nu, 1/(1-beta2**count_inc))
            else:
                nu_hat = None
        else:
            nu = None
            nu_hat = None

        # mu = jtu.tree_map(
        #     lambda m, g: beta1*m + (1-beta1)*g, state.mu, updates)
        # nu = jtu.tree_map(
        #     lambda v, g: beta2*v + (1-beta2)*g**2, state.nu, updates)
        
        # # Debias EMA of momentums.
        # debias_scaler1 = 1/(1-beta1**count_inc) if debias_beta1 else 1.0
        # debias_scaler2 = 1/(1-beta2**count_inc) if debias_beta2 else 1.0
        # mu_hat = tree_utils.scalar_dot(mu, debias_scaler1)
        # nu_hat = tree_utils.scalar_dot(nu, debias_scaler2)

        # Compute one-step update: -eta * [mu / (eps+sqrt(nu)) + lam * params]
        if nu_hat is not None:
            updates = jtu.tree_map(
                lambda m, v: m / (jnp.sqrt(v) + eps), mu_hat, nu_hat)
        else:
            updates = mu_hat

        # Apply decoupled weight decay.
        if decouple_weight_decay:
            updates = jtu.tree_map(
                lambda u, p: u + wd * p, updates, params)

        # Apply negative learning rate.
        eta = schedule.get_current_lr(learning_rate, count)
        updates = tree_utils.scalar_dot(updates, -eta)

        # Additional logs.
        # mu_hat = tree_utils.scalar_dot(mu, 1/(1-beta1**count_inc))
        # nu_hat = tree_utils.scalar_dot(nu, 1/(1-beta2**count_inc))
        # precond_mu = jtu.tree_map(lambda m, v: m / (eps+jnp.sqrt(v)), mu_hat, nu_hat)
        # precond_g = jtu.tree_map(lambda g, v: g / (eps+jnp.sqrt(v)), updates, nu_hat)
        # logging = {
        #     "optimizer/cos(g,m)": tree_utils.cosine(updates, mu),
        #     "optimizer/cos(g,m/sqrt(v))": tree_utils.cosine(updates, precond_mu),
        #     "optimizer/cos(g,g/sqrt(v))": tree_utils.cosine(updates, precond_g),
        # }
        
        # Logger callback.
        if logger is not None:
            log_state, log_metrics = logger.update(
                log_state,
                # pass extra args to your logger if needed.
            )
            log_metrics = log_utils.Log(log_metrics)
        else:
            log_state, log_metrics = (None, None)

        return updates, AdamBaseState(
            count=count_inc,
            mu=mu,
            nu=nu,
            log_state=log_state,
            log_metrics=log_metrics,
        )
    
    return optax.GradientTransformation(init_fn, update_fn)


def adam(
        learning_rate: optax.ScalarOrSchedule = 1e-4,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        logger: Optional[Logger] = None,
    ) -> optax.GradientTransformation:
    """The Adam optimizer from
    
    https://arxiv.org/pdf/1412.6980

    Does not apply nesterov's momentum nor decoupled weight decay.
    Use nadam or adamw instead for those updates.
    """
    return adam_base(
        learning_rate=learning_rate,
        beta1=beta1,
        beta2=beta2,
        eps=eps,
        weight_decay=weight_decay,
        decouple_weight_decay=False,
        use_nesterov=False,
        debias_beta1= True,
        debias_beta2=True,
        use_momentum=True,
        use_momentum_state=True,
        use_precond=True,
        use_precond_state=True,
        use_constant_wd=False,
        logger=logger,
    )


def adamw(
        learning_rate: optax.ScalarOrSchedule = 1e-4,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        use_nesterov: bool = False,
        logger: Optional[Logger] = None,
) -> optax.GradientTransformation:
    """Implements the AdamW optimizer from
    
    https://arxiv.org/pdf/1711.05101

    Nesterov's momentum is default to false, but 
    you can manually turn it on.
    """
    return adam_base(
        learning_rate=learning_rate,
        beta1=beta1,
        beta2=beta2,
        eps=eps,
        weight_decay=weight_decay,
        decouple_weight_decay=True,
        use_nesterov=use_nesterov,
        debias_beta1= True,
        debias_beta2=True,
        use_momentum=True,
        use_momentum_state=True,
        use_precond=True,
        use_precond_state=True,
        use_constant_wd=False,
        logger=logger,
    )


def nadam(
        learning_rate: optax.ScalarOrSchedule = 1e-4,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        decouple_weight_decay: bool = True,
        logger: Optional[Logger] = None,
) -> optax.GradientTransformation:
    """Implements the Nadam optimizer from
    
    https://cs229.stanford.edu/proj2015/054_report.pdf

    Decoupled weight decay is default to true due to the belief
    that adamw update is more suitable for larger models.
    However, you can manually turn it off.
    """
    return adam_base(
        learning_rate=learning_rate,
        beta1=beta1,
        beta2=beta2,
        eps=eps,
        weight_decay=weight_decay,
        decouple_weight_decay=decouple_weight_decay,
        use_nesterov=True,
        debias_beta1= True,
        debias_beta2=True,
        use_momentum=True,
        use_momentum_state=True,
        use_precond=True,
        use_precond_state=True,
        use_constant_wd=False,
        logger=logger,
    )


def rmsprop(
        learning_rate: optax.ScalarOrSchedule = 1e-4,
        momentum: float = 0.99,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        decouple_weight_decay: bool = True,
        logger: Optional[Logger] = None,
    ) -> optax.GradientTransformation:
    """The RMSProp optimizer from
    
    https://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf

    Decoupled weight decay is default to true.
    """
    return adam_base(
        learning_rate=learning_rate,
        beta2=momentum,
        eps=eps,
        weight_decay=weight_decay,
        decouple_weight_decay=decouple_weight_decay,
        use_nesterov=False,
        debias_beta1= True,
        debias_beta2=True,
        use_momentum=False,
        use_momentum_state=False,
        use_precond=True,
        use_precond_state=True,
        use_constant_wd=False,
        logger=logger,
    )


def sgdm(
        learning_rate: optax.ScalarOrSchedule = 1e-4,
        momentum: float = 0.9,
        weight_decay: float = 0.0,
        decouple_weight_decay: bool = True,
        use_nesterov: bool = False,
        logger: Optional[Logger] = None,
) -> optax.GradientTransformation:
    """The SGD-Momentum optimizer.
    
    Decoupled weight decay is default to true,
    and nesterov's momentum is default to false.

    Note: the implementation of SGD-momentum is slightly different
    from classical polyak momentum notation of decaying sum where

    :math:`\mu_t = \beta * \mu_{t-1} + g_t`.

    Instead, we compute the momentum as exponential average,
    which is the same as adam, where

    :math:`\mu_t = \beta * \mu_{t-1} + g_t`

    and later debiased as

    :math:`\hat \mu_t = \mu_t / (1+\beta**t)`.

    Nesterov's momentum is computed in the same way as Nadam, see:

    https://cs229.stanford.edu/proj2015/054_report.pdf
    """
    return adam_base(
        learning_rate=learning_rate,
        beta1=momentum,
        weight_decay=weight_decay,
        decouple_weight_decay=decouple_weight_decay,
        use_nesterov=use_nesterov,
        debias_beta1= True,
        debias_beta2=True,
        use_momentum=True,
        use_momentum_state=True,
        use_precond=False,
        use_precond_state=False,
        use_constant_wd=False,
        logger=logger,
    )


def sgd(
        learning_rate: optax.ScalarOrSchedule = 1e-4,
        weight_decay: float = 0.0,
        logger: Optional[Logger] = None,
) -> optax.GradientTransformation:
    """The vanilla SGD optimizer."""
    return adam_base(
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        use_momentum=False,
        use_momentum_state=False,
        use_precond=False,
        use_precond_state=False,
        use_constant_wd=False,
        logger=logger,
    )


# NOTE: we will use adam_base to derive any related optimizers including SGDM.
# class SgdmState(NamedTuple):
#     count: Array
#     momentum: optax.Updates


# def sgdm(
#     learning_rate: optax.ScalarOrSchedule,
#     beta: float=0.0,
#     weight_decay: ScalarOrPytree=0.0,
# ) -> optax.GradientTransformation:
#     """SGD with momentum.
    
#     Updates m_{t+1} = beta * m_t - (1-beta) * (g_t + mu*x_t)
#         and x_{t+1} = x_t - eta_t * m_{t+1}, 
#     where beta is the momentum constant and mu is the weight decay constant.

#     Args:
#         learning_rate: The learning rate scheduler.
#         beta: The momentum constant in [0, 1]. Defaults to 0.
#         weight_decay (float): The weight decay constant. Defaults to 0.

#     Returns:
#         A `GradientTransformation` object.
#     """
    
#     # use_pytree_wd = type(weight_decay) != float
#     use_pytree_wd = not isinstance(weight_decay, float)

#     def init_fn(params):
#         if use_pytree_wd and jtu.tree_structure(weight_decay)!=jtu.tree_structure(params):
#             raise ValueError("structure of weight_decay must match model structure.")
#         return SgdmState(
#             count = jnp.zeros([], jnp.int32),
#             momentum = jtu.tree_map(jnp.zeros_like, params),
#         )
    
#     def update_fn(updates, state, params):
#         eta = schedule.get_current_lr(learning_rate, state.count)
#         new_momentum = jtu.tree_map(
#             lambda m, g: beta*m + (1-beta)*g, state.momentum, updates)
#         if not use_pytree_wd:
#             new_updates = jtu.tree_map(
#                 lambda m, x: -eta * (m + weight_decay*x), new_momentum, params)
#         else:
#             new_updates = jtu.tree_map(
#                 lambda m, x, wd: -eta * (m + wd*x), new_momentum, params, weight_decay)
#         return new_updates, SgdmState(
#             count = optax.safe_int32_increment(state.count),
#             momentum = new_momentum
#         )
    
#     return optax.GradientTransformation(init_fn, update_fn)