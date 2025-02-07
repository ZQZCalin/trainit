# An optax implementation of muon optimizer from:
# 
# https://github.com/KellerJordan/Muon
# 
# This implements a variant of muon optimzer
# by incorporating a few tweaks.

"""Muon optimizer."""

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import optax

from typing import NamedTuple, Optional, Callable
from jaxtyping import Array, PyTree

from utils import tree_utils
from optimizers.base import adamw
from optimizers.combine import multi_transform
from optimizers.schedule import get_current_lr
from optimizers.muon.muon import scale_by_muon


def newton_schulz(G: Array, steps: int) -> Array:
    """An approximate Newton-Schulz method.
    
    Adapted from:

    https://github.com/KellerJordan/Muon/blob/master/muon.py

    Given a matrix G with SVD decomposition SUV^T, this function
    approximates US'V^T where S' is diagonal with values Uniform(0.5, 1.5)
    without needing to compute any SVD decomposition.
    """
    assert G.ndim == 2

    # NOTE: do we need to adapt to bfloat16 as in the OG repo?
    a, b, c = (3.4445, -4.7750,  2.0315)
    eps = 1e-7

    X = G
    if G.shape[0] > G.shape[1]:
        X = X.T

    # wrap iterative update with jax.lax.fori_loop.
    X /= (jnp.linalg.norm(X, ord=2) + eps)
    def body_func(i, val):
        X = val
        A = X @ X.T
        B = b * A + c * A @ A
        return a * X + B @ X
    X = jax.lax.fori_loop(
        0, steps, body_func, X
    )

    if G.shape[0] > G.shape[1]:
        X = X.T
    return X


class ScaleByNewtonSchulzState(NamedTuple):
    """scale_by_newton_schulz state. An empty node."""


def scale_by_newton_schulz(
        ns_steps: int = 6,
) -> optax.GradientTransformation:
    """Normalize update by Newton-Schulz iterations.
    
    Only works for 2d matrix updates.
    """

    def init_fn(params=None):
        del params
        return ScaleByNewtonSchulzState()
    
    def update_fn(updates, state, params=None):
        del state, params
        updates = jtu.tree_map(
            lambda G: newton_schulz(G, steps=ns_steps), updates
        )
        # Additional scaling based on shape (see line 135).
        updates = jtu.tree_map(
            lambda G: G * max(1, G.shape[0]/G.shape[1])**0.5, updates
        )
        return updates, ScaleByNewtonSchulzState()
    
    return optax.GradientTransformation(init_fn, update_fn)


class ScaleByGradSquaredState(NamedTuple):
    """scale_by_grad_squared state."""
    grad_squared: optax.Updates


# NOTE: the current implementation doesn't use beta*v + (1-beta)*g**2
# to align with the observed experiment results
# This requires normalization after using this scale.
# TODO: test if the same optimal value holds with the average formula,
# and update the implementation accordingly.
def scale_by_grad_squared(
        beta: float = 0.95,
        eps: float = 1e-8,
) -> optax.GradientTransformation:
    """Scale incoming updates (usually gradients) by grad_squared.

    ```
    grad_squared = beta * grad_squared + (1-beta) * update**2
    ```
    
    Adapted from the LaProp optimizer:

    https://arxiv.org/abs/2002.04839

    """
    
    def init_fn(params):
        return ScaleByGradSquaredState(
            grad_squared = tree_utils.zeros_like(params)
        )

    def update_fn(updates, state, params=None):
        del params
        grad_squared = state.grad_squared
        grad_squared = jtu.tree_map(
            lambda v, g: beta * v + g**2, grad_squared, updates
        )
        updates = jtu.tree_map(
            lambda g, v: g / (jnp.sqrt(v) + eps), updates, grad_squared
        )
        return updates, ScaleByGradSquaredState(grad_squared=grad_squared)

    return optax.GradientTransformation(init_fn, update_fn)


class ScaleByNormalizationState(NamedTuple):
    """scale_by_normalization state. An empty node."""


def scale_by_normalization(
        normalize_fn: Callable[[Array], Array],
) -> optax.GradientTransformation:
    """Normalize update per layer."""

    def init_fn(params=None):
        del params
        return ScaleByNormalizationState()
    
    def update_fn(updates, state, params=None):
        del state, params
        updates = jtu.tree_map(normalize_fn, updates)
        return updates, ScaleByNormalizationState()
    
    return optax.GradientTransformation(init_fn, update_fn)


class ScaleByOffsetState(NamedTuple):
    """scale_by_offset state."""
    offset: optax.Updates


def scale_by_offset(
        beta: float = 0.99,
) -> optax.GradientTransformation:
    """The offset update.
    
    ```
    offset = beta * offset + (1-beta) * update
    update = update + offset
    ```
    """

    def init_fn(params):
        return ScaleByOffsetState(
            offset = tree_utils.zeros_like(params)
        )

    def update_fn(updates, state, params=None):
        del params
        offset = state.offset
        offset = jtu.tree_map(
            lambda o, u: beta * o + (1-beta) * u, offset, updates
        )
        updates = tree_utils.add(updates, offset)
        return updates, ScaleByOffsetState(offset=offset)
    
    return optax.GradientTransformation(init_fn, update_fn)


def label_gpt(params):
    def fn(path, p):
        parts = [part.name for part in path if isinstance(part, jtu.GetAttrKey)]
        if "token_embedding" in parts or "position_embedding" in parts:
            return "embedding"
        if p.ndim == 2:
            return "2d"
        if p.ndim == 1 and "weight" in parts:
            return "1d_mul"
        if p.ndim == 1 and "bias" in parts:
            return "1d_add"
        raise ValueError(f"cannot categorize parameter: {p}")
    return jtu.tree_map_with_path(fn, params) 


def muon_laprop(
        learning_rate: optax.ScalarOrSchedule = 0.05,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 6,
        eps: float = 1e-8,
        lr_1d: optax.ScalarOrSchedule = 3e-4,
        beta2: Optional[float] = None,
        offset_beta: Optional[float] = None,
) -> optax.GradientTransformation:
    """The muon-laprop optimizer.
    
    Applies muon update on suitable parameters and
    applies adam update on the rest.

    We use `optax.multi_transform` to combine these updates.

    Args:
        learning_rate: muon learning rate.
        momentum: sgd momentum of muon.
        nesterov: whether to use nesterov momentum.
        ns_steps: number of steps of Newton-Schulz.
        adam_lr: adam learning rate.
        adam_beta1: adam beta1.
        adam_beta2: adam beta2.
        adam_eps: adam eps.
        adam_wd: adam weight decay.
    """

    optim_2d = optax.chain(
        scale_by_newton_schulz(ns_steps=ns_steps),
        optax.scale_by_learning_rate(learning_rate),
    )

    # Embedding layers have shape [V,D] or [L,D],
    # so we normalize along axis=1.
    optim_embedding = optax.chain(
        scale_by_normalization(
            normalize_fn=lambda G: G / (jnp.linalg.norm(G, axis=1, keepdims=True) + eps),
        ),
        optax.scale_by_learning_rate(lr_1d),
    )

    # Normalize by inf-norm.
    optim_1d_mul = optax.chain(
        scale_by_normalization(
            normalize_fn=lambda G: G / (jnp.linalg.norm(G, ord=jnp.inf) + eps),
        ),
        optax.scale_by_learning_rate(lr_1d),
    )

    # Normalize by l2-norm.
    optim_1d_add = optax.chain(
        scale_by_normalization(
            normalize_fn=lambda G: G / (jnp.linalg.norm(G) + eps),
        ),
        optax.scale_by_learning_rate(lr_1d),
    )

    transforms = {
        "embedding": optim_embedding,
        "2d": optim_2d,
        "1d_mul": optim_1d_mul,
        "1d_add": optim_1d_add,
    }

    return optax.chain(
        scale_by_grad_squared(beta=beta2) if beta2 else optax.identity(),
        optax.trace(decay=momentum, nesterov=nesterov),
        multi_transform(transforms, label_gpt),
        # NOTE: currently offset update is after lr update.
        scale_by_offset(beta=offset_beta) if offset_beta else optax.identity(),
    )


def muon_adamw(
        learning_rate: optax.ScalarOrSchedule = 0.05,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 6,
        eps: float = 1e-8,
        beta2: Optional[float] = None,
        offset_beta: Optional[float] = None,
        adam_lr: optax.ScalarOrSchedule = 3e-4,
        adam_beta1: float = 0.95,
        adam_beta2: float = 0.95,
        adam_eps: float = 1e-8,
        adam_wd: float = 0.0,
) -> optax.GradientTransformation:
    """The muon-adamw optimizer.
    
    This recreates the baseline using laprop-muon on 2d arrays, and adamw on 1d arrays without normalization.
    """

    optim_muon = optax.chain(
        scale_by_grad_squared(beta2, eps) if beta2 else optax.identity(),
        scale_by_muon(learning_rate, momentum, nesterov, ns_steps),
        scale_by_offset(offset_beta) if offset_beta else optax.identity(),
    )
    optim_adamw = optax.chain(
        adamw(
            learning_rate=adam_lr,
            beta1=adam_beta1,
            beta2=adam_beta2,
            eps=adam_eps,
            weight_decay=adam_wd,
            use_nesterov=False,
        ),
        scale_by_offset(offset_beta) if offset_beta else optax.identity(),
    )
    transforms = {
        "2d": optim_muon,
        "1d": optim_adamw,
    }
    def label_params(params):
        return jtu.tree_map(
            lambda p: "2d" if p.ndim == 2 else "1d", params
        )
    return multi_transform(transforms, label_params)
