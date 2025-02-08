"""Common scale functions for muon variants."""

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import optax

from typing import NamedTuple, Callable
from jaxtyping import Array, PyTree

from utils import tree_utils


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


class ScaleByFunctionState(NamedTuple):
    """scale_by_function state. An empty node."""


def scale_by_function(
        f: Callable[[Array], Array],
) -> optax.GradientTransformation:
    """Apply mapping f to update per layer."""

    def init_fn(params=None):
        del params
        return ScaleByFunctionState()
    
    def update_fn(updates, state, params=None):
        del state, params
        updates = jtu.tree_map(f, updates)
        return updates, ScaleByFunctionState()
    
    return optax.GradientTransformation(init_fn, update_fn)