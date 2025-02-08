"""Mango optimizer."""

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import optax

from typing import NamedTuple, Optional, Callable, Literal, Dict
from jaxtyping import Array, PyTree

from utils import tree_utils
from optimizers.base import adamw
from optimizers.combine import multi_transform
from optimizers.schedule import get_current_lr
from optimizers.muon.muon import scale_by_muon
from optimizers.muon.base import (
    newton_schulz,
    scale_by_newton_schulz,
    scale_by_grad_squared,
    scale_by_function,
    scale_by_offset,
)


mango_gpt_keys = ["2d", "embedding", "head", "attn_w", "attn_b", "1d_w", "1d_b"]


default_mango_normalizations = {
    "2d": "ns",
    "embedding": "l2",
    "head": "ns",
    "attn_w": "ns",
    "attn_b": "ns",
    "1d_w": "inf",
    "1d_b": "l2",
}


def mango_label_gpt(params):
    def fn(path, p):
        parts = [part.name for part in path if isinstance(part, jtu.GetAttrKey)]
        # Special ararys.
        if "token_embedding" in parts or "position_embedding" in parts:
            return "embedding"
        if "head" in parts:
            return "head"
        if "attn_fc" in parts and p.ndim == 2:
            return "attn_w"
        if "attn_fc" in parts and p.ndim == 1:
            return "attn_b"
        # General arrays.
        if p.ndim == 2:
            return "2d"
        if p.ndim == 1 and "weight" in parts:
            return "1d_w"
        if p.ndim == 1 and "bias" in parts:
            return "1d_b"
        raise ValueError(f"cannot categorize parameter: {p}")
    return jtu.tree_map_with_path(fn, params) 


def scale_by_normalization(
        normalization: Literal["l2", "inf", "ns"] | None,
        eps: float = 1e-8,
        axis: int | None = None,
        ns_steps: int | None = None,
) -> optax.GradientTransformation:
    """Normalize update per layer.

    A wrapper function that wraps different normalization methods for mango.

    * NOTE: we do not check dimension for users.
    
    Args:
        normalization:
            - "l2": l2 norm on vectors and frobenius norm on matrices;
            - "inf": inf norm on vectors and spectral norm on matrices;
            - "ns": newton-schulz on matrices only;
            - None: no normalization at all
        eps: numerical stability constant
        axis: specifies which axis to normalize if "l2" or "inf"
        ns_steps: number of newton-schulz if "ns"
    """
    if normalization is None:
        return optax.identity()
    if normalization == "ns":
        return scale_by_newton_schulz(ns_steps=ns_steps)
    if normalization == "l2":
        return scale_by_function(
            f=lambda G: G / (jnp.linalg.norm(G, axis=axis, keepdims=True) + eps)
        )
    if normalization == "inf":
        return scale_by_function(
            f=lambda G: G / (jnp.linalg.norm(G, ord=jnp.inf, axis=axis, keepdims=True) + eps)
        )
    raise ValueError(f"invalid normalization type = '{normalization}'.")


def mango(
        base_lr: float = 0.05,
        schedule: optax.Schedule | None = None,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 6,
        eps: float = 1e-8,
        beta2: float | None = None,
        offset_beta: float | None = None,
        lrs: Dict[str, float] | None = None,
        normalizations: Dict[str, str | None] | None = default_mango_normalizations,
) -> optax.GradientTransformation:
    """Mango (Momentum with Advanced Normalization, Gradient-preconditing and Offset update).
    
    Args:
        
    """
    # Gradient preconditioning by grad_squared.
    optim_grad_precond = scale_by_grad_squared(beta=beta2) if beta2 else optax.identity()

    # Standard momentum update.
    optim_momentum = optax.trace(decay=momentum, nesterov=nesterov)

    # Offset update.
    optim_offset = scale_by_offset(beta=offset_beta) if offset_beta else optax.identity()

    # Advanced normalization based on parameters.
    def get_normalization(key):
        normalization = normalizations[key]
        if normalization is None:
            return optax.identity()
        if normalization == "ns":
            return scale_by_newton_schulz(ns_steps=ns_steps)
        if normalization == "l2":
            return scale_by_function(
                f=lambda G: G / (jnp.linalg.norm(G, axis=axis, keepdims=True) + eps)
            )
        if normalization == "inf":
            return scale_by_function(
                f=lambda G: G / (jnp.linalg.norm(G, ord=jnp.inf, axis=axis, keepdims=True) + eps)
            )
        raise ValueError(f"invalid normalization type = '{normalization}'.")

    if normalizations is None:
        optim_normalization = optax.identity()
    else:
        transforms = { k: get_normalization(k) for k in mango_gpt_keys }
        optim_normalization = multi_transform(transforms, mango_label_gpt)

    # Advanced learning rate schedules based on parameters.
    if lrs is None:
        learning_rate = base_lr if schedule is None else lambda t: base_lr * schedule(t)
        optim_schedule = optax.scale_by_learning_rate(learning_rate)
    else:
        if schedule is None:
            lr_transforms = { k: optax.scale_by_learning_rate(lrs[k]) for k in mango_gpt_keys }
        else:
            lr_transforms = { k: optax.scale_by_learning_rate(
                lambda t: lrs[k] * schedule(t, log_callback=(k=="2d"))  # we use a wrapped schedule that logs to wandb
            ) for k in mango_gpt_keys }
        optim_schedule = multi_transform(lr_transforms, mango_label_gpt)

    return optax.chain(
        optim_grad_precond,
        optim_momentum,
        optim_normalization,
        optim_schedule,
        optim_offset,
    )