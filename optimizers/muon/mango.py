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


normalization_list = [
    "l2",               # l2 norm for vectors or frobenius norm for matrices
    "l2_col",           # column-wise l2 norm for matrices
    "l2_split",         # head-wise l2 (frobenius) norm, for attention weights / bias
    "inf",              # inf norm for vectors or spectral norm for matrices
    "inf_col",          # column-wise inf norm for matrices
    "inf_split",        # head-wise inf (spectral) norm, for attention weights / bias
    "ns",               # newton-schulz for matrices
    "ns_split",         # head-wise newton-schulz for matrices, particularly attention weights
]


default_mango_normalizations = {
    "2d": "ns",
    "embedding": "l2_col",
    "head": "ns",
    "attn_w": "ns_split",
    "attn_b": "l2_split",
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

    # Manually specify GPT-2 configs.
    num_heads = 12

    # Gradient preconditioning by grad_squared.
    optim_grad_precond = scale_by_grad_squared(beta=beta2) if beta2 else optax.identity()

    # Standard momentum update.
    optim_momentum = optax.trace(decay=momentum, nesterov=nesterov)

    # Offset update.
    optim_offset = scale_by_offset(beta=offset_beta) if offset_beta else optax.identity()

    # Advanced normalization based on parameters.
    def split_vmap(f):
        """Broadcasts a function f: [d,:] -> [d,:] to a matrix/vector [3nd,:]
        in a way that first reshapes into [3,n,d,:], then applies f on [d,:],
        and finally reshape back into [3nd,:].
        """
        def split_fn(G):
            assert G.ndim == 1 or G.ndim == 2
            assert G.shape[0] % (3 * num_heads) == 0
            n = num_heads
            d = G.shape[0] // (3 * num_heads)
            # Reshape into [3,n,d,:].
            if G.ndim == 1:
                G = jnp.reshape(G, (3, n, d))
            else:
                G = jnp.reshape(G, (3, n, d, G.shape[1]))
            # Use nested vmap to broadcast mapping f to the last axes (d,:).
            G = jax.vmap(jax.vmap(f))(G)
            # Reshape back into [3nd,:].
            if G.ndim == 1:
                G = jnp.reshape(G, (3*n*d,))
            else:
                G = jnp.reshape(G, (3*n*d, G.shape[1]))
            return G
        return split_fn

    def scale_by_normalization(normalize):
        if normalize is None:
            return optax.identity()
        if normalize == "l2":
            return scale_by_function(
                f=lambda G: G / (jnp.linalg.norm(G) + eps)
            )
        if normalize == "l2_col":
            return scale_by_function(
                f=lambda G: G / (jnp.linalg.norm(G, axis=1, keepdims=True) + eps)
            )
        if normalize == "l2_split":
            return scale_by_function(split_vmap(
                f=lambda G: G / (jnp.linalg.norm(G) + eps)
            ))
        if normalize == "inf":
            return scale_by_function(
                f=lambda G: G / (jnp.linalg.norm(G, ord=jnp.inf) + eps)
            )
        if normalize == "inf_col":
            return scale_by_function(
                f=lambda G: G / (jnp.linalg.norm(G, ord=jnp.inf, axis=1, keepdims=True) + eps)
            )
        if normalize == "inf_split":
            return scale_by_function(split_vmap(
                f=lambda G: G / (jnp.linalg.norm(G, ord=jnp.inf) + eps)
            ))
        if normalize == "ns":
            return scale_by_newton_schulz(ns_steps=ns_steps)
        if normalize == "ns_split":
            def f(G):
                G = newton_schulz(G, steps=ns_steps)
                # Optional upscale by shape (muon line 135),
                # although it's always 1 for GPT-2 attn layers 
                # since d=64 << D=768.
                G = G * max(1, G.shape[0]/G.shape[1])**0.5
            return scale_by_function(split_vmap(f))
        raise ValueError(f"invalid normalization type = '{normalize}'.")

    if normalizations is None:
        optim_normalization = optax.identity()
    else:
        transforms = { k: scale_by_normalization(normalizations[k]) for k in mango_gpt_keys }
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