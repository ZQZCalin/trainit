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

    # General 2d matrices.
    optim_2d = optax.chain(
        scale_by_newton_schulz(ns_steps=ns_steps),
        optax.scale_by_learning_rate(learning_rate),
    )

    # Embedding layers have shape [V,D] or [L,D],
    # so we normalize along axis=1.
    optim_embedding = optax.chain(
        scale_by_function(
            f=lambda G: G / (jnp.linalg.norm(G, axis=1, keepdims=True) + eps),
        ),
        # optax.scale_by_learning_rate(lr_1d),
        # change this to 2d array learning rates
        optax.scale_by_learning_rate(learning_rate),
    )

    # Normalize by inf-norm.
    optim_1d_mul = optax.chain(
        scale_by_function(
            f=lambda G: G / (jnp.linalg.norm(G, ord=jnp.inf) + eps),
        ),
        optax.scale_by_learning_rate(lr_1d),
    )

    # Normalize by l2-norm.
    optim_1d_add = optax.chain(
        scale_by_function(
            f=lambda G: G / (jnp.linalg.norm(G) + eps),
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
