"""Adamw-related optimizers."""

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import optax

from typing import NamedTuple, List, Tuple, Callable, Protocol, Any
from jaxtyping import Array, PyTree
from functools import partial
import warnings

from utils import tree_utils
from optimizers.combine import multi_transform
from optimizers.schedule import get_current_lr
from optimizers.muon.base import newton_schulz, LabelParamsFn, scale_by_offset
from optimizers.muon.mango import normalize_with_grad_squared, scale_by_function


class VisualizeRmsState(NamedTuple):
    """An empty node for visualize_rms state."""


def visualize_rms(
        wandb_logger: Any | None = None,
) -> optax.GradientTransformation:
    """Visualize RMS norm of updates.
    
    Does not affect updates.
    """
    def parse_path(path):
        parts = []
        for part in path:
            if isinstance(part, jtu.GetAttrKey):
                parts.append(part.name)
            elif isinstance(part, jtu.SequenceKey):
                parts[-1] += f"[{part.idx}]"
        return ".".join(parts)
    
    def rms_norm(G: Array):
        if G.ndim == 1:
            return jnp.linalg.norm(G) / len(G)**0.5
        if G.ndim == 2:
            return jnp.linalg.norm(G) / (G.shape[0]*G.shape[1])**0.5
    
    def log_norm(tree: optax.Updates, prefix: str):
        logs = {}
        for path, arr in jtu.tree_leaves_with_path(tree):
            logs.update({ f"{prefix}/{parse_path(path)}": rms_norm(arr) })
        jax.experimental.io_callback(wandb_logger, None, logs, commit=False)

    def init_fn(params=None):
        del params
        return VisualizeRmsState()
    
    def update_fn(updates, state=None, params=None):
        if wandb_logger is not None:
            log_norm(updates, prefix="update_RMS")
        return updates, VisualizeRmsState()
    
    return optax.GradientTransformation(init_fn, update_fn)


def adamw_visualize(
        learning_rate: optax.ScalarOrSchedule,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        nesterov: bool = False,
        weight_decay: float = 0.0,
        wandb_logger: Any | None = None,
) -> optax.GradientTransformation:
    return optax.chain(
        optax.scale_by_adam(
            b1=beta1, b2=beta2, eps=eps, nesterov=nesterov),
        visualize_rms(wandb_logger),
        optax.add_decayed_weights(weight_decay),
        optax.scale_by_learning_rate(learning_rate),
    )