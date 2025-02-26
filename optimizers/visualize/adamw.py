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


class VisualizeRmsState(NamedTuple):
    """An empty node for visualize_rms state."""


def visualize_rms(
        wandb_logger: Any | None = None,
) -> optax.GradientTransformation:
    """Visualize RMS norm of updates.
    
    Does not affect updates.
    """
    if not wandb_logger:
        return optax.identity()
    
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
        log_norm(updates, prefix="update_RMS")
        return updates, VisualizeRmsState()
    
    return optax.GradientTransformation(init_fn, update_fn)


def adamw_gpt(
        learning_rate: optax.ScalarOrSchedule,
        adam_lr: optax.ScalarOrSchedule | None = None,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        nesterov: bool = False,
        weight_decay: float = 0.0,
        *,
        wandb_logger: Any | None = None,
) -> optax.GradientTransformation:
    """Adam, but with extra layer of visualize for update RMS norm.
    Also, partitions muon part and adam part separately with different lr.
    """
    if adam_lr is None:
        adam_lr = learning_rate

    base_label = "muon"
    adam_label = "adam"

    def params_label(params):
        def parse_path(path, p):
            parts = [part.name for part in path if isinstance(part, jtu.GetAttrKey)]
            # Detect embedding layers and head layers
            if "token_embedding" in parts or "position_embedding" in parts:
                return adam_label
            if "head" in parts:
                return adam_label
            if p.ndim == 1:
                return adam_label
            if p.ndim == 2:
                return base_label
            raise ValueError(f"cannot categorize parameter: {p}")
        return jtu.tree_map_with_path(parse_path, params) 
    
    return optax.chain(
        multi_transform({
            base_label: optax.scale_by_adam(
                b1=beta1, b2=beta2, eps=eps, nesterov=nesterov),
            adam_label: optax.scale_by_adam(
                b1=beta1, b2=beta2, eps=eps, nesterov=False),
        }, params_label),
        visualize_rms(wandb_logger),
        optax.add_decayed_weights(weight_decay),
        multi_transform({
            base_label: optax.scale_by_learning_rate(learning_rate),
            adam_label: optax.scale_by_learning_rate(adam_lr),
        }, params_label),
    )