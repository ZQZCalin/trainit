"""Mango optimizer."""

import jax
import jax.experimental
import jax.numpy as jnp
import jax.tree_util as jtu
import optax

from typing import NamedTuple, Optional, Callable, Literal, Dict, Tuple
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
    implicit_gradient_transport,
)


ArrayFn = Callable[[Array], Array]

def split_vmap(f: ArrayFn, num_heads: int = 1) -> ArrayFn:
    """Broadcasts a function f: [d,:] -> [d,:] to a matrix/vector [3nd,:]
    in a way that first reshapes into [3,n,d,:], then applies f on [d,:],
    and finally reshape back into [3nd,:].
    """
    def split_fn(G):
        assert G.ndim == 1 or G.ndim == 2
        assert G.shape[0] % (3 * num_heads) == 0
        ndim = G.ndim
        shape = G.shape
        n = num_heads
        d = G.shape[0] // (3 * num_heads)
        # Reshape into [3,n,d,:].
        if ndim == 1:
            G = jnp.reshape(G, (3, n, d))
        else:
            G = jnp.reshape(G, (3, n, d, shape[1]))
        # Use nested vmap to broadcast mapping f to the last axes (d,:).
        G = jax.vmap(jax.vmap(f))(G)
        # Reshape back into [3nd,:].
        if ndim == 1:
            G = jnp.reshape(G, (3*n*d,))
        else:
            G = jnp.reshape(G, (3*n*d, shape[1]))
        return G
    return split_fn


def scale_by_normalization(
        normalize: str | None = None,
        eps: float = 1e-8,
        ns_steps: int = 6,
        num_heads: int = 12,
):
    if normalize is None:
        return optax.identity()
    # normalize = str(normalize)
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
            f=lambda G: G / (jnp.linalg.norm(G) + eps),
            num_heads=num_heads,
        ))
    # NOTE: Feb.11: the previous implementation is a literal normalization by inf-norm,
    # and is different from the "project" notion x <- argmax_{\|x\|=1} <x, update>.
    # The correct implementation would be
    # - x <- sign(x) for vectors if \|\| is inf-norm
    # - x <- sign(x) for matrices if \|\| is the max of column-wise inf-norm 
    #   (which is the same implementation as the vector case!).
    # - also, head-wise inf-normalization makes no sense now. 
    if normalize == "inf_":
        # return scale_by_function(
        #     f=lambda G: G / (jnp.linalg.norm(G, ord=jnp.inf) + eps)
        # )
        return scale_by_function(jnp.sign)
    if normalize == "inf_col":
        raise ValueError("Please use 'inf_' for all normalization.")
        return scale_by_function(
            f=lambda G: G / (jnp.linalg.norm(G, ord=jnp.inf, axis=1, keepdims=True) + eps)
        )
    if normalize == "inf_split":
        raise ValueError("Please use 'inf_' for all normalization.")
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
            return G
        return scale_by_function(split_vmap(f, num_heads=num_heads))
    raise ValueError(f"invalid normalization type = '{normalize}'.")




class ScaleByWeightNormState(NamedTuple):
    """An empty node for scale_by_weight_norm state."""


def scale_by_weight_norm(
        scale_weight: str | None = None,
        scale_power: float = 1,
        clip_low: float = 1.0,
        clip_high: float = jnp.inf,
) -> optax.GradientTransformation:
    if scale_weight is None:
        return optax.identity()
    clip = lambda x: jnp.clip(x, min=clip_low, max=clip_high)
    name, p = scale_weight, scale_power
    if name == "l2":
        scale_fn = lambda u, w: u * clip(jnp.linalg.norm(w)**p)
    if name == "l2_col":
        scale_fn = lambda u, w: u * clip(jnp.linalg.norm(w, axis=1, keepdims=True)**p)
    if name == "inf_":
        scale_fn = lambda u, w: u * clip(jnp.linalg.norm(w, ord=jnp.inf)**p)
    if name == "inf_col":
        scale_fn = lambda u, w: u * clip(jnp.linalg.norm(w, ord=jnp.inf, axis=1, keepdims=True)**p)
    if name == "op":
        scale_fn = lambda u, w: u * clip(jnp.linalg.norm(w, ord=2)**p)

    def init_fn(params=None):
        del params
        return ScaleByWeightNormState()
    
    def update_fn(updates, state, params):
        updates = jtu.tree_map(scale_fn, updates, params)
        return updates, ScaleByWeightNormState()

    return optax.GradientTransformation(init_fn, update_fn)


mango_gpt_keys = ["mat", "embedding", "head", "attn_w", "attn_b", "vec_w", "vec_b"]


normalization_list = [
    "l2",               # l2 norm for vectors or frobenius norm for matrices
    "l2_col",           # column-wise l2 norm for matrices
    "l2_split",         # head-wise l2 (frobenius) norm, for attention weights / bias
    "inf_",             # inf norm for vectors or spectral norm for matrices
    "inf_col",          # column-wise inf norm for matrices
    "inf_split",        # head-wise inf (spectral) norm, for attention weights / bias
    "ns",               # newton-schulz for matrices
    "ns_split",         # head-wise newton-schulz for matrices, particularly attention weights
]


default_mango_normalizations = {
    "mat": "ns",
    "embedding": "l2_col",
    "head": "ns",
    "attn_w": "ns_split",
    "attn_b": "l2_split",
    "vec_w": "inf_",
    "vec_b": "l2",
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
            return "mat"
        if p.ndim == 1 and "weight" in parts:
            return "vec_w"
        if p.ndim == 1 and "bias" in parts:
            return "vec_b"
        raise ValueError(f"cannot categorize parameter: {p}")
    return jtu.tree_map_with_path(fn, params) 


def mango(
        lrs: float | Dict[str, float] = 0.05,
        schedule: optax.Schedule | None = None,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 6,
        eps: float = 1e-8,
        beta2: float | None = None,
        offset_beta: float | None = None,
        normalizations: Dict[str, str | None] | None = default_mango_normalizations,
        schedule_wrapper: Callable[[optax.ScalarOrSchedule], optax.ScalarOrSchedule] | None = None,
) -> optax.GradientTransformation:
    """Mango (Momentum with Advanced Normalization, Gradient-preconditing and Offset update).
    
    Args:
        lrs: float if global lr, dict for parameter-specific lrs
        schedule: optax.Schedule function. 
            Note: schedule should be an unwrapped function. you can provide additional schedule_wrapper,
            which wraps the schedule for 2d matrices by default.
        normalizations: dict for normalization types.
    Other args should be self-explanatory.
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
    if normalizations is None:
        optim_normalization = optax.identity()
    else:
        transforms = { k: scale_by_normalization(normalizations[k], eps=eps, ns_steps=ns_steps, num_heads=num_heads) for k in mango_gpt_keys }
        optim_normalization = multi_transform(transforms, mango_label_gpt)

    # Advanced learning rate schedules based on parameters.
    if isinstance(lrs, float):
        learning_rate = lrs if schedule is None else lambda t: lrs * schedule(t)
        if schedule_wrapper is not None:
            learning_rate = schedule_wrapper(learning_rate)
        optim_schedule = optax.scale_by_learning_rate(learning_rate)
    else:
        if schedule is None:
            learning_rates = { k: lrs[k] for k in mango_gpt_keys }
        else:
            learning_rates = { k: lambda t: lrs[k] * schedule(t) for k in mango_gpt_keys }
        if schedule_wrapper is not None:
            learning_rates["mat"] = schedule_wrapper(learning_rates["mat"])
        lr_transforms = { k: optax.scale_by_learning_rate(v) for k,v in learning_rates.items() }
        optim_schedule = multi_transform(lr_transforms, mango_label_gpt)

    return optax.chain(
        optim_grad_precond,
        optim_momentum,
        optim_normalization,
        optim_schedule,
        optim_offset,
    )


class VisualizeNormState(NamedTuple):
    """An empty node for visualize_norm state."""


def visualize_norm(
        wandb_logger: None = None,
) -> optax.GradientTransformation:
    """Visualize norms of gpt2 weights and updates.
    
    Does not affect updates.
    """
    def parse_path(path, *args):
        # Parse path list into a single string.
        parts = []
        for part in path:
            if isinstance(part, jtu.GetAttrKey):
                parts.append(part.name)
            elif isinstance(part, jtu.SequenceKey):
                parts[-1] += f"[{part.idx}]"
        return ".".join(parts)
    
    def compute_norms(path: str, arr: Array):
        norms = {}
        if arr.ndim == 1:
            norms.update({
                "l2": jnp.linalg.norm(arr),
                "inf": jnp.linalg.norm(arr, ord=jnp.inf),
                "-inf": jnp.linalg.norm(arr, ord=-jnp.inf),
            })
        if arr.ndim == 2:
            norms.update({
                "op": jnp.linalg.norm(arr, ord=2),
                "-op": jnp.linalg.norm(arr, ord=-2),
                "fro": jnp.linalg.norm(arr),
            })
        if "embedding" in path or "head" in path:
            d = {
                "l2_row": jnp.linalg.norm(arr, axis=0),     # [768,]
                "l2_col": jnp.linalg.norm(arr, axis=1),     # [50258,]
                "inf_row": jnp.linalg.norm(arr, ord=jnp.inf, axis=0),
                "inf_col": jnp.linalg.norm(arr, ord=jnp.inf, axis=1),
                "range_row": jnp.linalg.norm(arr, ord=jnp.inf, axis=0) - jnp.linalg.norm(arr, ord=-jnp.inf, axis=0),
                "range_col": jnp.linalg.norm(arr, ord=jnp.inf, axis=1) - jnp.linalg.norm(arr, ord=-jnp.inf, axis=1),
            }
            for k, v in d.items():
                norms.update({
                    f"{k}_min": jnp.min(v),
                    f"{k}_max": jnp.max(v),
                    f"{k}_mean": jnp.mean(v),
                    f"{k}_std": jnp.std(v),
                    f"{k}_len": len(v),
                })
        return norms
    
    def log_norm(tree: optax.Updates, prefix: str):
        logs = {}
        for path, arr in jtu.tree_leaves_with_path(tree):
            path = parse_path(path)
            norms = compute_norms(path, arr)
            for k, v in norms.items():
                logs.update({ f"{prefix}/{path}/{k}": v })
        jax.experimental.io_callback(wandb_logger, None, logs, commit=False)

    def init_fn(params=None):
        del params
        return VisualizeNormState()
    
    def update_fn(updates, state, params):
        if wandb_logger:
            log_norm(params, prefix="params_norm")
            log_norm(updates, prefix="updates_norm")
        return updates, VisualizeNormState()
    
    return optax.GradientTransformation(init_fn, update_fn)


def mango_v2(
        lr: float | Dict[str, float] = 0.05,
        beta1: float | Dict[str, float] = 0.95,
        beta2: float | Dict[str, float | None] | None = None,
        nesterov: bool | Dict[str, bool] = True,
        use_adamw: bool | Dict[str, bool] = False,
        normalize: str | Dict[str, str | None] | None = default_mango_normalizations,
        scale_weight: str | Dict[str, str | None] | None = None,
        scale_power: float | Dict[str, float] = 1,
        eps: float = 1e-8,
        ns_steps: int = 6,
        num_heads: int = 12,
        offset_beta: float | None = None,
        schedule: optax.Schedule | None = None,
        schedule_wrapper: Callable[[optax.ScalarOrSchedule, str], optax.ScalarOrSchedule] | None = None,
        param_labels: Callable[[PyTree], PyTree] | None = mango_label_gpt,
        igt_scale: float = 0.0,
        scale_clip_low: float = 1.0,
        scale_clip_high: float | None = None,
) -> optax.GradientTransformation:
    """Mango v2. 

    Extend from base mango optimizer by 
    - adding LAMB-style weight-norm scaling;
    - enabling switching between LaProp and Adamw;


    """

    if not scale_clip_high:
        scale_clip_high = jnp.inf

    # Check all dict arguments have the same keys.
    dict_args = [arg for arg in (lr, beta1, beta2, nesterov, use_adamw, normalize, scale_weight, scale_power) if isinstance(arg, dict)]
    if len(dict_args) == 0:
        param_keys = []
    else:
        param_keys = set(dict_args[0].keys())
    if not all(set(arg.keys()) == param_keys for arg in dict_args):
        raise ValueError("All dictionary arguments must have the same keys.")

    def mango_component(
            lr: optax.ScalarOrSchedule,
            name: str = "",
            beta1: float = 0.95,
            beta2: float = 0.95,
            nesterov: bool = True,
            use_adamw: bool = False,
            normalize: str | None = None,
            scale_weight: str | None = None,
            scale_power: float = 1,
            offset_beta: float = 0.0,
            igt_scale: float = 0.0,
            scale_clip_low: float = 1.0,
            scale_clip_high: float = jnp.inf,
    ):
        if use_adamw:
            # Optax implements Nadam based on 
            # Dozat 2016 https://openreview.net/pdf?id=OM0jvwB8jIp57ZJjtNEZ
            # with a caveat that nu_hat is not multiplied by beta2.
            # See further notes in optax implementation.
            if beta2 is not None:
                optimizer = optax.scale_by_adam(b1=beta1, b2=beta2, eps=eps, nesterov=nesterov)
            # If beta2 is None, always use optax.trace regardless of use_adamw
            else:
                optimizer = optax.trace(decay=beta1, nesterov=nesterov)
        else:
            # Optax.trace uses the conventional mu = mu * beta + g
            # instead of the average formula, i.e., mu = mu * beta + (1-beta) * g.
            optimizer = optax.chain(
                scale_by_grad_squared(beta=beta2) if beta2 else optax.identity(),
                optax.trace(decay=beta1, nesterov=nesterov)
            )
        learning_rate = lr if schedule is None else lambda t: lr * schedule(t)
        if schedule_wrapper:
            learning_rate = schedule_wrapper(learning_rate, name)
        optimizer = optax.chain(
            optimizer,
            scale_by_normalization(normalize, eps=eps, ns_steps=ns_steps, num_heads=num_heads),
            scale_by_weight_norm(scale_weight, scale_power, clip_low=scale_clip_low, clip_high=scale_clip_high),
            optax.scale_by_learning_rate(learning_rate),
            scale_by_offset(beta=offset_beta) if offset_beta else optax.identity(),
            implicit_gradient_transport(beta=beta1, scale=igt_scale) if igt_scale else optax.identity(),
        )
        return optimizer
    
    # No dictionary argument: global config for all subgroups.
    if len(param_keys) == 0:
        optimizer = mango_component(
            lr=lr, 
            name="mango", 
            beta1=beta1, 
            beta2=beta2, 
            nesterov=nesterov, 
            use_adamw=use_adamw,
            normalize=normalize, 
            scale_weight=scale_weight, 
            scale_power=scale_power,
            offset_beta=offset_beta,
            igt_scale=igt_scale,
        )
    else:
        parse_args = lambda arg, key: arg if not isinstance(arg, dict) else arg[key]
        transforms = {
            param: mango_component(
                lr=parse_args(lr, param),
                name=param,
                beta1=parse_args(beta1, param),
                beta2=parse_args(beta2, param),
                nesterov=parse_args(nesterov, param),
                use_adamw=parse_args(use_adamw, param),
                normalize=parse_args(normalize, param),
                scale_weight=parse_args(scale_weight, param),
                scale_power=parse_args(scale_power, param),
                offset_beta=offset_beta,
                igt_scale=igt_scale,
                scale_clip_low=scale_clip_low,
                scale_clip_high=scale_clip_high,
            ) for param in param_keys
        }
        optimizer = multi_transform(transforms, param_labels)

    return optimizer
