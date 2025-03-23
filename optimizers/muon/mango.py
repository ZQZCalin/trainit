"""Mango optimizer."""

import jax
import jax.experimental
import jax.numpy as jnp
import jax.tree_util as jtu
import optax

from typing import NamedTuple, Optional, Callable, Literal, Dict, Tuple
from jaxtyping import Array, PyTree
import logging

from utils import tree_utils
from optimizers.combine import multi_transform
from optimizers.muon.base import (
    newton_schulz,
    scale_by_grad_squared,
    scale_by_function,
    scale_by_param_function,
    scale_by_offset,
    implicit_gradient_transport,
)
from optimizers.muon.normalization import get_normalize_fn
from optimizers.muon.tensor_norm import get_tensor_norm_fn


ArrayFn = Callable[[Array], Array]
NormalizeFn = ArrayFn


def scale_by_normalization(normalize, **kwargs):
    if normalize is None:
        return optax.identity()
    normalize_fn = get_normalize_fn(normalize, **kwargs)
    return scale_by_function(normalize_fn)


class NormalizeWithGradSquaredState(NamedTuple):
    """normalize_with_grad_squared wrapper state."""
    count: Array
    grad_squared: optax.Updates
    inner_state: optax.OptState


def normalize_with_grad_squared(
        inner: optax.GradientTransformation,
        normalize_fn: NormalizeFn,
        beta: float = 0.0,
        eps: float = 1e-8,
        power_pre: float = 0.5,
        power_post: float = 0.0,
        correct_bias: bool = True,
        stabilize_postcond: str = "",
) -> optax.GradientTransformation:
    """Normalize with grad_squared wrapper.

    The wrapper works like the following.  Given some norm |X|, we define 
    |X|_V = |sqrt(V)*X|, V = \sum g**2.

    This wrapper then normalizes update U_t w.r.t. |U_t|_{V_t}, i.e., 
    U \mapsto \argmax_{|X|_V=1} <X, U>
        = ( \argmax_{|Y|=1} <Y, U/sqrt(V)> ) / sqrt(V),

    where Y can be found by `normalize_fn`.
    
    This corresponds to the following work flow:
        1. Update preoonditioner V = V*beta + (1-beta)*g**2,
        2. Call inner optimizer and get update U,
        3. Precondition U -> U / sqrt(V),
        3. Normalize U,
        4. Precondition U again U -> U / sqrt(V).
    
    In general, we can replace sqrt(V) with arbitrary power, V**p.
    Moreover, we implement unbalanced power for pre- and post-conditioning.
    The final update looks like the following:
        1. Update V = V*beta + (1-beta)*g**2,
        2. U = inner.update(),
        3. U = U / V**p_{pre},
        4. U = normalize(U),
        5. U = U / V**p_{post}.
    """
    if not beta:
        return inner

    def init_fn(params):
        return NormalizeWithGradSquaredState(
            count=jnp.zeros([], dtype=jnp.int32),
            grad_squared=tree_utils.zeros_like(params),
            inner_state=inner.init(params),
        )
    
    def update_fn(updates, state, params):
        count = state.count
        grad_squared = state.grad_squared
        inner_state = state.inner_state

        count_inc = optax.safe_int32_increment(count)

        # NOTE: initially we implemented the series notion v*beta + g**2.
        # we now changed to EMA notion v*beta + (1-beta)*g**2. 
        update_grad_squared = lambda v, g: v*beta + (1-beta)*g**2
        # NOTE: numerical stability could be an issue if we use
        # arbitrary power and jnp.power(). 
        # For now, we probably could interest in sqrt(V) or V**0.25.
        bias_correction = lambda v: v / (1 - beta**count_inc)
        stabilize_rms = lambda v: v / (jnp.linalg.norm(v, ord="fro") / (v.shape[0]*v.shape[1])**0.5 + 1e-8)
        stabilize_mean = lambda v: jnp.mean(jnp.abs(v))
        def get_condition_fn(power, correct_bias, stabilize=""):
            def preprocess(v):
                if stabilize == "rms":
                    v = stabilize_rms(v)
                    # logging.info("Condition stabilized by RMS norm.")
                elif stabilize == "mean":
                    v =stabilize_mean(v)
                    # logging.info("Condition stabilized by mean norm.")
                else:
                    # logging.info("Condition not stabilized.")
                    pass
                if correct_bias:
                    v = bias_correction(v)
                    # logging.info("Condition bias corrected.")
                else:
                    # logging.info("Condition not bias corrected.")
                    pass
                return v
            if power == 0.0:
                condition_fn = lambda u, v: u
            elif power == 0.5:
                condition_fn = lambda u, v: u / (jnp.sqrt(preprocess(v)) + eps)
            elif power == 0.25:
                condition_fn = lambda u, v: u / (jnp.sqrt(jnp.sqrt(preprocess(v))) + eps)
            else:
                condition_fn = lambda u, v: u / (jnp.power(preprocess(v), power) + eps)
            return condition_fn
        precondition = get_condition_fn(power_pre, correct_bias)
        postcondition = get_condition_fn(power_post, correct_bias, stabilize_postcond)

        grad_squared = jtu.tree_map(update_grad_squared, grad_squared, updates)
        updates, inner_state = inner.update(updates, inner_state, params)
        updates = jtu.tree_map(precondition, updates, grad_squared)
        updates = jtu.tree_map(normalize_fn, updates)
        updates = jtu.tree_map(postcondition, updates, grad_squared)

        return updates, NormalizeWithGradSquaredState(
            count=count_inc, grad_squared=grad_squared, inner_state=inner_state
        )

    return optax.GradientTransformation(init_fn, update_fn)


list_of_induced_norms = [
    "Spectral", "ColNorm", "RowNorm", "Sign", "Euclidean"
]


# deprecated
def norm_pairing(
        induced_norm: str,
        dimension_correction: bool = True,
        *,
        eps: float = 1e-8,
        ns_steps: int = 6,
        transpose: bool = False,
        clip_ns: bool = True
) -> Tuple[ArrayFn, ArrayFn]:
    """Returns a pair of functions (norm_fn, normalize_fn).

    Implements dimension correction based on this paper (Table 2).

    https://arxiv.org/pdf/2502.07529
    
    norm_fn: G -> induced operator norm of G
    normalize_fn: G -> argmax_{|G'|_op} <G',G>
    """
    if induced_norm == "Spectral":
        """Standard spectral norm induced by 2 -> 2.
        norm: |G|_{Spectral} = sqrt(d_in/d_out) * |A|_{S_inf},
        normalize: G -> sqrt(d_out/d_in) * Newton-Schulz(G)
        """
        def norm_fn(G):
            assert G.ndim == 2, f"induced norm {induced_norm} requires 2d array."
            norm = jnp.linalg.norm(G, ord=jnp.inf)
            if dimension_correction:
                norm *= (G.shape[1]/G.shape[0])**0.5
            return norm
        def normalize_fn(G):
            assert G.ndim == 2, f"induced norm {induced_norm} requires 2d array."
            G = newton_schulz(G, steps=ns_steps)
            # Force to use the same scale as muon if clip_ns = True
            if clip_ns:
                G *= max(1, G.shape[0]/G.shape[1])**0.5
            elif dimension_correction:
                G *= (G.shape[0]/G.shape[1])**0.5
            return G
        
    if induced_norm == "ColNorm":
        """Row-max l2 norm induced by 2 -> inf.
        norm: |G|_{ColNorm} = max( sqrt(1/d_out) * |col_i(A)|_2 )
        normalize: col(G) -> sqrt(d_out) * col(G)/|col(G)|_2
        """
        def norm_fn(G):
            assert G.ndim == 2, f"induced norm {induced_norm} requires 2d array."
            if transpose:
                G = jnp.transpose(G)
            norm = jnp.max(jnp.linalg.norm(G, axis=0))  # axis=0 broadcasts along column
            if dimension_correction:
                norm *= 1 / (G.shape[0])**0.5
            return norm
        def normalize_fn(G):
            assert G.ndim == 2, f"induced norm {induced_norm} requires 2d array."
            if transpose:
                G = jnp.transpose(G)
            G = G / (jnp.linalg.norm(G, axis=0, keepdims=True) + eps)
            if dimension_correction:
                G *= (G.shape[0])**0.5
            if transpose:
                G = jnp.transpose(G)
            return G
        
    if induced_norm == "RowNorm":
        """Row-max """
        def norm_fn(G):
            assert G.ndim == 2, f"induced norm {induced_norm} requires 2d array."
            if transpose:
                G = jnp.transpose(G)
            norm = jnp.max(jnp.linalg.norm(G, axis=1))  # axis=1 broadcasts along row
            if dimension_correction:
                norm *= (G.shape[1])**0.5
            return norm
        def normalize_fn(G):
            assert G.ndim == 2, f"induced norm {induced_norm} requires 2d array."
            if transpose:
                G = jnp.transpose(G)
            G = G / (jnp.linalg.norm(G, axis=1, keepdims=True) + eps)
            if dimension_correction:
                G *= (1/G.shape[1])**0.5
            if transpose:
                G = jnp.transpose(G)
            return G
        
    if induced_norm == "Sign":
        """Sign induced by 1 -> inf.
        norm: |G| = max(G_{ij})
        normalize: G -> sign(G)
        """
        def norm_fn(G):
            if G.ndim == 1:
                return jnp.linalg.norm(G, ord=jnp.inf)
            if G.ndim == 2:
                return jnp.max(jnp.linalg.norm(G, ord=jnp.inf, axis=0))
        normalize_fn = jnp.sign
    
    if induced_norm == "Euclidean":
        """Euclidean induced by 2 -> scalar absolute value
        norm: |G| = sqrt(1/d) * |G|_2 (vector 2 norm)
        normalize: G -> sqrt(d) * G / |G|_2
        """
        def norm_fn(G):
            assert G.ndim == 1, f"induced norm {induced_norm} requires 1d array."
            norm = jnp.linalg.norm(G)
            if dimension_correction:
                norm *= (1/len(G))**0.5
            return norm
        def normalize_fn(G):
            assert G.ndim == 1, f"induced norm {induced_norm} requires 1d array."
            G = G / (norm_fn(G) + eps)
            return G
        
    return norm_fn, normalize_fn


def scale_by_absolute_tensor_norm(
        norm: str | None = None,
        power: float = 1,
        clip_min: float | None = 1.0,
        clip_max: float | None = None,
) -> optax.GradientTransformation:
    """Scale the update by tensor norm of parameter.
    
    Works as
    updates -> updates * clip(norm(params)**p, min, max).
    """
    if norm is None:
        return optax.identity()
    if not clip_min:
        clip_min = 0.0
    if not clip_max:
        clip_max = jnp.inf
    clip_fn = lambda x: jnp.clip(x, min=clip_min, max=clip_max)
    tensor_norm_fn = get_tensor_norm_fn(norm)
    scale_fn = lambda u, p: u * clip_fn(tensor_norm_fn(p)**power)
    return scale_by_param_function(scale_fn)


class ScaleByRelativeTensorNormState(NamedTuple):
    """scale_by_relative_tensor_norm state."""
    initial_clipped_norms: PyTree


def scale_by_relative_tensor_norm(
        norm: str | None = None,
        power: float = 1,
        clip_min: float | None = 1e-4,
        clip_max: float | None = None,
) -> optax.GradientTransformation:
    """Scale the update by ratio of tensor norm.
    
    Works as
    updates -> updates * ( norm(p_t) / clip(norm(p_0)) )**p
    """
    if norm is None:
        return optax.identity()
    
    tensor_norm_fn = get_tensor_norm_fn(norm)
    
    def init_fn(params):
        initial_clipped_norms = jtu.tree_map(
            lambda p: jnp.clip(tensor_norm_fn(p), min=clip_min, max=clip_max),
            params
        )
        return ScaleByRelativeTensorNormState(
            initial_clipped_norms=initial_clipped_norms
        )
    
    def update_fn(updates, state, params):
        current_norms = jtu.tree_map(tensor_norm_fn, params)
        updates = jtu.tree_map(
            lambda u, nt, n0: u * (nt / n0)**power,
            updates, current_norms, state.initial_clipped_norms
        )
        return updates, state
    
    return optax.GradientTransformation(init_fn, update_fn)


def scale_by_tensor_norm(
        norm: str | None = None,
        power: float = 1,
        clip_min: float | None = 1e-4,
        clip_max: float | None = None,
        scale_by_ratio: bool = False,
) -> optax.GradientTransformation:
    """Wraps both absolute and relative scale_by_tensor_norm methods."""
    if scale_by_ratio:
        return scale_by_relative_tensor_norm(norm, power, clip_min, clip_max)
    else:
        return scale_by_absolute_tensor_norm(norm, power, clip_min, clip_max)


# ========================================================================
# Default parameter partitioning.
# ========================================================================

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


# ========================================================================
# OG Mango.
# ========================================================================

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


# ========================================================================
# Visualization GradientTransformation.
# ========================================================================

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


# ========================================================================
# Mango v2.
# Enabling more flexible algorithm construction.
# ========================================================================

def mango_v2(
        lr: float | Dict[str, float] = 0.05,
        beta1: float | Dict[str, float] = 0.95,
        beta2: float | Dict[str, float | None] | None = None,
        nesterov: bool | Dict[str, bool] = True,
        use_adamw: bool | Dict[str, bool] = False,
        normalize: str | Dict[str, str | None] | None = default_mango_normalizations,
        scale_weight: str | Dict[str, str | None] | None = None,
        scale_power: float | Dict[str, float] = 1,
        scale_dim: bool | Dict[str, bool] = False,
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
        clip_ns: bool = True,
        transpose_embedding: bool = True,
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
            normalize: str | None = None,
            scale_dim: bool = False,
            scale_dim_clip_min: float | None = None,
            scale_dim_clip_max: float | None = None,
            scale_weight: str | None = None,
            scale_power: float = 1,
            scale_clip_low: float = 1.0,
            scale_clip_high: float = jnp.inf,
            clip_ns: bool = True,
            use_adamw: bool = False,
            offset_beta: float = 0.0,
            igt_scale: float = 0.0,
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
        # Normalization and weight scaling
        if normalize in list_of_induced_norms:
            # print(f"{name}/scale_dim={scale_dim}")
            # print(f"{name}/scale_weight={scale_weight}")
            transpose = (name == "embedding") and transpose_embedding
            norm_fn, normalize_fn = norm_pairing(
                induced_norm=normalize,
                dimension_correction=scale_dim,
                eps=eps,
                ns_steps=ns_steps,
                transpose=transpose,  # I think the correct thing is to transpose embedding matrix
                clip_ns=clip_ns,
            )
            normalize_and_scale = optax.chain(
                scale_by_function(normalize_fn),
                scale_by_param_function(
                    f=lambda u, w: u * jnp.clip(norm_fn(w)**scale_power, min=scale_clip_low, max=scale_clip_high)
                ) if scale_weight else optax.identity(),
            )
        else:
            normalize_and_scale = optax.chain(
                scale_by_normalization(
                    normalize, 
                    eps=eps, 
                    ns_steps=ns_steps, 
                    num_heads=num_heads,
                    scale_dim=scale_dim,
                    clip_min=scale_dim_clip_min,
                    clip_max=scale_dim_clip_max,
                ),
                scale_by_tensor_norm(scale_weight, scale_power, clip_min=scale_clip_low, clip_max=scale_clip_high),
            )
        # Learning rate
        learning_rate = lr if schedule is None else lambda t: lr * schedule(t)
        if schedule_wrapper:
            learning_rate = schedule_wrapper(learning_rate, name)
        optimizer = optax.chain(
            optimizer,
            normalize_and_scale,
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
            scale_dim=scale_dim,
            offset_beta=offset_beta,
            igt_scale=igt_scale,
            scale_clip_low=scale_clip_low,
            scale_clip_high=scale_clip_high,
            clip_ns=clip_ns
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
                scale_dim=parse_args(scale_dim, param),
                offset_beta=offset_beta,
                igt_scale=igt_scale,
                scale_clip_low=scale_clip_low,
                scale_clip_high=scale_clip_high,
                clip_ns=clip_ns
            ) for param in param_keys
        }
        optimizer = multi_transform(transforms, param_labels)

    return optimizer


# ========================================================================
# Unfortunately, mango_v2 becomes too clunky to use / modify.
# Below implements a cleaner version of mango_v3.
# ========================================================================

def mango_core(
        learning_rate: optax.ScalarOrSchedule,
        beta1: float = 0.95,
        beta2: float = 0.95,
        nesterov: bool = True,
        eps: float = 1e-8,
        # normalization
        normalize: str | None = None,
        scale_dim: bool = False,
        scale_dim_transpose: bool = False,
        scale_dim_clip_min: float | None = None,
        scale_dim_clip_max: float | None = None,
        ns_steps: int = 6,
        num_heads: int = 12,
        # norm scaling
        scale_norm: str | None = None,
        scale_norm_ratio: bool = False,
        scale_norm_power: float = 1.0,
        scale_norm_clip_min: float | None = 1.0,
        scale_norm_clip_max: float | None = None,
        # other tweaks
        use_adamw: bool = False,
        offset_beta: float = 0.0,
        igt_scale: float = 0.0,
        coupled_normalize: bool = False,
        coupled_normalize_power_pre: float = 0.5,
        coupled_normalize_power_post: float = 0.5,
        coupled_normalize_correct_bias: bool = True,
):
    # Base optimizer: AdamW or LaProp
    # TODO: include non-diagonal preconditioning like SOAP
    def init_optim_base():
        # NOTE: Optax.trace uses the conventional mu = mu * beta + g
        # instead of the average formula, i.e., mu = mu * beta + (1-beta) * g.
        optim_momentum = optax.trace(decay=beta1, nesterov=nesterov)
        # NOTE: Optax implements Nadam based on 
        # Dozat 2016 https://openreview.net/pdf?id=OM0jvwB8jIp57ZJjtNEZ
        # with a caveat that nu_hat is not multiplied by beta2.
        # See further notes in optax implementation.
        optim_adamw = optax.scale_by_adam(b1=beta1, b2=beta2, eps=eps, nesterov=nesterov)
        optim_laprop = optax.chain(
            scale_by_grad_squared(beta=beta2, eps=eps),
            optim_momentum,
        )
        if coupled_normalize:
            optim_base = optim_momentum
        else:
            if use_adamw:
                optim_base = optim_adamw
            else:
                optim_base = optim_laprop
        return optim_base
    
    def init_optim_normalized(inner):
        normalize_fn = get_normalize_fn(
            normalize, 
            eps=eps, 
            ns_steps=ns_steps, 
            num_heads=num_heads,
            scale_dim=scale_dim,
            transpose=scale_dim_transpose,
            clip_min=scale_dim_clip_min,
            clip_max=scale_dim_clip_max,
        )
        if coupled_normalize:
            optim_normalized = normalize_with_grad_squared(
                inner=inner, 
                normalize_fn=normalize_fn, 
                beta=beta2,
                eps=eps, 
                power_pre=coupled_normalize_power_pre,
                power_post=coupled_normalize_power_post,
                correct_bias=coupled_normalize_correct_bias,
            )
        else:
            optim_normalized = optax.chain(
                inner, scale_by_function(normalize_fn)
            )
        return optim_normalized
    
    def init_optim_scale_norm():
        optim_scale_norm = scale_by_tensor_norm(
            norm=scale_norm,
            power=scale_norm_power,
            clip_min=scale_norm_clip_min,
            clip_max=scale_norm_clip_max,
            scale_by_ratio=scale_norm_ratio,
        )
        return optim_scale_norm
    
    optim_base = init_optim_base()
    optim_normalized = init_optim_normalized(optim_base)
    optim_scale_norm = init_optim_scale_norm()
    return optax.chain(
        optim_normalized,
        optim_scale_norm,
        optax.scale_by_learning_rate(learning_rate),
        scale_by_offset(beta=offset_beta) if offset_beta else optax.identity(),
        implicit_gradient_transport(beta=beta1, scale=igt_scale) if igt_scale else optax.identity(),
    )


def mango_v3(
        config_dict: Dict[str, dict],
        schedule: optax.Schedule,
        param_labels: Callable[[PyTree], PyTree] | None = mango_label_gpt,
) -> optax.GradientTransformation:
    """Mango v3.
    
    Cleaner code, easier construction.
    """

    transforms = {}
    for param, config in config_dict.items():
        lr = config.pop("lr")
        transforms[param] = mango_core(
            learning_rate=lambda t: lr * schedule(t),
            **config
        )
    optimizer = multi_transform(transforms, param_labels)

    return optimizer


class ScaleByMangoState(NamedTuple):
    count: Array
    momentum: optax.Updates
    grad_squared: optax.Updates


def scale_by_mango(
        beta1: float = 0.95,
        beta2: float = 0.95,
        nesterov: bool = True,
        normalize_fn: callable | None = None,
        scale_rms: bool = True,
        eps: float = 1e-8,
        laprop: bool = False,
        precond_power: float = 0.5,
        postcond_power: float = 0.0,
) -> optax.GradientTransformation:
    def init_fn(params):
        return ScaleByMangoState(
            count=jnp.zeros([], dtype=jnp.int32),
            momentum=tree_utils.zeros_like(params),
            grad_squared=tree_utils.zeros_like(params) if beta2 else None,
        )
    
    def jnp_pow(G, power):
        if power == 0:
            return G
        if power == 0.5:
            return jnp.sqrt(G)
        if power == 0.25:
            return jnp.sqrt(jnp.sqrt(G))
        else:
            return jnp.pow(G, power)
        
    def rms(G):
        if G.ndim <= 1:
            return jnp.linalg.norm(G, ord=2) / len(G)**0.5
        if G.ndim == 2:
            return jnp.linalg.norm(G, ord="fro") / (G.shape[0]*G.shape[1])**0.5
    
    def update_fn(updates, state, params=None):
        del params
        count = state.count
        momentum = state.momentum
        grad_squared = state.momentum

        # 1. Update preconditioner.
        if beta2:
            grad_squared = jtu.tree_map(
                lambda v, g: v*beta2 + (1-beta2)*g**2, 
                grad_squared, updates
            )

        # [Optional] LaProp style pre-conditioning.
        if beta2 and precond_power and laprop:
            updates = jtu.tree_map(
                lambda g, v: g / (jnp_pow(v, precond_power) + eps)
            )
        
        # 2. Update momentum.
        momentum = jtu.tree_map(
            lambda m, g: m*beta1 + g,
        )
        if nesterov:
            updates = jtu.tree_map(
                lambda m, g: m*beta1 + g, momentum, updates)
        else:
            updates = momentum

        # 3. Adam-style pre-conditioning.
        if beta2 and precond_power and not laprop:
            updates = jtu.tree_map(
                lambda u, v: u / (jnp_pow(v, precond_power) + eps)
            )

        # 4. Apply normalization.
        if normalize_fn:
            updates = jtu.tree_map(normalize_fn, updates)
        
        # 5. Apply post-conditioning.
        if beta2 and postcond_power:
            updates = jtu.tree_map(
                lambda u, v: u / (jnp_pow(v, postcond_power) + eps)
            )
        
        # 6. Apply rms normalizaiton.
        if scale_rms:
            updates = jtu.tree_map(
                lambda u: u / rms(u), updates
            )

        return updates, ScaleByMangoState(
            count=optax.safe_int32_increment(count),
            momentum=momentum,
            grad_squared=grad_squared,
        )
    return optax.GradientTransformation(init_fn, update_fn)