# An optax implementation of muon optimizer from:
# 
# https://github.com/KellerJordan/Muon
# 
# This implementation adapts the pytorch optimizer
# to an optax.GradientTransformation object.

"""Muon optimizer."""

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import optax

from typing import NamedTuple, List, Tuple
from jaxtyping import Array, PyTree
from functools import partial
import warnings

from utils import tree_utils
from optimizers.base import adamw
from optimizers.combine import multi_transform
from optimizers.schedule import get_current_lr
from optimizers.muon.base import newton_schulz, LabelParamsFn, scale_by_offset
from optimizers.muon.mango import normalize_with_grad_squared, scale_by_function


# =============================================================================
# Label functions to partition GPT model.
# =============================================================================

def muon_label_params_default(params):
    """Default way to partition layers.
    
    Following the default implementation from (Jordan, 2024)

    https://github.com/KellerJordan/Muon/blob/28c793b55ef1cf86e5d6091bfbdbe0029b11eabb/muon.py#L34

    Parses 1d arrays and embedding, head layers to adamw; and all other 2d arrays as muon.
    """
    def parse_path(path, p):
        parts = [part.name for part in path if isinstance(part, jtu.GetAttrKey)]
        muon_label = "muon"
        adamw_label = "adamw"
        # Detect embedding layers and head layers
        if "token_embedding" in parts or "position_embedding" in parts:
            return adamw_label
        if "head" in parts:
            return adamw_label
        if p.ndim == 1:
            return adamw_label
        if p.ndim == 2:
            return muon_label
        raise ValueError(f"cannot categorize parameter: {p}")
    return jtu.tree_map_with_path(parse_path, params) 


def muon_label_params_by_dimension(params):
    """A simplified label function that labels 2d arrays as muon and 1d arrays as adamw."""
    return jtu.tree_map(
        lambda p: "muon" if p.ndim == 2 else "adamw", params
    )


# =============================================================================
# Main scale_by_muon and muon function.
# =============================================================================

class ScaleByMuonState(NamedTuple):
    """scale_by_muon state."""
    count: Array
    muon_momentum: optax.Updates


def scale_by_muon(
        learning_rate: optax.ScalarOrSchedule = 0.05,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 6,
) -> optax.GradientTransformation:
    """Muon update on parameters with 2d arrays."""
    
    def init_fn(params):
        return ScaleByMuonState(
            count = jnp.zeros([], dtype=jnp.int32),
            muon_momentum = tree_utils.zeros_like(params),
        )
    
    def update_fn(updates, state, params=None):
        del params
        count = state.count
        muon_momentum = state.muon_momentum

        # Update momentum.
        muon_momentum = jtu.tree_map(
            lambda mu, g: momentum * mu + g, muon_momentum, updates)

        # Apply nesterov's momentum before applying normalization.
        if nesterov:
            updates = jtu.tree_map(
                lambda mu, g: momentum * mu + g, muon_momentum, updates)
        else:
            updates = muon_momentum

        # Orthogonalize momentum matrix.
        updates = jtu.tree_map(
            lambda G: newton_schulz(G, steps=ns_steps), updates)
        
        # Additional scaling based on shape (see line 135).
        updates = jtu.tree_map(
            lambda G: G * max(1, G.shape[0]/G.shape[1])**0.5, updates)

        # Wrap final update.
        lr = get_current_lr(learning_rate, count)
        updates = tree_utils.scalar_dot(updates, -lr)

        return updates, ScaleByMuonState(
            count = optax.safe_int32_increment(count),
            muon_momentum = muon_momentum
        )
    
    return optax.GradientTransformation(init_fn, update_fn)


def muon(
        learning_rate: optax.ScalarOrSchedule = 0.05,
        momentum: float = 0.95,
        nesterov: bool = True,
        beta2: float = 0.0,
        p_pre: float = 0.5,
        p_post: float = 0.0,
        stabilize: str = "",
        offset_beta: float = 0.0,
        ns_steps: int = 6,
        adam_lr: optax.ScalarOrSchedule = 3e-4,
        adam_beta1: float = 0.95,
        adam_beta2: float = 0.95,
        adam_eps: float = 1e-8,
        adam_wd: float = 0.0,
        label_params: LabelParamsFn | None = None
) -> optax.GradientTransformation:
    """The muon optimizer.
    
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
    if label_params is None:
        label_params = muon_label_params_default    # use default muon partition.
    # if not beta2:
    #     optim_muon = scale_by_muon(
    #         learning_rate, momentum, nesterov, ns_steps
    #     )
    # else:
    def normalize(G):
        G = newton_schulz(G, steps=ns_steps)
        G = G * max(1, G.shape[0]/G.shape[1])**0.5
        return G
    optim_muon = optax.trace(decay=momentum, nesterov=nesterov)
    optim_muon = normalize_with_grad_squared(
        inner=optim_muon,
        normalize_fn=normalize,
        beta=beta2,
        eps=1e-8, 
        power_pre=p_pre,
        power_post=p_post,
        correct_bias=True,
        stabilize_postcond=stabilize,
    )
    optim_muon = optax.chain(
        optim_muon,
        optax.scale_by_learning_rate(learning_rate)
    )
    optim_adam = adamw(
        learning_rate=adam_lr,
        beta1=adam_beta1,
        beta2=adam_beta2,
        eps=adam_eps,
        weight_decay=adam_wd,
        use_nesterov=False,
    )
    # NOTE: we could switch to optax implementation if needed.
    # optim_adam = optax.adamw(
    #     learning_rate=adam_lr,
    #     b1=adam_beta1,
    #     b2=adam_beta2,
    #     eps=adam_eps,
    #     weight_decay=adam_wd,
    #     nesterov=False,
    # )
    transforms = {
        "muon": optim_muon,
        "adamw": optim_adam,
    }
    optim = multi_transform(transforms, label_params)
    if offset_beta:
        optim = optax.chain(
            optim,
            scale_by_offset(offset_beta),
        )
    return optim


def muon_og(
        learning_rate: optax.ScalarOrSchedule = 0.05,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 6,
        ns_embedding: bool = False,
        ns_head: bool = False,
        adam_lr: optax.ScalarOrSchedule = 3e-4,
        adam_beta1: float = 0.95,
        adam_beta2: float = 0.95,
        adam_eps: float = 1e-8,
        adam_wd: float = 0.0,
) -> optax.GradientTransformation:
    """The OG muon optimizer that doesn't apply newton-schulz on embedding nor head layers."""
    warnings.warn("muon_og is deprecated. please use muon instead.",
                  "define new label functions if necessary.")
    
    optim_muon = scale_by_muon(
        learning_rate, momentum, nesterov, ns_steps
    )
    optim_momentum = optax.chain(
        optax.trace(decay=momentum, nesterov=nesterov),
        optax.scale_by_learning_rate(learning_rate)
    )
    optim_adamw = adamw(
        learning_rate=adam_lr,
        beta1=adam_beta1,
        beta2=adam_beta2,
        eps=adam_eps,
        weight_decay=adam_wd,
        use_nesterov=False,
    )
    transforms = {
        "muon": optim_muon,
        "momentum": optim_momentum,
        "adamw": optim_adamw,
    }
    def label_params(params):
        def get_layer(path, p):
            parts = [part.name for part in path if isinstance(part, jtu.GetAttrKey)]
            # Special ararys.
            if "token_embedding" in parts or "position_embedding" in parts:
                return "embedding"
            if "head" in parts:
                return "head"
            # General arrays.
            if p.ndim == 2:
                return "mat"
            if p.ndim == 1:
                return "vec"
            raise ValueError(f"cannot categorize parameter: {p}")
        parse_table = {
            # NOTE: we change to use adamw instead of SGDM for all layers which muon is not applied
            # "embedding": "muon" if ns_embedding else "momentum",
            # "head": "muon" if ns_head else "momentum",
            "embedding": "muon" if ns_embedding else "adamw",
            "head": "muon" if ns_head else "adamw",
            "mat": "muon",
            "vec": "adamw",
        }
        def fn(path, p):
            return parse_table[get_layer(path, p)]
        return jtu.tree_map_with_path(fn, params) 
    return multi_transform(transforms, label_params)


def general_newton_schulz(G, p):
    """Normalizes by 
    
    Using SVD now for test of idea."""
    U, S, Vh = jnp.linalg.svd(G)

    # Normalization w.r.t. ell-p vector norm.
    q = p / (p-1)
    ratio = 1 / (p-1)
    S = (S / jnp.linalg.norm(S, ord=q)) ** ratio
    # NOTE: fix to ord=q

    # SVD on G and tranlate to diagonal matrix.
    k = min(G.shape[0], G.shape[1])
    S = jnp.zeros_like(G).at[jnp.arange(k), jnp.arange(k)].set(S)

    return U@S@Vh


def muon_p(
        learning_rate: optax.ScalarOrSchedule = 0.05,
        momentum: float = 0.95,
        nesterov: bool = True,
        schatten_p: float = 2,
        ns_steps: int = 6,
        adam_lr: optax.ScalarOrSchedule = 3e-4,
        adam_beta1: float = 0.95,
        adam_beta2: float = 0.95,
        adam_eps: float = 1e-8,
        adam_wd: float = 0.0,
        label_params: LabelParamsFn | None = None
) -> optax.GradientTransformation:
    """Muon with newton-schulz replaced with normalization
    w.r.t. Schatten-p norm. p=inf recovers muon, and p=2 recovers normalized by frobenius norm.
    """
    if label_params is None:
        label_params = muon_label_params_default    # use default muon partition.
        
    def normalize(G):
        G = general_newton_schulz(G, schatten_p)
        G = G * max(1, G.shape[0]/G.shape[1])**0.5
        return G
    optim_muon = optax.chain(
        optax.trace(decay=momentum, nesterov=nesterov),
        scale_by_function(normalize),
        optax.scale_by_learning_rate(learning_rate),
    )
    optim_adam = adamw(
        learning_rate=adam_lr,
        beta1=adam_beta1,
        beta2=adam_beta2,
        eps=adam_eps,
        weight_decay=adam_wd,
        use_nesterov=False,
    )
    transforms = {
        "muon": optim_muon,
        "adamw": optim_adam,
    }
    optim = multi_transform(transforms, label_params)
    return optim


@partial(jax.jit, static_argnames=("k"))
def inverse_scale_svd(G, k=0.7):
    """G -> U @ exp( -k*transform(S) ) @ V^T,
    where transform(S) maps min(S) -> 0 and max(S) -> [0,1]
    """
    U, S, Vh = jnp.linalg.svd(G)

    # Normalization w.r.t. ell-p vector norm.
    # S = (S - jnp.min(S)) / (jnp.max(S) - jnp.min(S))
    S = (S - jnp.min(S)) / jnp.max(S)
    S = jnp.exp(-k*S)

    # SVD on G and tranlate to diagonal matrix.
    k = min(G.shape[0], G.shape[1])
    S = jnp.zeros_like(G).at[jnp.arange(k), jnp.arange(k)].set(S)

    return U@S@Vh


def muon_inverse(
        learning_rate: optax.ScalarOrSchedule = 0.05,
        momentum: float = 0.95,
        nesterov: bool = True,
        inverse_k: float = 0.693,
        ns_steps: int = 6,
        scale_rms: bool = False,
        adam_lr: optax.ScalarOrSchedule = 3e-4,
        adam_beta1: float = 0.95,
        adam_beta2: float = 0.95,
        adam_eps: float = 1e-8,
        adam_wd: float = 0.0,
        label_params: LabelParamsFn | None = None
) -> optax.GradientTransformation:
    """Muon with newton-schulz replaced with normalization
    w.r.t. Schatten-p norm. p=inf recovers muon, and p=2 recovers normalized by frobenius norm.
    """
    if label_params is None:
        label_params = muon_label_params_default    # use default muon partition.
        
    def normalize(G):
        G = inverse_scale_svd(G, inverse_k)
        if scale_rms:
            # explicit RMS normalization
            G = G * (G.shape[0]*G.shape[1])**0.5 / jnp.linalg.norm(G)
        else:
            # default muon scaling
            G = G * max(1, G.shape[0]/G.shape[1])**0.5
        return G
    optim_muon = optax.chain(
        optax.trace(decay=momentum, nesterov=nesterov),
        scale_by_function(normalize),
        optax.scale_by_learning_rate(learning_rate),
    )
    optim_adam = adamw(
        learning_rate=adam_lr,
        beta1=adam_beta1,
        beta2=adam_beta2,
        eps=adam_eps,
        weight_decay=adam_wd,
        use_nesterov=False,
    )
    transforms = {
        "muon": optim_muon,
        "adamw": optim_adam,
    }
    optim = multi_transform(transforms, label_params)
    return optim


inverse_newton_schulz_constants = {
    "2.3-0": [
        (3.1304, -4.0549,  1.8318),
        (1.1063, -0.1834, -0.0043),
        (5, 4)
    ],
    "4.6-0": [
        ( 2.7061, -2.9047,  1.1529),
        (-1.3690,  0.4209,  0.0554),
        (5, 4)
    ],
    "4.6-0.1": [
        ( 3.1356, -3.2752,  1.1671),
        (-1.4120,  0.6515, -0.1686),
        (5, 4)
    ],
    "10-0.1": [
        ( 3.1981, -3.4250,  1.2386),
        (-1.6111,  0.7298, -0.0578),
        (5, 4)
    ]
}

@partial(jax.jit, static_argnames=("k"))
def inverse_scale_newton_schulz(
        G: Array, 
        k: str="", 
) -> Array:
    """Approximates exp(-2.3*x) based on

    https://www.desmos.com/calculator/mhlbv7tirz.

    Muon reaches f(x)=0.85 level from x>=0.002, while this function 
    starts from 0.003.

    More precisely, this function returns G such that
    G/|G|_2 ~= G'/|G'|_2, where G' is the output of `inverse_scale_svd` with k=2.3
    """
    assert G.ndim == 2

    # NOTE: do we need to adapt to bfloat16 as in the OG repo?
    if not k:
        k = "2.3-0"
    if k in inverse_newton_schulz_constants:
        (a1, b1, c1), (a2, b2, c2), (N1, N2) = inverse_newton_schulz_constants[k]
    else:
        raise ValueError(f"k='{k}' is an invalid argument.")
    eps = 1e-7

    X = G
    if G.shape[0] > G.shape[1]:
        X = X.T

    # Initial normalization of X to bound |S|.
    X /= (jnp.linalg.norm(X, ord="fro") + eps)
    def fn1(i, val):
        X = val
        A = X @ X.T
        B = b1 * A + c1 * A @ A
        return a1 * X + B @ X
    X1 = jax.lax.fori_loop(
        0, N1, fn1, X
    )
    def fn2(i, val):
        X = val
        A = X @ X.T
        B = b2 * A + c2 * A @ A
        return a2 * X + B @ X
    X2 = jax.lax.fori_loop(
        0, N2, fn2, X
    )
    X = X1 - X2

    if G.shape[0] > G.shape[1]:
        X = X.T
    return X


def muon_inverse_ns(
        learning_rate: optax.ScalarOrSchedule = 0.05,
        momentum: float = 0.95,
        nesterov: bool = True,
        scale_rms: bool = True,
        newton_schulz_k: str = "",
        adam_lr: optax.ScalarOrSchedule = 3e-4,
        adam_beta1: float = 0.95,
        adam_beta2: float = 0.95,
        adam_eps: float = 1e-8,
        adam_wd: float = 0.0,
        label_params: LabelParamsFn | None = None
) -> optax.GradientTransformation:
    """Fast implementation of muon_inverse with k=2.3,
    while SVD is replaced with newton-schulz type polynomial approximation.
    """
    if label_params is None:
        label_params = muon_label_params_default    # use default muon partition.
        
    def normalize(G):
        G = inverse_scale_newton_schulz(G, k=newton_schulz_k)
        if scale_rms:
            # explicit RMS normalization
            G = G * (G.shape[0]*G.shape[1])**0.5 / jnp.linalg.norm(G)
        else:
            # default muon scaling
            G = G * max(1, G.shape[0]/G.shape[1])**0.5
        return G
    optim_muon = optax.chain(
        optax.trace(decay=momentum, nesterov=nesterov),
        scale_by_function(normalize),
        optax.scale_by_learning_rate(learning_rate),
    )
    optim_adam = adamw(
        learning_rate=adam_lr,
        beta1=adam_beta1,
        beta2=adam_beta2,
        eps=adam_eps,
        weight_decay=adam_wd,
        use_nesterov=False,
    )
    transforms = {
        "muon": optim_muon,
        "adamw": optim_adam,
    }
    optim = multi_transform(transforms, label_params)
    return optim


class NewtonSchulzConfig(NamedTuple):
    const: Tuple[Tuple[float]]
    base: float
    scale: float


NEWTON_SCHULZ_COMPONENTS = {
    "one": NewtonSchulzConfig(
        const=(
            (498, -1048, 628),
            (458, -808, 416),
            (446, -784, 402),
            (412, -760, 414),
            (340, -556, 328),
            (260, -276, 130),
        ),
        base=128,
        scale=1.5,
    ),
}


NEWTON_SCHULZ_CONFIGS = {
    # Stabilized muon constants.
    "muon": (
        NEWTON_SCHULZ_COMPONENTS["one"],
    ),
    # OG muon constants from Keller Jordan's repo.
    "muon_OG": (
        NewtonSchulzConfig(
            const=(
                (3.4445, -4.7750,  2.0315),
            ) * 6,
            base=1,
            scale=1,
        ),
    ),
    # https://www.desmos.com/calculator/jbxzf3ovgd
    "k4.6": (
        NEWTON_SCHULZ_COMPONENTS["one"],
        NewtonSchulzConfig(
            const=(
                (348, -382, 212),
                (69, -34, 19.125),
                (342, -468, 282),
            ),
            base=128,
            scale=-1,
        ),
    ),
    # https://www.desmos.com/calculator/fet7utqjfd
    "k2.3": (
        NEWTON_SCHULZ_COMPONENTS["ones"],
        NewtonSchulzConfig(
            const=(
                (96, -88.5, 84.5),
                (334, -564, 440),
            ),
            base=128,
            scale=-1,
        )
    )
}


@partial(jax.jit, static_argnames=("configs",))
def stable_newton_schulz(
        G: jnp.ndarray,
        configs: Tuple[NewtonSchulzConfig, ...] | None = None,
):
    """Stabilized newton-schulz approximation.
    
    Constants are optimized from 
    https://gist.github.com/YouJiacheng/393c90cbdc23b09d5688815ba382288b
    """
    if configs is None:
        configs = NEWTON_SCHULZ_CONFIGS["muon"]
    eps = 1e-7
    X = G
    if G.shape[0] > G.shape[1]:
        X = X.T
    X /= (jnp.linalg.norm(X, ord="fro") + eps)

    def apply_config(X, conf):
        const = jnp.array(conf.const, dtype=jnp.bfloat16)
        N = len(conf.const)
        
        def body_fn(i, X):
            a, b, c = const[i] / conf.base
            A = X @ X.T
            B = b * A + c * (A @ A)
            return a * X + B @ X
        
        return conf.scale * jax.lax.fori_loop(0, N, body_fn, X)

    res = jnp.zeros_like(X)
    for conf in configs:
        res = res + apply_config(X, conf)

    if G.shape[0] > G.shape[1]:
        res = res.T
    return res


def muon_stable(
        learning_rate: optax.ScalarOrSchedule = 0.03,
        momentum: float = 0.95,
        nesterov: bool = True,
        scale_rms: bool = True,
        ns_name: str = "",
        adam_lr: optax.ScalarOrSchedule = 0.03,
        adam_beta1: float = 0.95,
        adam_beta2: float = 0.95,
        adam_eps: float = 1e-8,
        adam_wd: float = 0.0,
        label_params: LabelParamsFn | None = None
) -> optax.GradientTransformation:
    """Muon with general and stabilized Newton-Schulz."""
    if label_params is None:
        label_params = muon_label_params_default    # use default muon partition.
    if not ns_name:
        ns_name = "muon"
    if ns_name not in NEWTON_SCHULZ_CONFIGS:
        raise ValueError(f"cannot find ns_name='{ns_name}' in NEWTON_SCHULZ_CONSTANTS.")
        
    def normalize(G):
        G = stable_newton_schulz(G, configs=tuple(NEWTON_SCHULZ_CONFIGS[ns_name]))
        if scale_rms:
            # explicit RMS normalization
            G = G * (G.shape[0]*G.shape[1])**0.5 / jnp.linalg.norm(G)
        else:
            # default muon scaling
            G = G * max(1, G.shape[0]/G.shape[1])**0.5
        return G
    optim_muon = optax.chain(
        optax.trace(decay=momentum, nesterov=nesterov),
        scale_by_function(normalize),
        optax.scale_by_learning_rate(learning_rate),
    )
    optim_adam = adamw(
        learning_rate=adam_lr,
        beta1=adam_beta1,
        beta2=adam_beta2,
        eps=adam_eps,
        weight_decay=adam_wd,
        use_nesterov=False,
    )
    transforms = {
        "muon": optim_muon,
        "adamw": optim_adam,
    }
    optim = multi_transform(transforms, label_params)
    return optim