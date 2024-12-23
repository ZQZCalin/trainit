"""Online-to-non-convex Conversion."""

import jax
from jax import numpy as jnp
from jax import random as jr
from jax import tree_util as jtu
import optax
import chex
import jaxtyping
from jaxtyping import PRNGKeyArray
from typing import Any, Tuple, NamedTuple, Optional, Callable
from online_learners import OnlineLearner
import sys
sys.path.append('../trainit')
from utils import tree_scalar_multiply, tree_norm
from logger import RateLimitedWandbLog
import logstate
import online_learners as ol


SampleFunction = Callable[[chex.Array], chex.Numeric]


# class ScaleByRandomState(NamedTuple):
#     """scale_by_random state."""
#     key: chex.Array


# def scale_by_random(
#     sample_fn: SampleFunction,
#     seed: int = 0,
# ) -> GradientTransformation:
#     """Scales the update by a random variable.
    
#     Args:
#         sample_fn: A function that receives a PRNGKeyArray and returns a random number.
#         seed (int): Seed for jax.random.PRNGKey.
    
#     Returns:
#         A `GradientTransform` object.
#     """

#     def init_fn(params=None):
#         del params
#         return ScaleByRandomState(key=jr.PRNGKey(seed))
    
#     def update_fn(updates, state, params=None):
#         del params
#         key1, key2 = jr.split(state.key)
#         scaling = sample_fn(key1)
#         new_updates = tree_scalar_multiply(updates, scaling)
#         return new_updates, ScaleByRandomState(key=key2)
    
#     return GradientTransformation(init_fn, update_fn)


# def scale_by_exponential(
#     lam: float = 1.0,
#     seed: int = 0,
# ) -> GradientTransformation:
#     """Scales the update by exponential random variable with mean = lam.

#     Args:
#         lam (float): Mean of sampled random variable. Defaults to 1.0.
#         seed (int): Seed for jax.random.PRNGKey.

#     Returns:
#         A `GradientTransformation` object.
#     """
#     sample_fn = lambda key: lam * jr.exponential(key)
#     return scale_by_random(sample_fn, seed)


# class OnlineNonconvexState(NamedTuple):
#     """online to nonconvex state."""
#     ol_params: Params
#     ol_state: OptState
#     key: chex.Array
#     logging: logstate.Log


# # deprecated: please use deterministic_online_nonconvex together with wrap_random_scaling, where
# # the former handles wrapping online learners and the latter handles random scaling.
# def online_nonconvex(
#     online_learner: OnlineLearner,
#     random_scaling: SampleFunction = None,
#     seed: int = 0,
# ) -> GradientTransformation:
#     """General Online-to-non-convex conversion.

#     For simplicity of logging message, this function combines `scale_by_random` and `wrap_online_learner`.
#     See documentations of the two functions for more details.

#     Args:
#         online_learner: Online learner subroutine.
#         random_scaling: Function to sample random scalar. Defaults to exponential scaling.
#         seed: PRNGKey to generate random scalar.
#     """
    
#     # Scaling defaults to exponential scaling.
#     exponential_scaling = lambda key: jr.exponential(key)
#     if random_scaling is None:
#         random_scaling = exponential_scaling
    
#     def init_fn(params):
#         # NOTE: For now, I assume online learner parameters are always initialized to zeros.
#         # ol_params = jtu.tree_map(lambda p: jnp.zeros_like(p, dtype=jnp.float32), params)
#         ol_params = jtu.tree_map(jnp.zeros_like, params)
#         ol_state = online_learner.init(ol_params)
#         key = jr.PRNGKey(seed)
#         logging = logstate.Log(
#             {
#                 "update/random_scaling": 0.0,
#                 "update/norm_pre_scaling": 0.0,
#                 "update/norm_post_scaling": 0.0,
#             }
#         )
#         return OnlineNonconvexState(
#             ol_params=ol_params,
#             ol_state=ol_state, 
#             key=key, 
#             logging=logging
#         )
    
#     def update_fn(updates, state, params=None):
#         del params
#         # Update online learner.
#         ol_params, ol_state = online_learner.update(updates, state.ol_state, state.ol_params)
#         norm_pre_scaling = tree_norm(ol_params)
#         # Apply random scaling.
#         key, new_key = jr.split(state.key)
#         scaling = random_scaling(key)
#         new_updates = tree_scalar_multiply(ol_params, scaling)
#         norm_post_scaling = scaling * norm_pre_scaling
#         return new_updates, OnlineNonconvexState(
#             ol_params=ol_params,
#             ol_state=ol_state,
#             key=new_key,
#             logging=logstate.Log({
#                 "update/random_scaling": scaling,
#                 "update/norm_pre_scaling": norm_pre_scaling,
#                 "update/norm_post_scaling": norm_post_scaling,
#             })
#         )
    
#     return GradientTransformation(init_fn, update_fn)


class DeterministicOnlineNonconvexState(NamedTuple):
    """deterministic_online_nonconvex state."""
    params: optax.Updates 
    state:  optax.OptState


def deterministic_online_nonconvex(
    online_learner: OnlineLearner
) -> optax.GradientTransformation:
    """Wraps an OnlineLearner object into a GradientTransformation object.

    Stores the online learner parameters, which is different from the model parameter. 
    This wrapper allows us to separate random scaling from online-nonconvex.

    Args:
        online_learner: An `OnlineLearner` object to be wrapped.

    Returns:
        A wrapped `GradientTransformation` object.
    """

    def init_fn(params, init_zero=True):
        """If init_zero is true, initialize to zero array; otherwise, initialize to params."""
        if init_zero:
            params = jtu.tree_map(jnp.zeros_like, params)
        return DeterministicOnlineNonconvexState(
            params=params, state=online_learner.init(params))
    
    def update_fn(updates, state, params=None):
        del params
        new_params, state = online_learner.update(updates, state.state, state.params)
        return new_params, DeterministicOnlineNonconvexState(params=new_params, state=state)

    return optax.GradientTransformation(init_fn, update_fn)


RandomScalingFn = Callable[[jaxtyping.PRNGKeyArray], chex.Numeric]
ImportanceSamplingFn = Callable[[chex.Numeric], chex.Numeric]

def get_random_scaling(
    name: Optional[str] = None,
    **kwargs,
) -> Tuple[RandomScalingFn, ImportanceSamplingFn]:
    """Returns a tuple of random scaling and importance sampling."""
    exponential_rs = lambda key: jr.exponential(key)
    exponential_is = lambda s: jnp.ones([])
    uniform_rs = lambda key, low, high: jr.uniform(key, minval=low, maxval=high)
    uniform_is = lambda s, low, high: high - s
    null_rs = lambda key: jnp.ones([])
    null_is = lambda s: jnp.ones([])

    if name == "exponential":
        return exponential_rs, exponential_is
    if name == "uniform":
        # for now, we always fix low=0 and high=2
        low, high = 0, 2
        return lambda key: uniform_rs(key, low, high), lambda s: uniform_is(s, low, high)
    else:
        return null_rs, null_is


class WrapRandomScalingState(NamedTuple):
    """wrap_random_scaling state."""
    opt_state:  optax.OptState
    weight:     jax.Array
    key:        jax.Array
    logging:    logstate.Log


def wrap_random_scaling(
    gradient_transformation:    optax.GradientTransformation,
    random_scaling:             Optional[str] = None,
    use_importance_sampling:    bool = True,
    *,
    key: PRNGKeyArray,
) -> optax.GradientTransformation:
    
    random_scaling_fn, importance_sampling_fn = get_random_scaling(random_scaling)
    if not use_importance_sampling:
        importance_sampling_fn = lambda s: jnp.ones([])

    def init_fn(params):
        logging = {
            "update/params_norm": jnp.zeros([]),
            "update/update_norm_pre_scaling": jnp.zeros([]),
            "update/update_norm_post_scaling": jnp.zeros([]),
            "update/random_scaling": jnp.ones([]),      # random scalar should be initialized to 1 (to avoid divide by 0 issue).
            "update/importance_sampling": jnp.ones([]),
        }
        return WrapRandomScalingState(
            opt_state=gradient_transformation.init(params),
            weight=jnp.ones([]),
            key=key,
            logging=logstate.Log(logging),
        )
    
    def update_fn(updates, state, params):
        updates = tree_scalar_multiply(updates, state.weight)
        updates, opt_state = gradient_transformation.update(updates, state.opt_state, params)
        key, new_key = jr.split(state.key)
        scalar = random_scaling_fn(key)
        weight = importance_sampling_fn(scalar)
        new_updates = tree_scalar_multiply(updates, scalar)
        update_norm = tree_norm(updates)
        logging = {
            "update/params_norm": tree_norm(params),
            "update/update_norm_pre_scaling": update_norm,
            "update/update_norm_post_scaling": update_norm*scalar,
            "update/random_scaling": scalar,
            "update/importance_sampling": weight,
        }
        return new_updates, WrapRandomScalingState(
            opt_state=opt_state,
            weight=weight,
            key=new_key,
            logging=logstate.Log(logging),
        )
    
    return optax.GradientTransformation(init_fn, update_fn)