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
from utils import tree_utils, log_utils
import online_learners as ol


SampleFunction = Callable[[chex.Array], chex.Numeric]
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
    logging:    log_utils.Log


# TODO: might need to re-implement this?
# Now I think about it, we don't need to wrap an optimizer with this.
# Instead, we can just use this as an additional stateless gradient transformation to be chained.
def wrap_random_scaling(
    gradient_transformation:    optax.GradientTransformation,
    random_scaling:             Optional[str] = None,
    use_importance_sampling:    bool = True,
    *,
    key: PRNGKeyArray,
) -> optax.GradientTransformation:
    """Wraps an optimizer by applying random scaling to its update."""
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
            logging=log_utils.Log(logging),
        )
    
    def update_fn(updates, state, params):
        updates = tree_utils.scalar_dot(updates, state.weight)
        updates, opt_state = gradient_transformation.update(updates, state.opt_state, params)
        key, new_key = jr.split(state.key)
        scalar = random_scaling_fn(key)
        weight = importance_sampling_fn(scalar)
        new_updates = tree_utils.scalar_dot(updates, scalar)
        update_norm = tree_utils.norm(updates)
        logging = {
            "update/params_norm": tree_utils.norm(params),
            "update/update_norm_pre_scaling": update_norm,
            "update/update_norm_post_scaling": update_norm*scalar,
            "update/random_scaling": scalar,
            "update/importance_sampling": weight,
        }
        return new_updates, WrapRandomScalingState(
            opt_state=opt_state,
            weight=weight,
            key=new_key,
            logging=log_utils.Log(logging),
        )
    
    return optax.GradientTransformation(init_fn, update_fn)


class OnlineToGradientTransformationState(NamedTuple):
    """deterministic_online_nonconvex state."""
    params: optax.Updates 
    state:  optax.OptState


def online_to_gradient_transformation(
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
        return OnlineToGradientTransformationState(
            params=params, state=online_learner.init(params))
    
    def update_fn(updates, state, params=None):
        del params
        new_params, state = online_learner.update(updates, state.state, state.params)
        return new_params, OnlineToGradientTransformationState(params=new_params, state=state)

    return optax.GradientTransformation(init_fn, update_fn)


def online_to_non_convex(
        online_learner: OnlineLearner,
) -> optax.GradientTransformation:
    """Wraps an `OnlineLearner` object with the online to non-convex framework."""
    return optax.chain(
        online_to_gradient_transformation(online_learner),
        wrap_random_scaling,
    )