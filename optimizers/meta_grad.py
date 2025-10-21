import optax
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from typing import NamedTuple, Callable
from jaxtyping import Array, PyTree


MetaGradRegularizationFn = Callable[[Array], Array]


class MetaGradState(NamedTuple):
    """wrap_with_meta_gradient state."""
    lr: Array
    last_updates: PyTree


def wrap_with_meta_gradient(
        base_lr: float,
        clip_min: float = 1e-8,
        clip_max: float = float("inf"),
        learning_rate: optax.ScalarOrSchedule = 1e-6,
        regularization_fn: MetaGradRegularizationFn = None,
) -> optax.GradientTransformation:
    """Wrap an optimizer with meta-gradient for lr update.
    
    This should be wrapped around any optimizer such that the input updates
    represents the gradients (or clipped gradients).

    Args:
        base_lr: base learning rate to start from
        clip_min: minimum value of lr
        clip_max: maximum value of lr
        learning_rate: lr schedule for meta-gradient descent
        regularization_fn: a regularization function that maps input lr into gradient of lr

    ..Example::

        # TBD
    """

    assert regularization_fn is not None

    def init_fn(params):
        return MetaGradState(
            lr=jnp.asarray(base_lr, dtype=jnp.float32),
            last_updates=jtu.tree_map(jnp.zeros_like, params)
        )
    
    def update_fn(updates, state, params=None):
        del params
        lr = state.lr
        last_updates = state.last_updates


        return updates, MetaGradState(
            lr=None,
            last_updates=updates,
        )
    
    return optax.GradientTransformation(init_fn, update_fn)