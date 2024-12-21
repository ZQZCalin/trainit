"""Optimizers."""

import jax
from jax import numpy as jnp
from jax import random as jr
from jax import tree_util as jtu
import chex
import optax
from optax import Updates, Params, OptState, ScalarOrSchedule, GradientTransformation
from typing import Any, Tuple, NamedTuple, Optional, Callable
import sys
sys.path.append('../trainit')
import utils
import scheduler
import logstate
import optimizers.base as base


class PolarDescentState(NamedTuple):
    """polar_descent state."""
    dir_state: OptState
    mag_state: OptState


def polar_descent(
    direction_optim: GradientTransformation,
    magnitude_optim: GradientTransformation,
) -> GradientTransformation:
    """Decomposes direction and magnitude and updates each separately.

    Updates both direction and magnitude with AdamW. 
    Specifically, parameter x is decomposed into x = r*u where r=|x| and u=x/|x|.
    The gradient is then decomposed into df/dr = <grad, u> and df/du = grad*r.
    In the future, this should support any base optimizer (just like any blackbox reduction).

    Args:
        direction_lr: Learning rate schedule for direction adamw.
        magnitude_lr: Learning rate schedule for magnitude adamw.
        b1: First order momentum constant. Defaults to 0.9.
        b2: Second order momentum constant. Defaults to 0.999.
        weight_decay: Weight decay constant. Defaults to 0.01.

    Returns:
        A `GradientTransformation` object.
    """

    def init_fn(params):
        mag = utils.tree_l2_norm(params)
        dir = utils.tree_scalar_multiply(params, 1/mag)
        return PolarDescentState(
            dir_state=direction_optim.init(dir),
            mag_state=magnitude_optim.init(mag),
        )
    
    def update_fn(updates, state, params):
        # Decompose parameters and gradients.
        mag = utils.tree_l2_norm(params)
        dir = utils.tree_scalar_multiply(params, 1/mag)
        dir_grads = utils.tree_scalar_multiply(updates, mag)
        mag_grads = utils.tree_inner_product(updates, dir)

        # Update direction.
        dir_updates, dir_state = direction_optim.update(dir_grads, state.dir_state, dir)
        new_dir = utils.tree_normalize(
            optax.apply_updates(dir, dir_updates))      # project direction back to norm=1.

        # Update magnitude.
        mag_updates, mag_state = magnitude_optim.update(mag_grads, state.mag_state, mag)
        new_mag = optax.apply_updates(mag, mag_updates)

        # Combine direction and magnitude.
        new_params = utils.tree_scalar_multiply(new_dir, new_mag)
        new_updates = utils.tree_subtract(new_params, params)
        return new_updates, PolarDescentState(
            dir_state=dir_state,
            mag_state=mag_state,
        )
    
    return GradientTransformation(init_fn, update_fn)


class JumpTrajectoryState(NamedTuple):
    """jump_trajectory state."""
    count: chex.Array
    normal_state: OptState
    jump_state: OptState


# TODO: maybe we can implement something even smarter that automatically detects local minimum and performs a tangential jump.
def jump_trajectory(
    normal_optim: GradientTransformation,
    jump_optim: GradientTransformation,
    normal_steps: int = 4500,
    jump_steps: int = 500,
) -> GradientTransformation:
    
    total_steps = normal_steps + jump_steps

    def init_fn(params):
        return JumpTrajectoryState(
            count=jnp.zeros([], jnp.int32),
            normal_state=normal_optim.init(params),
            jump_state=jump_optim.init(jnp.zeros([])),
        )

    def update_fn(updates, state, params):
        net_count = jnp.mod(state.count, total_steps)
        normal_phase = net_count < normal_steps
        # Re-initialize normal / jump optimizer per stage change.
        normal_state = jax.lax.cond(
            net_count == 0,
            lambda _: normal_optim.init(params),
            lambda _: state.normal_state,
            operand=None
        )
        jump_state = jax.lax.cond(
            net_count == normal_steps,
            lambda _: jump_optim.init(utils.tree_l2_norm(params)),
            lambda _: state.jump_state,
            operand=None
        )
        # Update normal / jump optimizer.
        def normal_update(_):
            new_updates, new_normal_state = normal_optim.update(updates, normal_state, params)
            return new_updates, new_normal_state, jump_state
        def jump_update(_):
            mag = utils.tree_l2_norm(params)
            dir = utils.tree_scalar_multiply(params, 1/mag)
            mag_updates, new_jump_state = jump_optim.update(
                utils.tree_inner_product(updates, dir), jump_state, mag
            )
            new_updates = utils.tree_scalar_multiply(dir, mag_updates)
            return new_updates, normal_state, new_jump_state
        new_updates, normal_state, jump_state = jax.lax.cond(
            normal_phase, normal_update, jump_update, operand=None)
        return new_updates, JumpTrajectoryState(
            count=optax.safe_int32_increment(state.count),
            normal_state=normal_state,
            jump_state=jump_state,
        )
    
    return GradientTransformation(init_fn, update_fn)


if __name__ == "__main__":
    print(jnp.zeros([]) == 0)