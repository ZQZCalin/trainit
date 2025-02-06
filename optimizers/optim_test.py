"""Test optimizers."""

import jax
from jax import numpy as jnp
from jax import random as jr
from jax import tree_util as jtu

import optax

from optimizers import base
from optimizers import optim
from optimizers import schedule

from utils import tree_utils


def test_optimizer(
        optimizer: optax.GradientTransformation
):

    grad_clip = optax.clip_by_global_norm(10.0)
    optimizer = optax.chain(
        grad_clip,
        optax.apply_if_finite(optimizer, 15)
    )

    # We use a PyTree of 2d arrays to test muon.
    params = {
        'a': jnp.ones((3,2)), 
        'b': jnp.array([[1,2,3], [2,3,4]], dtype=jnp.float32),
    }
    print("initial params:\n", params)
    
    opt_state = optimizer.init(params)
    
    for i in range(3):
        grads = tree_utils.normalize(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        print(f"iter {i+1}\n  >> grads\n", grads)
        print(f"  >> updates\n", updates)
        print(f"  >> new params\n", params)


def test_sgdm():
    learning_rate = optax.linear_schedule(0.1, 0.01, 10000)

    optimizer = base.sgdm(learning_rate, beta=1.0, weight_decay=0.0)
    
    grad_clip = optax.clip_by_global_norm(10.0)
    optimizer = optax.chain(
        grad_clip,
        optax.apply_if_finite(optimizer, 15)
    )

    test_optimizer(optimizer)
    

def test_jump():
    normal_lr = schedule.warmup_linear_decay_schedule(0.0, 3e-4, 450, 4500)
    normal_optim = base.adamw(
        normal_lr, 0.9, 0.999, 1e-8, 0.0
    )
    jump_lr = schedule.warmup_linear_decay_schedule(0.0, 1e-6, 50, 500)
    jump_optim = base.adamw(
        jump_lr, 0.9, 0.999, 1e-8, 0.0
    )
    optimizer = optim.jump_trajectory(
        normal_optim, jump_optim, 4500, 500
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(10.0),
        optax.apply_if_finite(optimizer, 15)
    )

    test_optimizer(optimizer)


def test_jump():
    normal_lr = schedule.warmup_linear_decay_schedule(0.0, 3e-4, 450, 4500)
    normal_optim = base.adamw(
        normal_lr, 0.9, 0.999, 1e-8, 0.0
    )
    jump_lr = schedule.warmup_linear_decay_schedule(0.0, 1e-6, 50, 500)
    jump_optim = base.adamw(
        jump_lr, 0.9, 0.999, 1e-8, 0.0
    )
    optimizer = optim.jump_trajectory(
        normal_optim, jump_optim, 4500, 500
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(10.0),
        optax.apply_if_finite(optimizer, 15)
    )

    test_optimizer(optimizer)


def test_adam_wd():
    params = {
        'a': [jnp.array(1.), jnp.array(2.)],  # List of arrays
        'b': (jnp.array(3.), jnp.array(4.)),  # Tuple of arrays
        'c': {'d': jnp.array(5.)}  # Nested dictionary with an array
    }
    grads = jtu.tree_map(jnp.ones_like, params)
    adam = base.adamw(
        learning_rate=0.01, weight_decay=params
    )
    opt_state = adam.init(params)
    adam.update(grads, opt_state, params)


def test_sgdm_wd():
    params = {
        'a': [jnp.array(1.), jnp.array(2.)],  # List of arrays
        'b': (jnp.array(3.), jnp.array(4.)),  # Tuple of arrays
        'c': {'d': jnp.array(5.)}  # Nested dictionary with an array
    }
    grads = jtu.tree_map(jnp.ones_like, params)
    sgdm = base.sgdm(
        learning_rate=0.01, weight_decay=params
    )
    opt_state = sgdm.init(params)
    sgdm.update(grads, opt_state, params)


if __name__ == "__main__":
    # test_sgdm()
    # test_jump()
    # test_adam_wd()
    test_sgdm_wd()