"""Tests the muon optimizer."""

import jax.numpy as jnp
import optax
from utils import tree_utils
from optimizers.muon.muon import scale_by_muon, muon


def test_scale_by_muon():
    optimizer = scale_by_muon(
        learning_rate=1.0
    )

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


def test_muon():
    optimizer = muon(
        learning_rate=1.0,
        adam_lr=1.0,
    )

    grad_clip = optax.clip_by_global_norm(10.0)
    optimizer = optax.chain(
        grad_clip,
        optax.apply_if_finite(optimizer, 15)
    )

    # We use a PyTree of mix of 1d and 2d arrays to test muon.
    params = {
        'a': jnp.ones((3,)), 
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


if __name__ == "__main__":
    test_muon()