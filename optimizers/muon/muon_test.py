"""Tests the muon optimizer."""

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import optax
import equinox as eqx
from omegaconf import OmegaConf
from utils import tree_utils
from optimizers.muon.muon import scale_by_muon, muon
from optimizers.muon.muon_laprop import label_gpt, muon_laprop
from optimizers.muon.mango import mango_label_gpt, mango
from models import summarize_model_parmas
from _src import init_language_model


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


def visualize_label_params():
    """A visualization util function that structures the partition of muon_laprop."""
    model = init_language_model(config=OmegaConf.load("conf/model/gpt.yaml"), key=jax.random.PRNGKey(42))
    summary = summarize_model_parmas(model, verbose=False)
    # labels = label_gpt(eqx.filter(model, eqx.is_array))
    labels = mango_label_gpt(eqx.filter(model, eqx.is_array))

    summary_list, _ = jtu.tree_flatten(summary)
    labels_list, _ = jtu.tree_flatten(labels)

    max_len = max(len(s) for s in summary_list)
    lines = [f"{s:<{max_len}} | {label}" for s, label in zip(summary_list, labels_list)]

    print("\n".join(lines))


def test_muon_laprop():
    model = init_language_model(config=OmegaConf.load("conf/model/gpt.yaml"), key=jax.random.PRNGKey(42))
    params = eqx.filter(model, eqx.is_array)

    optimizer = muon_laprop()
    opt_state = optimizer.init(params)

    for i in range(2):
        grads = tree_utils.normalize(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        print(f"iter {i+1}\n  >> grads\n", jtu.tree_flatten(grads)[0])
        print(f"  >> updates\n", jtu.tree_flatten(updates)[0])
        print(f"  >> new params\n", jtu.tree_flatten(params)[0])


def test_mango():
    model = init_language_model(config=OmegaConf.load("conf/model/gpt.yaml"), key=jax.random.PRNGKey(42))
    params = eqx.filter(model, eqx.is_array)

    optimizer = mango()
    opt_state = optimizer.init(params)

    for i in range(2):
        grads = tree_utils.normalize(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        print(f"iter {i+1}\n  >> grads\n", jtu.tree_flatten(grads)[0])
        print(f"  >> updates\n", jtu.tree_flatten(updates)[0])
        print(f"  >> new params\n", jtu.tree_flatten(params)[0])


if __name__ == "__main__":
    test_muon()