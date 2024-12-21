import jax
from jax import numpy as jnp
from jax import random as jr
from jax import tree_util as jtu
import optax
from optax import Updates, Params, OptState, ScalarOrSchedule, GradientTransformation
from typing import Any, Tuple, NamedTuple, Optional, Union, Callable, Protocol
from tqdm import tqdm
import sys
sys.path.append('../trainit')
import utils
import online_learners as ol
import wandb


def train_step(learner, loss_fn, params, opt_state):
    grads = jax.grad(loss_fn)(params)
    params, opt_state = learner.update(grads, opt_state, params)
    return params, opt_state


def train(learner, loss_fn, params, num_steps):
    opt_state = learner.init(params)
    pbar = tqdm(range(num_steps), total=num_steps)
    for step in pbar:
        params, opt_state = train_step(learner, loss_fn, params, opt_state)
        loss = loss_fn(params)
        pbar.set_description(f"Step {step}, Params: {params}, Loss: {loss:.2f}")
        wandb.log({
            "params/norm": utils.tree_norm(params),
            "loss": loss,
        })


def simple_train(learner, loss_fn, params, num_steps):
    opt_state = learner.init(params)
    for i in range(num_steps):
        params, opt_state = train_step(learner, loss_fn, params, opt_state)
        print(f"Step {i+1}; Params: {params}; Loss: {loss_fn(params):.2f}")


def test_kt_bettor(eps, G, train_full=True):
    """Test KT bettor on a 1d function f(x) = |x-100|."""
    # eps = 10
    # G = 1
    learner = ol.kt_bettor(eps=eps, G=G)
    print(f">>>Testing kt bettor with eps={eps} G={G} on f(x) = |x-100|...")

    # Define the loss function
    def loss_fn(x):
        return jnp.abs(x - 100)

    if train_full:
        train(learner, loss_fn, params=jnp.array(0.0), num_steps=1000)
    else:
        simple_train(learner, loss_fn, params=jnp.array(0.0), num_steps=20)


def test_kt_bettor_large_G(eps, G, train_full=True):
    """Test KT bettor on a 1d function f(x) = |x-100|^2."""
    # eps = 10
    # G = 100
    learner = ol.kt_bettor(eps=eps, G=G, log_reward=True)
    print(f">>>Testing kt bettor with eps={eps} G={G} on f(x) = |x-100|^2...")

    # Define the loss function
    def loss_fn(x):
        return (x - 100)**2

    if train_full:
        train(learner, loss_fn, params=jnp.array(0.0), num_steps=1000)
    else:
        simple_train(learner, loss_fn, params=jnp.array(0.0), num_steps=20)


def test_normalized_blackbox():
    print(">>>Testing normalized blackbox reduction on loss f(x) = \|x-100e\|")
    learner = ol.normalized_blackbox(
        base_learner=ol.kt_bettor(eps=10, G=1),
        beta=1.0,
        weight_decay=0.0,
        seed=0,
    )
    d = 3
    x_min = 100*jnp.ones(shape=d)

    def loss_fn(x):
        return jnp.sqrt(jnp.sum((x-x_min) * (x-x_min)))

    init_params = jnp.ones(shape=d)
    num_steps = 1000

    train(learner, loss_fn, init_params, num_steps)
    # params = init_params
    # opt_state = learner.init(params)
    # for i in range(20):
    #     params, opt_state = train_step(learner, loss_fn, params, opt_state)
    #     print(f"Step {i+1}; Params: {params}; Loss: {loss_fn(params)}")


def test_normalized_blackbox_large_G(eps, G, train_full=False):
    d = 3
    D = 100
    x_min = D*jnp.ones(shape=d)
    learner = ol.normalized_blackbox(
        base_learner=ol.kt_bettor(eps=eps, G=G),
        beta=1.0,
        weight_decay=0.0,
        seed=0,
    )
    print(f">>>Testing normalized blackbox reduction (eps={eps}, G={G:.2f}) on loss f(x) = \|x-{D}e\|^2")

    def loss_fn(x):
        return jnp.sum((x-x_min) * (x-x_min))

    init_params = jnp.zeros(shape=d)
    init_params = init_params.at[0].set(-100)

    if train_full:
        train(learner, loss_fn, params=init_params, num_steps=1000)
    else:
        simple_train(learner, loss_fn, params=init_params, num_steps=20)


def test_normalized_blackbox_per_layer_large_G(eps, G, train_full=False):
    d = 3
    D = 100
    learner = ol.normalized_blackbox(
        base_learner=ol.kt_bettor(eps=eps, G=G),
        beta=1.0,
        weight_decay=0.0,
        seed=0,
        per_layer=True,
    )
    print(f">>>Testing PER-LAYER normalized blackbox reduction (eps={eps}, G={G:.2f}) on loss f(x) = \|x-{D}e\|^2")

    init_params = [jnp.array([-100.]), jnp.zeros([]), jnp.zeros([])]

    def loss_fn(x):
        leaves, _ =jtu.tree_flatten(x)
        return sum(jnp.sum((leaf - D) ** 2) for leaf in leaves)

    if train_full:
        train(learner, loss_fn, params=init_params, num_steps=1000)
    else:
        simple_train(learner, loss_fn, params=init_params, num_steps=20)


def test_ogd_mirror_descent_large_G(train_full=False):
    d = 3
    D = 100
    print(f">>>Testing ogd_mirror_descent on loss f(x) = \|x-{D}e\|^2")

    x_min = D*jnp.ones(shape=d)
    T = 1000 if train_full else 20
    learning_rate = optax.linear_schedule(init_value=1.0, end_value=0.01, transition_steps=T)
    learner = ol.ogd_mirror_descent(
        learning_rate=learning_rate,
        beta=1.0,
        mu=0.0
    )

    def loss_fn(x):
        return jnp.sum((x-x_min) * (x-x_min))

    init_params = jnp.zeros(shape=d)
    init_params = init_params.at[0].set(-100)

    if train_full:
        train(learner, loss_fn, params=init_params, num_steps=T)
    else:
        simple_train(learner, loss_fn, params=init_params, num_steps=T)


def test_pfmd(train_full=False):
    d = 3
    D = 100
    print(f">>>Testing ogd_mirror_descent on loss f(x) = \|x-{D}e\|^2")

    x_min = D*jnp.ones(shape=d)
    T = 1000 if train_full else 20
    learner = ol.parameter_free_mirror_descent(
        G=800, eps=1, num_grids=1
    )

    def loss_fn(x):
        return jnp.sum((x-x_min) * (x-x_min))

    init_params = jnp.zeros(shape=d)

    if train_full:
        train(learner, loss_fn, params=init_params, num_steps=T)
    else:
        simple_train(learner, loss_fn, params=init_params, num_steps=1000)


if __name__ == "__main__":
    config = {
        "eps": 10,
        "G": 200,
    }
    # wandb.init(project="KT_Lipschitz", config=config)
    # test_kt_bettor(**config, train_full=True)
    # test_kt_bettor_large_G(**config, train_full=False)
    # test_normalized_blackbox()
    # test_normalized_blackbox_large_G(eps=10, G=200)
    # test_normalized_blackbox_per_layer_large_G(eps=10, G=100)
    # test_ogd_mirror_descent_large_G(train_full=False)
    test_pfmd()