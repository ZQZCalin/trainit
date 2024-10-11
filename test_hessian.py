# Checks the properties of Hessian along training trajectories.
# ===========================================================================

import logging
import warnings

import jax
from jax import numpy as jnp
from jax import random as jr
from jax import tree_util as jtu

import optax
from optax import GradientTransformation, Updates, OptState, Params
import equinox as eqx

import transformers

from jaxamp import amp, DynamicScalerState, dynamic_scale_value_and_grad
from typing import List, Tuple, Any, Optional, Sequence, Union, NamedTuple, Callable
from jaxtyping import Array, PRNGKeyArray

import tqdm
import wandb
import hydra
from omegaconf import OmegaConf, DictConfig

import utils
import logstate
from logger import TimeKeeper, RateLimitedWandbLog
from model.mingpt import GPT
from loader.lm_loader import get_lm_loader_next_token, shift_labels
from loadit import LoadIt, chunk_shuffle

import os, sys
sys.path.append('./optimizer')
from optimizer.o2nc import deterministic_online_nonconvex, wrap_random_scaling
import optimizer.online_learners as ol
import optimizer.benchmark as benchmark
import optimizer.scheduler as scheduler
import optimizer.optim as optim

import random
import numpy as np
import torch

import serialize.serializer as serializer

from train_jax import TrainState, MiniBatch, \
    init_tokenizer, init_aux_state, load_lm_data, init_model, init_optimizer, init_config, \
    back_prop, update_aux_state, init_train_state
from train_jax import loss_fn
from utils import get_dtype, get_accuracy


def power_iteration(
    grad_f: Callable[[Array], Array],
    x: Array,
    iter: int = 100,
    key: Optional[PRNGKeyArray] = None,
) -> Tuple[Array, Array]:
    """Performs power iteration with Hessian of f(x).
    
    Returns:
        A tuple of max eigenvalue and corresponding eigenvector.
    """
    if key is None:
        key = jr.PRNGKey(42)
    v = utils.random_unit_vector(x, key=key)
    eigenvector = jax.lax.fori_loop(
        lower=0, upper=iter, 
        body_fun=lambda i, val: utils.tree_normalize(jax.jvp(grad_f, (x,), (val,))[1]),
        init_val=v
    )
    eigenvalue = utils.tree_norm(jax.jvp(grad_f, (x,), (eigenvector,))[1])   # NOTE: jax.jvp returns (f(x), Hessian-vector product)
    return eigenvalue, eigenvector


def test_power_iteration():
    """Simple test case."""
    H = jnp.array([[2., 0], [0, 1.]])
    f = lambda x: x.T @ H @ x

    x = utils.tree_normalize(jnp.array([1., 3.]))
    jit_power_iteration = eqx.filter_jit(
        jtu.Partial(power_iteration, iter=100)
    )
    e, v = jit_power_iteration(jax.grad(f), x)
    print(f"max eigen-value is {e}, max eigen-vector is ", v)

    return


class SpectrumState(NamedTuple):
    max_eigenvalue: Array
    max_eigenvector: Array
    min_eigenvalue: Array
    min_eigenvector: Array


def coupled_power_iteration(
    train_state: TrainState,
    batches: MiniBatch,
    config: DictConfig,
    iter: int = 100,
    logger: Optional[RateLimitedWandbLog] = None,
) -> SpectrumState:
    """Power iteration modified for back_prop and train_state.
    
    Computes the pair of maximum and minimum eigen-pairs.
    """
    key, new_key = jr.split(train_state.train_key)
    params = eqx.filter(train_state.model, eqx.is_array)
    delta = utils.random_unit_vector(params, key=key)

    spectrum_state = SpectrumState(
        max_eigenvalue=jnp.zeros([]),
        max_eigenvector=delta,
        min_eigenvalue=jnp.zeros([]),
        min_eigenvector=delta,
    )
    
    def hessian(train_state, x, v, a, b):
        """Computes Jacobian vector product of (bI+aH)*v where H is hessian at x.
        Defaults to a=1 and b=0 (i.e., the actual hessian). Set a=-1 and b=lambda_max for the other eigen-pair.
        """
        def grad_f(x):
            _train_state = train_state._replace(model=eqx.combine(x, train_state.model))
            _train_state, _, _, grads = back_prop(_train_state, batches, config)
            res = jtu.tree_map(
                lambda g, _x: a*g + b*_x, grads, x
            )
            return res, _train_state
        _, v, train_state = jax.jvp(grad_f, (x,), (v,), has_aux=True)
        return train_state, v

    # First eigen-pair.
    if config.logging.wandb_project is not None:
        wandb.init(project=config.logging.wandb_project)
        wandb.config.update(OmegaConf.to_container(config))
    jit_hessian = eqx.filter_jit(
        jtu.Partial(hessian, a=1, b=0)
    )
    v_prev = delta
    pbar = tqdm.tqdm(range(iter), total=iter)
    for it in pbar:
        train_state, v = jit_hessian(train_state, params, v_prev)
        lam = utils.tree_norm(v)
        v = utils.tree_normalize(v)
        similarity = utils.tree_cosine_similarity(v, v_prev)
        v_prev = v
        pbar.set_description(f"iter {it+1}/{iter}: eigen-value={lam}")
        # Log to wandb.
        metrics = {
            "eigenvalue": lam,
            "cos(vn,vn-1)": similarity,
        }
        if logger is not None:
            logger(
                metrics,
                step=it,
            )
    if similarity > 0:                              # positive max-eigenvalue
        spectrum_state = spectrum_state._replace(
            max_eigenvalue=lam,
            max_eigenvector=v,
        )
        a = -jnp.ones([])
    else:                                           # negative min-eigenvalue
        spectrum_state = spectrum_state._replace(
            min_eigenvalue=-lam,
            min_eigenvector=v,
        )
        a = jnp.ones([])
    b = lam + 1
    wandb.finish()

    # Second eigen-pair.
    if config.logging.wandb_project is not None:
        wandb.init(project=config.logging.wandb_project)
        wandb.config.update(OmegaConf.to_container(config))
    jit_hessian = eqx.filter_jit(
        jtu.Partial(hessian, a=a, b=b)
    )
    v_prev = delta
    pbar = tqdm.tqdm(range(iter), total=iter)
    for it in pbar:
        train_state, v = jit_hessian(train_state, params, v_prev)
        lam = utils.tree_norm(v)
        lam = (lam - b) * a
        v = utils.tree_normalize(v)
        similarity = utils.tree_cosine_similarity(v, v_prev)
        v_prev = v
        pbar.set_description(f"iter {it+1}/{iter}: eigen-value={lam}")
        # Log to wandb.
        metrics = {
            "eigenvalue": lam,
            "cos(vn,vn-1)": similarity,
        }
        if logger is not None:
            logger(
                metrics,
                step=it,
            )
    if similarity > 0:                              # positive max-eigen
        spectrum_state = spectrum_state._replace(
            max_eigenvalue=lam,
            max_eigenvector=v,
        )
    else:
        spectrum_state = spectrum_state._replace(
            min_eigenvalue=lam,
            min_eigenvector=v,
        )
    wandb.finish()

    return spectrum_state


def get_batches(dataloader, idx: int, num_batches: int, use_shift_labels: bool):
    batches = []
    for batch in dataloader[idx: idx+num_batches]:
        if use_shift_labels:
            batch = shift_labels(batch)
        input_ids = jnp.asarray(batch["input_ids"])
        labels = jnp.asarray(batch["labels"])
        batches.append((input_ids, labels))
    return batches


# USE \nabla f(x+v) - \nabla f(x) = H*v (finite approximation instead of exact jvp).

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(config: DictConfig) -> None:
    """A systematic testing on stochastic loss local landscape."""
    # For test purpose only, fix some configurations.
    # Pre-loading presets.
    test_config = config.test
    config.checkpoint.overwrite_config = False
    config = init_config(config)
    # Post-loading changes.
    config.test = test_config
    config.logging.wandb_project = "hessian_spectrum"
    config.train.use_amp = False
    logging.info(OmegaConf.to_yaml(config))

    train_state, optimizer, train_loader, tokenizer, limited_log = init_train_state(config)

    # Testing variables.
    num_batches = config.dataset.total_batch_size // config.dataset.batch_size
    idx = train_state.iteration * num_batches
    batches = get_batches(train_loader, idx, num_batches, config.dataset.shift_labels)

    # Temporary tests
    updates = train_state.aux_state.params_diff
    spectrum_state = SpectrumState(
        max_eigenvalue=jnp.zeros([]), 
        max_eigenvector=updates,
        min_eigenvalue=jnp.zeros([]),
        min_eigenvector=updates,
    )
    path = "checkpoint/hessian_spectrum/adam_benchmark/iter_100.ckpt"
    spectrum_state = serializer.load(path, spectrum_state)

    def hessian(train_state, x, v, a, b):
        """Computes Jacobian vector product of (bI+aH)*v where H is hessian at x.
        Defaults to a=1 and b=0 (i.e., the actual hessian). Set a=-1 and b=lambda_max for the other eigen-pair.
        """
        def grad_f(x):
            _train_state = train_state._replace(model=eqx.combine(x, train_state.model))
            _train_state, _, _, grads = back_prop(_train_state, batches, config)
            res = jtu.tree_map(
                lambda g, _x: a*g + b*_x, grads, x
            )
            return res, _train_state
        _, v, train_state = jax.jvp(grad_f, (x,), (v,), has_aux=True)
        return train_state, v

    x = eqx.filter(train_state.model, eqx.is_array)
    v = spectrum_state.min_eigenvector
    _, v_next = hessian(train_state, x, v, a=1, b=0)
    print(f"max eigenvalue: {spectrum_state.max_eigenvalue}; min: {spectrum_state.min_eigenvalue}")
    print(f"max norm: {utils.tree_norm(spectrum_state.max_eigenvector)}; min norm: {utils.tree_norm(spectrum_state.min_eigenvector)}")
    print(utils.tree_norm(v_next), utils.tree_cosine_similarity(v, v_next))


    raise KeyboardInterrupt()

    # Spectrum power iteration.
    spectrum_state = coupled_power_iteration(
        train_state, batches=batches, config=config, logger=limited_log, iter=20
    )
    path = config.test.hessian.save_path
    if not path:
        path = "checkpoint/hessian_spectrum/adam_benchmark/spectrum.ckpt"
    serializer.save(path, spectrum_state)
    

if __name__ == "__main__":
    # test_power_iteration()
    main()