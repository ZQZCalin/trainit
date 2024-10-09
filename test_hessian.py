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


def power_iteration_back_prop(
    train_state: TrainState,
    batches: MiniBatch,
    config: DictConfig,
    spectrum: int = 0,
    iter: int = 100,
    logger: Optional[RateLimitedWandbLog] = None,
):
    """Power iteration modified for back_prop and train_state."""
    key, new_key = jr.split(train_state.train_key)
    x = eqx.filter(train_state.model, eqx.is_array)
    v = utils.random_unit_vector(x, key=key)

    def get_grad_f(train_state):
        def grad_f(x):
            _train_state = train_state._replace(model=eqx.combine(x, train_state.model))
            _train_state, loss, acc, grads = back_prop(_train_state, batches, config)
            # Minimum eigen-pair.
            if spectrum == -1:
                max_eigenvalue = 6  # TODO: specify in argument
                grads = jtu.tree_map(
                    lambda g, _x: max_eigenvalue * _x - g, grads, x
                )
            return grads, _train_state
        return grad_f
    
    def hessian(train_state, x, v):
        return jax.jvp(get_grad_f(train_state), (x,), (v,), has_aux=True)
    jit_hessian = eqx.filter_jit(hessian)

    pbar = tqdm.tqdm(range(iter), total=iter)
    last_v = v
    for it in pbar:
        _, v, train_state = jit_hessian(train_state, x, v)
        eigenvalue = utils.tree_norm(v)
        v = utils.tree_normalize(v)
        pbar.set_description(f"iter {it+1}/{iter}: eigen-value={eigenvalue}")
        # Log to wandb.
        metrics = {
            "eigenvalue": eigenvalue,
            "cos(vn,vn-1)": utils.tree_cosine_similarity(v, last_v)
        }
        last_v = v
        if logger is not None:
            logger(
                metrics,
                step=it,
            )

    return eigenvalue, v


def get_batches(dataloader, idx: int, num_batches: int, use_shift_labels: bool):
    batches = []
    for batch in dataloader[idx: idx+num_batches]:
        if use_shift_labels:
            batch = shift_labels(batch)
        input_ids = jnp.asarray(batch["input_ids"])
        labels = jnp.asarray(batch["labels"])
        batches.append((input_ids, labels))
    return batches


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(config: DictConfig) -> None:
    """A systematic testing on stochastic loss local landscape."""
    # manually prevent overwriting test config
    test_config = config.test
    config = init_config(config)
    config.test = test_config
    config.logging.wandb_project = "hessian_spectrum"
    # config.logging.wandb_project = None
    # logging.info(OmegaConf.to_yaml(config))

    train_state, optimizer, train_loader, tokenizer, limited_log = init_train_state(config)

    # [Optional] uncomment to turn off AMP (this must be after loading the checkpoint)
    config.train.use_amp = False

    # Log to wandb.
    if config.logging.wandb_project is not None:
        wandb.init(project=config.logging.wandb_project, name=config.logging.wandb_name)
        wandb.config.update(OmegaConf.to_container(config))

    # Testing variables.
    num_batches = config.dataset.total_batch_size // config.dataset.batch_size
    idx = train_state.iteration * num_batches
    batches = get_batches(train_loader, idx, num_batches, config.dataset.shift_labels)

    spectrum = -1
    func = jtu.Partial(power_iteration_back_prop, batches=batches, config=config, iter=100, logger=limited_log, spectrum=spectrum)
    eigenvalue, eigenvector = func(train_state)
    # print(f"eigen-value: {eigenvalue}, eigen-vector: ", eigenvector)
    

if __name__ == "__main__":
    # test_power_iteration()
    main()