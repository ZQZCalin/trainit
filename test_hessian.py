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
    back_prop, update_aux_state
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
    iter: int = 100,
):
    """Power iteration modified for back_prop and train_state."""
    key, new_key = jr.split(train_state.train_key)
    x = eqx.filter(train_state.model, eqx.is_array)
    v = utils.random_unit_vector(x, key=key)

    def get_grad_f(train_state):
        def grad_f(x):
            _train_state = train_state._replace(model=eqx.combine(x, train_state.model))
            _train_state, loss, acc, grads = back_prop(_train_state, batches, config)
            return grads, _train_state
        return grad_f

    def _one_step(i, val):
        train_state, v = val
        _, v, train_state = jax.jvp(get_grad_f(train_state), (x,), (v,), has_aux=True)
        v = utils.tree_normalize(v)
        return (train_state, v)
    
    train_state, eigenvector = jax.lax.fori_loop(
        lower=0, upper=iter, 
        body_fun=_one_step,
        init_val=(train_state, v)
    )
    eigenvalue = utils.tree_norm(jax.jvp(get_grad_f(train_state), (x,), (eigenvector,))[1])
    return eigenvalue, eigenvector


def get_batches(dataloader, idx: int, num_batches: int, use_shift_labels: bool):
    batches = []
    for batch in dataloader[idx: idx+num_batches]:
        if use_shift_labels:
            batch = shift_labels(batch)
        input_ids = jnp.asarray(batch["input_ids"])
        labels = jnp.asarray(batch["labels"])
        batches.append((input_ids, labels))
    return batches


# TODO: merge into train_jax.py
def init_train_state(
    config: DictConfig
) -> Tuple[TrainState, optax.GradientTransformation, Any, Any, RateLimitedWandbLog]:
    """Initializes / loads train state.
    
    Returns:
        A tuple of train state, optimizer, dataloader, tokenizer, wandb logger.
    """
    # Initialize wandb logger.
    if config.logging.wandb_project is not None:
        limited_log = RateLimitedWandbLog(config.logging.wandb_logs_per_sec)
    else:
        limited_log = None

    # Initialize model tokenizer.
    tokenizer = init_tokenizer(config)

    # Initialize dataloader.
    train_loader = load_lm_data(config, tokenizer)

    # Initialize random keys
    seed = config.random_seed
    key = jr.PRNGKey(seed)
    model_key, train_key = jr.split(key, 2)
    # Also fix random seed for other random libraries.
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Initialize model.
    model = init_model(len(tokenizer), config.model, key=model_key)

    # Initialize optimizer and opt_state.
    optimizer, opt_state = init_optimizer(model, config, logger=limited_log)

    # Initialize train state.
    train_state = TrainState(
        model=model,
        opt_state=opt_state,
        dynamic_scaler_state=DynamicScalerState() if config.train.use_amp else None,
        iteration=jnp.array(0),
        train_key=train_key,
        aux_state=init_aux_state(config.logging, model, opt_state)
    )

    # If loading is true, loads train state from checkpoint.
    if config.checkpoint.load:
        checkpoint_file = os.path.join(config.checkpoint.load_path, config.checkpoint.load_file)
        train_state = serializer.load(checkpoint_file, train_state)
        logging.info(f"Successfully loaded checkpoint file from '{checkpoint_file}'.")

    return train_state, optimizer, train_loader, tokenizer, limited_log


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(config: DictConfig) -> None:
    """A systematic testing on stochastic loss local landscape."""
    # manually prevent overwriting test config
    test_config = config.test
    config = init_config(config)
    config.test = test_config
    config.logging.wandb_project = "local_landscape"
    config.logging.wandb_project = None
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

    jit_power_iteration = eqx.filter_jit(
        jtu.Partial(power_iteration_back_prop, batches=batches, config=config, iter=3)
    )
    eigenvalue, eigenvector = jit_power_iteration(train_state)
    print(f"eigen-value: {eigenvalue}, eigen-vector: ", eigenvector)
    

if __name__ == "__main__":
    # test_power_iteration()
    main()