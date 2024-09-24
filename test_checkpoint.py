# Loads training checkpoints and tests experimental features.
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

from train_jax import TrainState, \
    init_tokenizer, init_aux_state, load_lm_data, init_model, init_optimizer, init_config, \
    back_prop, update_aux_state
from train_jax import loss_fn
from utils import get_dtype, get_accuracy


def test_ckpt(
    train_state: TrainState,
    dataloader: Any,
    config: DictConfig,
    logger: RateLimitedWandbLog,
) -> TrainState:
    num_steps = 1000
    sep = 0.01
    use_last_batch = config.checkpoint.use_last_batch

    num_batches = config.dataset.total_batch_size // config.dataset.batch_size   # number of mini-batches per iter
    start_steps = train_state.iteration                 # 0 if not loading from checkpoint
    end_steps = start_steps + num_steps
    dataloader_idx = range(start_steps*num_batches, end_steps*num_batches, num_batches)
    pbar = tqdm.tqdm(enumerate(dataloader_idx), total=num_steps)

    last_batch_idx = start_steps * num_batches      # batch_idx of the checkpoint iteration

    back_prop_jit = eqx.filter_jit(
        jtu.Partial(back_prop, config=config, no_grads=False),
    )

    params_diff = train_state.aux_state.params_diff     # x_n - x_{n-1}

    # Loss diff for reference
    if False:
        batches = []
        idx = last_batch_idx
        for batch in dataloader[idx: idx+num_batches]:
            # Manually shift labels for loadit dataset.
            if config.dataset.shift_labels:
                batch = shift_labels(batch)
            input_ids = jnp.asarray(batch["input_ids"])
            labels = jnp.asarray(batch["labels"])
            batches.append((input_ids, labels))
        _, loss, _, _ = back_prop_jit(train_state, batches)
        train_state = train_state._replace(
            model=eqx.apply_updates(train_state.model, utils.negative_tree(params_diff))
        )
        _, last_loss, _, _ = back_prop_jit(train_state, batches)
        print(loss-last_loss)
        logger({"loss-diff": loss-last_loss}, step=1)
        return

    if not config.checkpoint.use_last_params:
        model = eqx.apply_updates(train_state.model, utils.negative_tree(params_diff))
        train_state = train_state._replace(model=model)
    increment = utils.tree_scalar_multiply(params_diff, sep)

    reward = 0
    for it, batch_idx in pbar:
        if it >= num_steps:
            break
        batches = []
        idx = last_batch_idx if use_last_batch else batch_idx
        for batch in dataloader[idx: idx+num_batches]:
            # Manually shift labels for loadit dataset.
            if config.dataset.shift_labels:
                batch = shift_labels(batch)
            input_ids = jnp.asarray(batch["input_ids"])
            labels = jnp.asarray(batch["labels"])
            batches.append((input_ids, labels))
            
        train_state, loss, accuracy, grads = back_prop_jit(train_state, batches)

        if not config.checkpoint.use_last_params:
            model = eqx.apply_updates(train_state.model, increment)
            train_state = train_state._replace(model=model)

        pbar.set_description(f"batch index {idx}:{idx+num_batches}")

        reward += utils.tree_inner_product(grads, params_diff)
        metrics = {
            "iterations": train_state.iteration,
            "loss": loss,
            "accuracy": accuracy,
            "grads": utils.tree_norm(grads),
            "<g,Delta>": utils.tree_inner_product(grads, params_diff),
            "reward": reward / (it+1),
        }

        if config.logging.wandb_project is not None:
            logger(
                metrics,
                step=it,
            )


def test_ckpt_random_direction(
    train_state: TrainState,
    dataloader: Any,
    config: DictConfig,
    logger: RateLimitedWandbLog,
):
    num_steps = 200
    sep = 0.01
    # use_last_batch = config.checkpoint.use_last_batch

    num_batches = config.dataset.total_batch_size // config.dataset.batch_size   # number of mini-batches per iter
    start_steps = train_state.iteration                 # 0 if not loading from checkpoint
    end_steps = start_steps + num_steps
    dataloader_idx = range(start_steps*num_batches, end_steps*num_batches, num_batches)
    pbar = tqdm.tqdm(enumerate(dataloader_idx), total=num_steps)

    last_batch_idx = start_steps * num_batches      # batch_idx of the checkpoint iteration

    back_prop_jit = eqx.filter_jit(
        jtu.Partial(back_prop, config=config, no_grads=False),
    )

    # Use a random direction Delta' instead of actual Delta.
    # Now samples a random direction with norm \|\Delta_n\| * s_n where s_n \sim Exponential(1).
    key1, key2, new_key = jr.split(train_state.train_key, 3)
    Delta = train_state.aux_state.params_diff     # x_n - x_{n-1}
    delta_norm = utils.tree_norm(Delta)

    keys = jr.split(key1, num=len(jtu.tree_leaves(Delta)))
    keys = jtu.tree_unflatten(jtu.tree_structure(Delta), keys)
    params_diff = jtu.tree_map(lambda t, k: jr.normal(key=k, shape=t.shape), Delta, keys)
    params_diff = utils.tree_scalar_multiply(params_diff, delta_norm / utils.tree_norm(params_diff))

    Delta_perturbed = jtu.tree_map(
        lambda d, r: d + 1e-5*r, Delta, params_diff
    )
    print(f"rand_delta: {utils.tree_norm(params_diff)}, delta: {delta_norm}, delta_perturbed: {utils.tree_norm(Delta_perturbed)}")

    print(jtu.tree_leaves(Delta))

    print(jtu.tree_leaves(Delta_perturbed))

    return

    # Fix a training batch
    print("Preparing testing batch...")
    batches = []
    # idx = last_batch_idx
    idx = last_batch_idx + 1000 # NOTE
    for batch in dataloader[idx: idx+num_batches]:
        # Manually shift labels for loadit dataset.
        if config.dataset.shift_labels:
            batch = shift_labels(batch)
        input_ids = jnp.asarray(batch["input_ids"])
        labels = jnp.asarray(batch["labels"])
        batches.append((input_ids, labels))

    # Loss diff for reference
    model = train_state.model

    # print("computing current loss...")
    # _, loss, _, _ = back_prop_jit(train_state, batches)
    # logging.info(f"current loss = {loss}")

    # print("computing last loss...")
    # # Now train_state uses x_n - Delta_n'
    # train_state = train_state._replace(
    #     model=eqx.apply_updates(train_state.model, utils.negative_tree(params_diff))
    # )
    # _, last_loss, _, _ = back_prop_jit(train_state, batches)
    # logging.info(f"last loss = {last_loss}")
    # logging.info(f"Loss difference = {loss-last_loss}")

    # train_state = train_state._replace(
    #     model=eqx.apply_updates(model, utils.tree_scalar_multiply(params_diff, -sep))
    # )
    # _, last_loss, _, _ = back_prop_jit(train_state, batches)
    # logging.info(f"xn-0.01*Delta = {last_loss}")

    # train_state = train_state._replace(
    #     model=eqx.apply_updates(model, utils.tree_scalar_multiply(params_diff, sep))
    # )
    # _, last_loss, _, _ = back_prop_jit(train_state, batches)
    # logging.info(f"xn+0.01*Delta = {last_loss}")

    # train_state = train_state._replace(
    #     model=eqx.apply_updates(model, utils.tree_scalar_multiply(params_diff, 0))
    # )
    # _, last_loss, _, _ = back_prop_jit(train_state, batches)
    # logging.info(f"xn+0*Delta = {last_loss}")

    for scale in [1e-2, 1e-5, 1e-8, 1e-12, 0]:
        train_state = train_state._replace(
            model=eqx.apply_updates(model, utils.tree_scalar_multiply(params_diff, scale))
        )
        _, last_loss, _, _ = back_prop_jit(train_state, batches)
        logging.info(f"xn + ({scale})*rand_Delta = {last_loss}")

        train_state = train_state._replace(
            model=eqx.apply_updates(model, utils.tree_scalar_multiply(Delta_perturbed, scale))
        )
        _, last_loss, _, _ = back_prop_jit(train_state, batches)
        logging.info(f"xn + ({scale})*Delta_perturbed = {last_loss}")

        train_state = train_state._replace(
            model=eqx.apply_updates(model, utils.tree_scalar_multiply(Delta, scale))
        )
        _, last_loss, _, _ = back_prop_jit(train_state, batches)
        logging.info(f"xn + ({scale})*Delta = {last_loss}")

    return

    # Check loss landscape
    increment = utils.tree_scalar_multiply(params_diff, sep)
    reward = 0
    for it, batch_idx in pbar:
        if it >= num_steps:
            break
        train_state, loss, accuracy, grads = back_prop_jit(train_state, batches)

        model = eqx.apply_updates(train_state.model, increment)
        train_state = train_state._replace(model=model)

        pbar.set_description(f"batch index {idx}:{idx+num_batches}")

        reward += utils.tree_inner_product(grads, params_diff)
        metrics = {
            "iterations": train_state.iteration,
            "loss": loss,
            "accuracy": accuracy,
            "grads": utils.tree_norm(grads),
            "<g,Delta>": utils.tree_inner_product(grads, params_diff),
            "reward": reward / (it+1),
        }

        if config.logging.wandb_project is not None:
            logger(
                metrics,
                step=it,
            )
    

def test_random_update(
    train_state: TrainState,
    dataloader: Any,
    config: DictConfig,
    logger: RateLimitedWandbLog,
):
    num_batches = config.dataset.total_batch_size // config.dataset.batch_size   # number of mini-batches per iter
    start_steps = train_state.iteration                 # 0 if not loading from checkpoint
    last_batch_idx = start_steps * num_batches      # batch_idx of the checkpoint iteration

    back_prop_jit = eqx.filter_jit(
        jtu.Partial(back_prop, config=config, no_grads=False),
    )

    # Use a random direction Delta' instead of actual Delta.
    # Now samples a random direction with norm \|\Delta_n\| * s_n where s_n \sim Exponential(1).
    key, new_key = jr.split(train_state.train_key)
    delta = train_state.aux_state.params_diff     # x_n - x_{n-1}
    delta = utils.tree_normalize(delta)

    keys = jr.split(key, num=len(jtu.tree_leaves(delta)))
    keys = jtu.tree_unflatten(jtu.tree_structure(delta), keys)
    rand = jtu.tree_map(lambda t, k: jr.normal(key=k, shape=t.shape), delta, keys)
    rand = utils.tree_normalize(rand)

    unit = utils.zero_tree(delta)
    leaves, treedef = jtu.tree_flatten(unit)
    leaf = leaves[0]
    leaves[0] = leaf.at[0, 0].set(1.0)
    unit = jtu.tree_unflatten(treedef, leaves)

    ones = jtu.tree_map(jnp.ones_like, delta)
    ones = utils.tree_normalize(ones)

    leaf_diag = jnp.ones_like(leaf)
    leaf_diag /= jnp.sum(leaf_diag * leaf_diag) ** 0.5
    leaves[0] = leaf_diag
    diag = jtu.tree_unflatten(treedef, leaves)

    updates = {
        "delta": delta,
        "rand": rand,
        "unit": unit,
        "ones": ones,
        "diag": diag,
    }

    # san check:
    for k, v in updates.items():
        logging.info(f"{k}: {utils.tree_norm(v)}")
        logging.info(jtu.tree_leaves(v)[0])

    # Fix a training batch
    print("Preparing testing batch...")
    batches = []
    idx = last_batch_idx + 1000 # NOTE
    for batch in dataloader[idx: idx+num_batches]:
        # Manually shift labels for loadit dataset.
        if config.dataset.shift_labels:
            batch = shift_labels(batch)
        input_ids = jnp.asarray(batch["input_ids"])
        labels = jnp.asarray(batch["labels"])
        batches.append((input_ids, labels))

    # Loss diff for reference
    print("Testing different losses...")
    model = train_state.model
    scales = [1e-2, 1e-5, 1e-8, 1e-12, 0]
    scales = [1e-12, 0]
    for scale in scales:
        for name, params_diff in updates.items():
            train_state = train_state._replace(
                model=eqx.apply_updates(model, utils.tree_scalar_multiply(params_diff, scale))
            )
            _, loss, _, _ = back_prop_jit(train_state, batches)
            logging.info(f"xn + ({scale})*{name} = {loss}")


from matplotlib import pyplot as plt

def test_param_distribution(
    train_state: TrainState,
    dataloader: Any,
    config: DictConfig,
    logger: RateLimitedWandbLog,
):
    model = eqx.filter(train_state.model, eqx.is_array)
    leaves, treedef = jtu.tree_flatten(model)

    norms = []
    means = []
    std = []
    shapes = []
    for leaf in leaves:
        norms.append(jnp.sum(leaf*leaf)**.5)
        means.append(jnp.mean(leaf))
        std.append(jnp.std(leaf))
        shapes.append(leaf.shape)

    means = jtu.tree_unflatten(treedef, means)
    print(means)

    print(shapes)

    # print(norms)
    # print(means)
    # print(std)

    # plt.bar(range(len(norms)), norms)
    # plt.savefig("test/norms.png")
    # plt.close()
    # plt.bar(range(len(means)), means)
    # plt.savefig("test/means.png")
    # plt.close()
    # plt.bar(range(len(std)), std)
    # plt.savefig("test/std.png")
    # plt.close()


def test_leaf(
    train_state: TrainState,
    dataloader: Any,
    config: DictConfig,
    logger: RateLimitedWandbLog,
):
    print("Params diff")
    updates = train_state.aux_state.params_diff
    updates_mask = jtu.tree_leaves(updates)[34]
    print(utils.tree_norm(updates_mask))
    print(updates_mask)

    print("model")
    model = eqx.filter(train_state.model, eqx.is_array)
    model_mask = jtu.tree_leaves(model)[34]
    print(utils.tree_norm(model_mask))
    print(model_mask)

    num_batches = config.dataset.total_batch_size // config.dataset.batch_size   # number of mini-batches per iter
    start_steps = train_state.iteration                 # 0 if not loading from checkpoint
    last_batch_idx = start_steps * num_batches      # batch_idx of the checkpoint iteration

    back_prop_jit = eqx.filter_jit(
        jtu.Partial(back_prop, config=config, no_grads=False),
    )

    # Use a random direction Delta' instead of actual Delta.
    # Now samples a random direction with norm \|\Delta_n\| * s_n where s_n \sim Exponential(1).
    delta = train_state.aux_state.params_diff     # x_n - x_{n-1}
    delta = utils.zero_tree(delta)

    # Fix a training batch
    print("Preparing testing batch...")
    batches = []
    idx = last_batch_idx + 1000 # NOTE
    for batch in dataloader[idx: idx+num_batches]:
        # Manually shift labels for loadit dataset.
        if config.dataset.shift_labels:
            batch = shift_labels(batch)
        input_ids = jnp.asarray(batch["input_ids"])
        labels = jnp.asarray(batch["labels"])
        batches.append((input_ids, labels))

    _, _, _, grads = back_prop_jit(train_state, batches)
    grads_mask = jtu.tree_leaves(grads)[34]
    print(utils.tree_norm(grads_mask))
    print(grads_mask)

    return

    # Load leaves
    with open("test/GPT2_note.txt", "r") as file:
        lines = file.readlines()
    names = [line.strip() for line in lines]

    # Check each leaf node
    model = train_state.model
    leaves_idx = list(range(1, 162))

    for idx in leaves_idx:
        i = idx - 1
        logging.info(f"index {idx}, name: '{names[i]}'")

        # Construct all-ones on i-th layer.
        leaves, treedef = jtu.tree_flatten(delta)
        leaf_diag = jnp.ones_like(leaves[i])
        leaf_diag /= jnp.sum(leaf_diag * leaf_diag) ** 0.5
        leaves[i] = leaf_diag
        diag = jtu.tree_unflatten(treedef, leaves)

        # Compute loss.
        train_state = train_state._replace(
            model=eqx.apply_updates(model, utils.tree_scalar_multiply(diag, 1e-12))
        )
        _, loss, _, _ = back_prop_jit(train_state, batches)
        logging.info(f"loss = {loss}\n")


def test_ckpt_rand_delta_fix_mask(
    train_state: TrainState,
    dataloader: Any,
    config: DictConfig,
    logger: RateLimitedWandbLog,
) -> TrainState:
    num_steps = 200
    sep = 0.01

    num_batches = config.dataset.total_batch_size // config.dataset.batch_size   # number of mini-batches per iter
    start_steps = train_state.iteration                 # 0 if not loading from checkpoint
    pbar = tqdm.tqdm(range(num_steps))

    last_batch_idx = start_steps * num_batches      # batch_idx of the checkpoint iteration

    back_prop_jit = eqx.filter_jit(
        jtu.Partial(back_prop, config=config, no_grads=False),
    )

    # Actual update
    delta = train_state.aux_state.params_diff     # x_n - x_{n-1}
    params_diff = delta

    # Gaussian but all-zeros in mask layers
    # key = jr.PRNGKey(42)
    # keys = jr.split(key, num=len(jtu.tree_leaves(delta)))
    # keys = jtu.tree_unflatten(jtu.tree_structure(delta), keys)
    # rand = jtu.tree_map(lambda t, k: jr.normal(key=k, shape=t.shape), delta, keys)
    
    # # Manually set zero updates on mask layers
    # mask_idx = [8 + 13*i for i in range(12)]
    # leaves, treedef = jtu.tree_flatten(rand)
    # for i in mask_idx:
    #     leaves[i] = jnp.zeros_like(leaves[i])

    # params_diff = jtu.tree_unflatten(treedef, leaves)
    # params_diff = utils.tree_normalize(params_diff)

    # all-ones in first layer
    # leaves, treedef = jtu.tree_flatten(utils.zero_tree(delta))
    # leaves[0] = jnp.ones_like(leaves[0])
    # params_diff = jtu.tree_unflatten(treedef, leaves)
    # params_diff = utils.tree_normalize(params_diff)

    increment = utils.tree_scalar_multiply(params_diff, sep)

    # Fix a training batch
    print("Preparing testing batch...")
    batches = []
    idx = last_batch_idx + 1000 # NOTE
    for batch in dataloader[idx: idx+num_batches]:
        # Manually shift labels for loadit dataset.
        if config.dataset.shift_labels:
            batch = shift_labels(batch)
        input_ids = jnp.asarray(batch["input_ids"])
        labels = jnp.asarray(batch["labels"])
        batches.append((input_ids, labels))

    # Loss difference for reference
    _, loss, _, _ = back_prop_jit(train_state, batches)
    train_state = train_state._replace(
        model=eqx.apply_updates(train_state.model, utils.negative_tree(params_diff))
    )
    _, last_loss, _, _ = back_prop_jit(train_state, batches)
    logging.info(f"loss diff = {loss-last_loss}")

    reward = 0
    for it in pbar:
        if it >= num_steps:
            break
        
        # Loss
        train_state, loss, accuracy, grads = back_prop_jit(train_state, batches)

        # Increment parameter
        model = eqx.apply_updates(train_state.model, increment)
        train_state = train_state._replace(model=model)

        pbar.set_description(f"batch index {idx}:{idx+num_batches}")

        reward += utils.tree_inner_product(grads, params_diff)
        metrics = {
            "iterations": train_state.iteration,
            "loss": loss,
            "accuracy": accuracy,
            "grads": utils.tree_norm(grads),
            "<g,Delta>": utils.tree_inner_product(grads, params_diff),
            "reward": reward / (it+1),
        }

        if config.logging.wandb_project is not None:
            logger(
                metrics,
                step=it,
            )


# TODO: merge changes in ckpt loading in train_jax.py
def train(config: DictConfig):
    # Fix random seed.
    seed = config.random_seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Initialize Wandb logging.
    if config.logging.wandb_project is not None:
        limited_log = RateLimitedWandbLog(config.logging.wandb_logs_per_sec)
        wandb.init(project=config.logging.wandb_project)
        wandb.config.update(OmegaConf.to_container(config))
    else:
        limited_log = None

    # Initialize dataloader for gpt2.
    tokenizer = init_tokenizer(config)

    train_loader = load_lm_data(config, tokenizer)

    # Initialize random keys.
    key = jr.PRNGKey(config.random_seed)
    model_key, train_key = jr.split(key, 2)

    # Initialize optimizer and train state.
    model = init_model(len(tokenizer), config.model, key=model_key)
    optimizer, opt_state = init_optimizer(model, config, logger=limited_log)
    train_state = TrainState(
        model=model,
        opt_state=opt_state,
        dynamic_scaler_state=DynamicScalerState() if config.train.use_amp else None,
        iteration=jnp.array(0),
        train_key=train_key,
        aux_state=init_aux_state(config.logging, model, opt_state)
    )

    # [CHECKPOINT]: Load train state from checkpoint.
    if config.checkpoint.load:
        checkpoint_path = os.path.join(config.checkpoint.load_path, config.checkpoint.load_file)
        # load_config should take care of this part.
        # if not os.path.exists(checkpoint_path):
        #     raise ValueError(f"checkpoint path {checkpoint_path} does not exist.")
        train_state = serializer.load(checkpoint_path, train_state)

    time_keeper = TimeKeeper()

    # test_ckpt(
    #     train_state,
    #     train_loader,
    #     config,
    #     limited_log,
    # )
    # test_ckpt_random_direction(
    #     train_state,
    #     train_loader,
    #     config,
    #     limited_log,
    # )
    # test_random_update(
    #     train_state,
    #     train_loader,
    #     config,
    #     limited_log,
    # )
    # test_param_distribution(
    #     train_state,
    #     train_loader,
    #     config,
    #     limited_log,
    # )
    # test_leaf(
    #     train_state,
    #     train_loader,
    #     config,
    #     limited_log,
    # )
    test_ckpt_rand_delta_fix_mask(
        train_state,
        train_loader,
        config,
        limited_log,
    )


def init_config_load_ckpt(config: DictConfig) -> DictConfig:
    # ======================================================================
    # [CHECKPOINT]: check and load checkpoint config files.
    if config.checkpoint.load:
        # Check path existing: load_path, load_file, config file in load_path.
        checkpoint_path = config.checkpoint.load_path
        checkpoint_file = os.path.join(checkpoint_path, config.checkpoint.load_file)
        config_path = os.path.join(checkpoint_path, 'config.yaml')
        if checkpoint_path is None:
            raise ValueError("checkpoint.load_path cannot be empty.")
        if not os.path.exists(checkpoint_path):
            raise ValueError(f"checkpoint path '{checkpoint_path}' does not exist.")
        if not os.path.exists(checkpoint_file):
            raise ValueError(f"checkpoint file '{checkpoint_file}' does not exist.")
        if not os.path.exists(config_path):
            raise ValueError(f"config file '{config_path}' does not exist.")
        # Load checkpoint config.
        checkpoint_config = config.checkpoint
        config = OmegaConf.load(config_path)
        config.checkpoint = checkpoint_config
        print(f"Successfully loaded config file from '{config_path}'.")
        print(f"Successfully loaded checkpoint file from '{checkpoint_file}'.")
    return config


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(config: DictConfig) -> None:
    config = init_config(config)
    # For this test purpose, we always specify config.load=True, so it's fine to just call `init_config_load_ckpt`.
    config = init_config_load_ckpt(config)

    config.logging.wandb_project = "o2nc_ckpt"
    # config.logging.wandb_project = None
    # config.train.use_amp = False                    # manually turn off AMP
    # config.checkpoint.use_last_batch = True         # use z_n if True, else use z_{n+i}
    # config.checkpoint.use_last_params = False       # use x_n if True, else use w_i

    # config.train.use_amp = False

    # logging.info(OmegaConf.to_yaml(config))

    train(config)

if __name__ == "__main__":
    main()
