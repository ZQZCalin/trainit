# The main training file.  
# 
# This file provides an example of integrating 
# all components into a complete training pipeline.
# ===================================================================


import logging
import warnings

import jax
from jax import numpy as jnp
from jax import random as jr
from jax import tree_util as jtu

import optax
import equinox as eqx

from jaxamp import amp, DynamicScalerState, dynamic_scale_value_and_grad
from typing import List, Tuple, Any, Optional, NamedTuple
from jaxtyping import Array, PRNGKeyArray

import tqdm
import wandb
import hydra
from omegaconf import OmegaConf, DictConfig, ListConfig

# import utils
# from utils import softmax_cross_entropy, tree_norm, get_accuracy, get_dtype
from utils import tree_utils, wandb_utils, log_utils
from models.mingpt import GPT
from datasets import shift_labels

import random
import numpy as np
import torch
import os

import serialize.serializer as serializer


MiniBatch = List[Tuple[Array, Array]]



# TODO: move all AuxState-related components to `loggings/`
class AuxState(NamedTuple):
    """Auxiliary states stored for additional loggings."""
    params_diff: Optional[optax.Updates]        # x_n - x_{n-1} = s_n * Delta_n
    last_grads: Optional[optax.Updates]         # grad_{n-1}
    past_grads: Optional[optax.Updates]         # sum_{i=1}^{n-1} grad_i
    random_scalar: Optional[Array]              # s_n
    importance_sampling: Optional[Array]        # w_n = [1-P(s)] / p(s)
    loggings: Optional[dict]


class TrainState(NamedTuple):
    model: eqx.Module
    opt_state: optax.OptState
    dynamic_scaler_state: Optional[DynamicScalerState]
    iteration: Array
    train_key: Array
    aux_state: Optional[AuxState]




# TODO: to be moved to `loss.py` and contained by a wrapper.
def loss_fn(model: eqx.Module, batch: Tuple[Array, Array], key: PRNGKeyArray):
    """Wrapper for cross entropy loss.
    Applies jax.vmap to all data in a data batch.

    Args:
        model: equinox module
        batch: data batch of form (feature, target).
        key: random key used for model forward. 
            This will be neglected if model forward is deterministic (e.g., no dropout).

    Returns:
        Loss value and logits (model outputs).
    """
    def single_example_loss_fn(input, target):
        logits = model(input, key=key)
        loss = softmax_cross_entropy(logits, target)
        return loss, logits

    vmapped_loss_fn = jax.vmap(single_example_loss_fn, in_axes=(0, 0), out_axes=(0, 0))
    input, target = batch
    loss, logits = vmapped_loss_fn(input, target)

    return jnp.mean(loss), logits


def back_prop(
    train_state: TrainState,
    batches: MiniBatch,
    config: DictConfig,
    no_grads: bool = False,
):
    """Computes (potentially multi-batch average) loss, grads, accuracy.
    
    Returns:
        train_state, loss, accuracy, grads (averaged over batches)
    """
    # Apply auto mixed precision.
    if config.train.use_amp:
        amp_loss_fn = amp(loss_fn, compute_dtype=get_dtype(config.train.precision))
        value_and_grad_fn = dynamic_scale_value_and_grad(
            amp_loss_fn, filter=True, has_aux=True, redo_on_nan=0
        )
    else:
        value_and_grad_fn = eqx.filter_value_and_grad(loss_fn, has_aux=True)

    model = train_state.model                                       # x_n
    current_key, new_key = jr.split(train_state.train_key)
    num_batches = len(batches)
    keys = jr.split(current_key, num_batches)

    # Compute f(x_n, z_n) and g(x_n, z_n) for multi-batches.
    batches = jnp.array(batches)
    keys = jnp.array(keys)
    def back_prop_single_batch(i, val):
        loss, accuracy, grads, dynamic_scaler_state = val
        batch, key = batches[i], keys[i]
        if no_grads:
            # Forward prop without gradient.
            if config.train.use_amp:
                loss_, logits_ = amp_loss_fn(model, batch, key=key)
            else:
                loss_, logits_ = loss_fn(model, batch, key=key)
            grads_ = utils.zero_tree(grads)
        else:
            # Back prop with gradient.
            if config.train.use_amp:
                dynamic_scaler_state, ((loss_, logits_), grads_) = value_and_grad_fn(
                    model, batch, key=key, dynamic_scaler_state=dynamic_scaler_state
                )
            else:
                (loss_, logits_), grads_ = value_and_grad_fn(model, batch, key=key)
        loss += loss_
        accuracy += get_accuracy(logits_, batch)
        grads = utils.tree_add(grads, grads_)
        return (loss, accuracy, grads, dynamic_scaler_state)
    
    loss = 0
    accuracy = 0
    grads = jtu.tree_map(jnp.zeros_like, eqx.filter(model, eqx.is_array))
    dynamic_scaler_state = train_state.dynamic_scaler_state

    loss, accuracy, grads, dynamic_scaler_state = jax.lax.fori_loop(
        0, num_batches, back_prop_single_batch,
        (loss, accuracy, grads, dynamic_scaler_state)
    )
    loss /= num_batches
    accuracy /= num_batches
    grads = tree_utils.scalar_dot(grads, 1/num_batches)

    train_state = train_state._replace(
        dynamic_scaler_state=dynamic_scaler_state,
        train_key=new_key,
    )

    return train_state, loss, accuracy, grads


def train_step(
    train_state: TrainState,
    batches: MiniBatch,
    optimizer: optax.GradientTransformation,
    config: DictConfig,
):
    model = train_state.model                                       # x_n
    opt_state = train_state.opt_state                               # opt_state of x_n

    # Compute f(x_n, z_n) and g(x_n, z_n).
    train_state, loss, accuracy, grads = back_prop(train_state, batches, config)

    # Apply one-step update.
    updates, opt_state = optimizer.update(
        grads, opt_state, eqx.filter(model, eqx.is_array)
    )                                                               # s_(n+1) * Delta_(n+1) = x_(n+1) - x_n
    new_model = eqx.apply_updates(model, updates)                   # x_(n+1)

    # Update new train_state.
    train_state = train_state._replace(
        model=new_model,
        opt_state=opt_state,
        iteration=train_state.iteration+1,
    )

    # Update aux_state and related loggings.
    train_state = update_aux_state(
        train_state, updates, grads, batches, loss, config=config)
    log_data = train_state.aux_state.loggings
    return loss, accuracy, log_data, train_state


def save_checkpoint(
    train_state: TrainState,
    config: DictConfig,
):
    """A wrapper of checkpoint saving in the train loop.
    
    Checks saving conditions and saves the checkpoint when the conditions are met.
    A checkpoint is saved either when `it % save_steps == 0` or when `it in save_steps`.
    """
    if config.checkpoint.save:
        save_steps = config.checkpoint.save_steps
        it = int(train_state.iteration)
        if isinstance(save_steps, int):
            save_checkpoint = it % save_steps == 0
        elif isinstance(save_steps, ListConfig):
            save_checkpoint = it in save_steps
        else:
            raise TypeError(f"checkpoint.save_steps has invalid type '{type(save_steps)}'.")
        if save_checkpoint:
            checkpoint_file = os.path.join(config.checkpoint.save_path, f"iter_{it}.ckpt")
            serializer.save(checkpoint_file, train_state)
            logging.info(f"Successfully saves checkpoint file to '{checkpoint_file}'.")


def train_loop(
    train_state: TrainState,
    optimizer: optax.GradientTransformation,
    dataloader: Any,
    config: DictConfig,
    time_keeper: TimeKeeper,
    logger: RateLimitedWandbLog,
) -> TrainState:
    # [CHECKPOINT]: Handling restarting from checkpoints.
    # do_save_checkpoint = config.checkpoint.save
    # checkpoint_path = config.checkpoint.save_path
    # num_steps = config.train.max_steps
    # if do_save_checkpoint:
    #     if checkpoint_path is None:
    #         raise ValueError("checkpoint.save_path cannot be empty.")
    #     # checkpoint_path = os.path.join(os.getcwd(), "saved_checkpoints", checkpoint_path)
    #     if not os.path.exists(checkpoint_path):
    #         raise ValueError(f"checkpoint path {checkpoint_path} does not exist.")
    #     if config.checkpoint.num_steps is not None:
    #         num_steps = config.checkpoint.num_steps
    num_steps = config.train.max_steps
    if config.checkpoint.save and config.checkpoint.num_steps:
        num_steps = config.checkpoint.num_steps

    # TODO: consider adding a batch index in train_state, instead of hardcoding batch index like this
    num_batches = config.dataset.total_batch_size // config.dataset.batch_size   # number of mini-batches per iter
    start_steps = train_state.iteration                 # 0 if not loading from checkpoint
    end_steps = start_steps + num_steps
    dataloader_idx = range(start_steps*num_batches, end_steps*num_batches, num_batches)
    pbar = tqdm.tqdm(enumerate(dataloader_idx), total=num_steps)

    running_loss, running_accuracy, total_tokens = 0, 0, 0
    
    train_step_jit = eqx.filter_jit(
        jtu.Partial(train_step, config=config),
    )
    
    # Initialize Wandb Logger
    beta = 1.0 - 1.0 / config.logging.running_stats_window
    iteration_timing_events = ["iteration", "dataloader", "train_step"]
    time_keeper.mark(start_events=["dataloader", "iteration", "tokens", "samples"])

    for it, batch_idx in pbar:
        if it >= num_steps:
            break
        # Load training batch.
        batches = []
        tokens = 0
        samples = 0
        for batch in dataloader[batch_idx: batch_idx+num_batches]:
            # Manually shift labels for loadit dataset.
            if config.dataset.shift_labels:
                batch = shift_labels(batch)
            input_ids = jnp.asarray(batch["input_ids"])
            labels = jnp.asarray(batch["labels"])
            batches.append((input_ids, labels))
            tokens += jnp.sum(jnp.asarray(batch["attention_mask"]))
            samples += labels.shape[0]

        time_keeper.mark(end_events={"dataloader": 1}, start_events=["train_step"])

        # Apply one-step train_step.
        loss, accuracy, log_data, train_state = train_step_jit(
            train_state, batches, optimizer
        )
        # A dumb san check: end train loop if loss is infinite.
        if jnp.isnan(loss):
            break
        time_keeper.mark(
            end_events={"train_step": 1},
        )

        # Update loss and accuracy.
        running_loss = beta * running_loss + (1.0 - beta) * loss
        total_tokens += tokens
        running_accuracy = beta * running_accuracy + (1 - beta) * accuracy
        pbar.set_description(
            f"train iter: {it}, tokens: {total_tokens}, loss: {loss:.2f}, accuracy: {accuracy:.4f}, running_loss: {running_loss/(1.0-beta**(it+1)):.2f}, running_accuracy: {running_accuracy/(1.0-beta**(it+1)):.4f}"
        )

        # ======================================================================
        # BELOW UPDATES ADDITIONAL LOG MESSAGES...
        # Basic states.
        metrics = {
            "iterations": train_state.iteration,
            "loss": loss,
            "total_tokens": total_tokens,
            "accuracy": accuracy,
        }
        metrics.update(log_data)

        # Time complexity related statistics.
        time_keeper.mark(
            start_events=["dataloader", "iteration", "tokens", "samples"],
            end_events={"iteration": 1, "tokens": tokens, "samples": samples},
        )
        durations = time_keeper.get_durations()
        proportions = time_keeper.get_proportions()
        metrics.update(
            {
                f"time/secs_per/{k}": durations[k]
                for k in iteration_timing_events
                if k in durations
            }
        )
        metrics.update(
            {
                f"time/fraction_spent/{k}": proportions[k]
                for k in iteration_timing_events
                if k in proportions
            }
        )

        if "iteration" in durations:
            throughput = {
                "throughput/iteration_per_sec": 1.0 / durations["iteration"],
                "throughput/samples_per_sec": 1.0 / durations["samples"],
                "throughput/tokens_per_sec": 1.0 / durations["tokens"],
            }
            metrics.update(throughput)

        if config.logging.wandb_project is not None:
            logger(
                metrics,
                step=train_state.iteration,
            )

        # ======================================================================
        # [CHECKPOINT]: saves checkpoint.
        save_checkpoint(train_state, config)

    return train_state


def init_train_state(
    config: DictConfig
) -> Tuple[TrainState, optax.GradientTransformation, Any, Any, RateLimitedWandbLog]:
    """Initializes / loads train state.

    If loading checkpoint train_state, it is assumed that 
    
    Returns:
        A tuple of train state, optimizer, dataloader, tokenizer, wandb logger.
    """
    # Initialize random keys.
    seed = config.random_seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    key = jr.PRNGKey(seed)
    model_key, train_key = jr.split(key, 2)

    # Initialize wandb logger.
    if config.logging.wandb_project is not None:
        limited_log = RateLimitedWandbLog(config.logging.wandb_logs_per_sec)
    else:
        limited_log = None

    # Initialize model tokenizer.
    tokenizer = init_tokenizer(config)

    # Initialize dataloader.
    train_loader = load_lm_data(config, tokenizer)

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
        checkpoint_config = OmegaConf.load(
            os.path.join(config.checkpoint.load_path, 'config.yaml'))
        
        # We need to make sure the current train_state has the same structure as the checkpoint.
        if config.checkpoint.overwrite_optimizer:                           # opt_state
            ckpt_optimizer, ckpt_opt_state = init_optimizer(model, checkpoint_config, logger=None)
            train_state = train_state._replace(
                opt_state=ckpt_opt_state
            )
        if config.train.use_amp != checkpoint_config.train.use_amp:         # dynamic_scaler_state
            train_state = train_state._replace(
                dynamic_scaler_state=DynamicScalerState() if checkpoint_config.train.use_amp else None
            )
        
        # Load train_state from checkpoint.
        train_state = serializer.load(checkpoint_file, train_state)

        # Undo previous changes and replace with the current opt_state and dynamic_scaler_state.
        if config.checkpoint.overwrite_optimizer:                           # initialize opt_state
            train_state = train_state._replace(
                opt_state=opt_state
            )
        if config.train.use_amp and not checkpoint_config.train.use_amp:    # turn on amp
            train_state = train_state._replace(
                dynamic_scaler_state=DynamicScalerState()
            )
        if not config.train.use_amp and checkpoint_config.train.use_amp:    # turn off amp
            train_state = train_state._replace(
                dynamic_scaler_state=DynamicScalerState()
            )
        logging.info(f"Successfully loaded checkpoint file from '{checkpoint_file}'.")

    return train_state, optimizer, train_loader, tokenizer, limited_log


def init_config(config: DictConfig) -> DictConfig:
    """Pre-process config files."""

    def init_config_dataset(config):
        """Pre-process dataset configs."""
        # If using loadit data, turn on shift_labels and fix batch_size=2.
        if config.dataset.name == "pile":
            if config.dataset.use_loadit:
                config.dataset.batch_size = 2
                config.dataset.shift_labels = True
            else:
                config.dataset.shift_labels = False
        # If total_batch_size is not specified, default to batch_size.
        if not config.dataset.total_batch_size:
            config.dataset.total_batch_size = config.dataset.batch_size
        return config

    def init_config_load_ckpt(config):
        """Pre-process checkpoint loading configs.

        Overwrites all config with loaded config, except for config.checkpoint.
        """
        if config.checkpoint.load:
            # Check if path exists: load_path, load_file, config file in load_path.
            checkpoint_path = config.checkpoint.load_path
            checkpoint_file = os.path.join(checkpoint_path, config.checkpoint.load_file)
            config_path = os.path.join(checkpoint_path, 'config.yaml')
            if checkpoint_path is None:
                raise ValueError("checkpoint.load_path cannot be empty.")
            if not os.path.exists(checkpoint_path):
                raise ValueError(f"loading checkpoint path '{checkpoint_path}' does not exist.")
            if not os.path.exists(checkpoint_file):
                raise ValueError(f"loading checkpoint file '{checkpoint_file}' does not exist.")
            if not os.path.exists(config_path):
                raise ValueError(f"loading checkpoint config '{config_path}' does not exist.")
            # Load checkpoint config.
            if not config.checkpoint.overwrite_config:
                checkpoint_config = config.checkpoint
                config = OmegaConf.load(config_path)            # loads config from loaded checkpoint
                config.checkpoint = checkpoint_config           # overwrites config.checkpoint with the current config
                logging.info(f"Successfully loaded checkpoint config from '{config_path}'.")
            else:
                warnings.warn("Please be aware that current config overwrites loaded config.")
        return config

    def init_config_save_ckpt(config):
        """Pre-process checkpoint saving configs.
        
        Will raise an error if config.checkpoint.save_path already exists.
        """
        if config.checkpoint.save:
            # Check if path exists.
            checkpoint_path = config.checkpoint.save_path
            config_path = os.path.join(checkpoint_path, 'config.yaml')
            if checkpoint_path is None:
                raise ValueError("checkpoint.save_path cannot be empty.")
            if os.path.exists(checkpoint_path):
                raise ValueError(f"saving checkpoint path '{checkpoint_path}' already exists.")
            # Pre-process save iterations.
            checkpoint_steps = config.checkpoint.save_steps
            if checkpoint_steps is None:
                raise ValueError("checkpoint.save_steps cannot be empty.")
            invalid_checkpoint_steps_type = False
            if not (isinstance(checkpoint_steps, int)):
                if isinstance(checkpoint_steps, ListConfig):
                    if not all(isinstance(item, int) for item in checkpoint_steps):
                        invalid_checkpoint_steps_type = True
                else:
                    invalid_checkpoint_steps_type = True
            if invalid_checkpoint_steps_type:
                print(checkpoint_steps)
                print(type(checkpoint_steps))
                print(20 in checkpoint_steps)
                raise ValueError("checkpoint.save_steps must be either int or list of int.")
            # Check num_steps.
            num_steps = config.checkpoint.num_steps
            if num_steps and not isinstance(num_steps, int):
                raise ValueError("checkpoint.num_steps must be either null or int.")
            # Create checkpoint file and save checkpoint config.
            os.makedirs(checkpoint_path)
            with open(config_path, "w") as f:
                OmegaConf.save(config, f)
            logging.info(f"Successfully created checkpoint path '{checkpoint_path}'.")
            logging.info(f"Successfully saved checkpoint config to '{config_path}'.")
        return config

    config = init_config_dataset(config)
    config = init_config_load_ckpt(config)
    config = init_config_save_ckpt(config)
    return config


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(config: DictConfig) -> None:
    config = init_config(config)
    logging.info(OmegaConf.to_yaml(config))
    
    train_state, optimizer, train_loader, tokenizer, limited_log = init_train_state(config)

    time_keeper = TimeKeeper()

    if config.logging.wandb_project is not None:
        wandb.init(project=config.logging.wandb_project, name=config.logging.wandb_name)
        wandb.config.update(OmegaConf.to_container(config))

    train_loop(
        train_state,
        optimizer,
        train_loader,
        config,
        logger=limited_log,
        time_keeper=time_keeper
    )


if __name__ == "__main__":
    main()
