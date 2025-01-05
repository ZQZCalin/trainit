"""Functions related to training."""

import jax
import jax.tree_util as jtu
import jax.random as jr
import jax.numpy as jnp

import equinox as eqx
import optax

from typing import Any, Tuple, Optional, NamedTuple
from jaxtyping import Array, PRNGKeyArray, PyTree

from jaxamp import amp, DynamicScalerState, dynamic_scale_value_and_grad
from omegaconf import DictConfig

import os, sys, logging, warnings
from tqdm import tqdm

import numpy as np
import torch
import random

from serialize import serializer
import utils
from utils import tree_utils
from utils import TimeKeeper, RateLimitedWandbLog
from loggers import LogMetrics
from losses import LossFn
from _src.base import MiniBatch, TrainState


def back_prop(
        loss_fn: LossFn,
        train_state: TrainState,
        batches: MiniBatch,
        config: DictConfig,
        no_grads: bool = False,
) -> Tuple[Array, Array, PyTree, TrainState]:
    """Computes (potentially multi-batch average) loss, grads, accuracy.

    Only modifies 
    
    Returns:
        train_state, loss, accuracy, grads (averaged over batches)
    """
    # Apply auto mixed precision.
    if config.train.use_amp:
        amp_loss_fn = amp(loss_fn, compute_dtype=utils.get_dtype(config.train.precision))
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
        accuracy += utils.get_accuracy(logits_, batch)
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

    return loss, accuracy, grads, train_state


def train_step(
        train_state: TrainState,
        batches: MiniBatch,
        optimizer: optax.GradientTransformation,
        config: DictConfig,
) -> Tuple[Array, Array, LogMetrics, TrainState]:
    """Wraps one training step, including back-prop, optimizer update, log update, etc."""
    model = train_state.model                                       # x_n
    opt_state = train_state.opt_state                               # opt_state of x_n

    # Compute f(x_n, z_n) and g(x_n, z_n).
    loss, accuracy, grads, train_state = back_prop(train_state, batches, config)

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


# TODO: this train loop specifies many details targeting to LMs only.
# so maybe it's a good idea to call it `lm_train_loop` and distinguish it with other possible train loops like `cv_train_loop`
def train_loop(
    train_state: TrainState,
    optimizer: optax.GradientTransformation,
    dataloader: Any,
    config: DictConfig,
    time_keeper: TimeKeeper,
    logger: RateLimitedWandbLog,
) -> TrainState:
    """The main train loop that handles training, logging, and checkpointing."""
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
                # TODO: manually specify dataset.shift_labels=False for other tasks.
                # In the future, we might want a cleaner way to integrate any pre-processing to data batches.
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
            logging.info(f"iteration {train_state.iteration}: loss = inf, training stopped.")
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
        # A checkpoint is saved either when `it % save_steps == 0` or when `it in save_steps`.
        if config.checkpoint.save:
            save_steps = config.checkpoint.save_steps
            it = int(train_state.iteration)     # NOTE: this is 1-indexing by construction
            if isinstance(save_steps, int):
                to_save = it % save_steps == 0
            elif isinstance(save_steps, ListConfig):
                to_save = it in save_steps
            else:
                raise TypeError(f"checkpoint.save_steps has invalid type '{type(save_steps)}'.")
            if to_save:
                checkpoint_file = os.path.join(config.checkpoint.save_path, f"iter_{it}.ckpt")
                serializer.save(checkpoint_file, train_state)
                logging.info(f"Successfully saves checkpoint file to '{checkpoint_file}'.")

    return train_state