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


class TrainState(NamedTuple):
    """The train state class that wraps all necessary states for training.
    
    The train state is defined to be easily serializable for checkpointing.
    
    Right now, I'm thinking we should use a consistent structure of train state
    for all training tasks, including lm, cv, and others. 
    It consists of the follwing components:
    
    Attr:
        model: an equinox model
        opt_state: optax optimizer state
        log_state: extra states for the logger
        dynamic_scaler_state: aka amp_state, extra state for amp
        epoch: current epoch number
        iteration: current iteration number
        train_key: the root seed for all randomness of training
    """
    model: eqx.Module
    opt_state: optax.OptState
    log_state: loggers.LogState
    dynamic_scaler_state: Optional[DynamicScalerState]
    iteration: Array
    train_key: PRNGKeyArray


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
