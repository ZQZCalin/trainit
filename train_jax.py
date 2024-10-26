# Train gpt2 model on c4 dataset.
# 
# We will fix our model and dataset and test the 
# performance of different optimizers on this task.
# ===========================================================================


import logging
import warnings

import jax
from jax import numpy as jnp
from jax import random as jr
from jax import tree_util as jtu

import optax
import equinox as eqx

import transformers

from jaxamp import amp, DynamicScalerState, dynamic_scale_value_and_grad
from typing import List, Tuple, Any, Optional, Sequence, Union, NamedTuple, Callable
from jaxtyping import Array, PRNGKeyArray

import tqdm
import wandb
import hydra
from omegaconf import OmegaConf, DictConfig, ListConfig

import utils
from utils import softmax_cross_entropy, tree_norm, get_accuracy, get_dtype
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

sys.path.append('./minGPT')
from mingpt.model import GPT as torch_GPT

import random
import numpy as np
import torch

import serialize.serializer as serializer


MiniBatch = List[Tuple[Array, Array]]


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


def load_lm_data(config: DictConfig, tokenizer: Any, split: str = "train"):
    """Wrapper for Pile dataset. 
    config: global config.

    Returns:
        torch.utils.data.DataLoader.
    """
    context_length = config.model.context_length
    max_steps = config.train.max_steps
    seed = config.random_seed
    config = config.dataset
    if config.use_loadit:
        loadit_path = config.loadit_path
        if loadit_path is None:
            loadit_path = "/projectnb/aclab/tranhp/trainDataloader_pile/"
        loader = LoadIt(loadit_path)
        # TODO: either specify max_samples in config, or set length to full dataset length.
        # current implementation is vulnerable to batch size change 
        if config.shuffle_buffer_size > 0:
            length = max_steps * config.total_batch_size
            loader = chunk_shuffle(loader, chunk_size=config.shuffle_buffer_size, length=length, seed=seed)
    else:
        if config.name not in ["c4", "pile"]:
            raise ValueError("dataset name must be c4 or pile.")
        loader = get_lm_loader_next_token(
            tokenizer,
            split=split,
            batch_size=config.batch_size,
            max_length=context_length,
            shuffle_buffer_size=config.shuffle_buffer_size,
            pad_to_multiple_of=context_length,
            num_workers=config.dataloader_workers,
            dataset=config.name,
        )
    return loader


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


def init_tokenizer(config: DictConfig, pad_token: bool = True):
    """Initializes tokenizer. If `pad_token` is true, adds pad_token as a special token. Defaults to true."""
    model_name = config.model.name
    if model_name == "gpt":
        tokenizer = transformers.GPT2TokenizerFast.from_pretrained("gpt2")
        if pad_token:
            tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
    else:
        raise ValueError(f"model {model_name} is not supported.")
    return tokenizer


def init_model(vocab_size: int, config: DictConfig, *, key: PRNGKeyArray) -> eqx.Module:
    """Initializes model. config: global_config.model"""
    if not config.load_pytorch:
        model = GPT(vocab_size, config, key=key)
    else:
        model_config = torch_GPT.get_default_config()
        model_config.model_type = 'gpt2'
        model_config.vocab_size = vocab_size                    # openai's model vocabulary
        model_config.block_size = config.context_length         # openai's model block_size (i.e. input context length)
        model_config.embd_pdrop = config.transformer_dropout
        model_config.resid_pdrop = config.attn_linear_dropout
        model_config.attn_pdrop = config.attn_dropout
        torch_model = torch_GPT(model_config)
        model = GPT(vocab_size, config, state_dict=torch_model.state_dict())
    return model


def init_scheduler(lr_config: DictConfig, **kwargs) -> optax.ScalarOrSchedule:
    """Parses the config and initializes a learning rate scheduler.

    Args:
        lr_config: The learning rate config.
        kargs: Additional arguments to overwrite learning rate config.

    Returns:
        A `optax.ScalarOrSchedule` object.
    """
    def init_constant_lr(config):
        learning_rate = config.lr
        return learning_rate
    
    def init_cosine_lr(config):
        use_warmup = isinstance(config.warmup, int) and (config.warmup > 0)
        if use_warmup:
            learning_rate = optax.warmup_cosine_decay_schedule(
                init_value=0.0,
                peak_value=config.lr,
                warmup_steps=config.warmup,
                decay_steps=config.max_steps,
            )
        else:
            learning_rate = optax.cosine_decay_schedule(
                init_value=config.lr,
                decay_steps=config.max_steps,
            )
        return learning_rate
    
    def init_linear_lr(config):
        use_warmup = isinstance(config.warmup, int) and (config.warmup > 0)
        if use_warmup:
            learning_rate = scheduler.warmup_linear_decay_schedule(
                init_value=0.0,
                peak_value=config.lr,
                warmup_steps=config.warmup,
                decay_steps=config.max_steps,
            )
        else:
            learning_rate = scheduler.linear_decay_schedule(
                init_value=config.lr,
                decay_steps=config.max_steps,
            )
        return learning_rate

    def init_piecewise_linear_lr(config):
        learning_rate = optax.linear_schedule(
            init_value=config.lr1,
            end_value=config.lr2,
            transition_steps=config.max_steps,
            transition_begin=config.start_steps,    # NOTE: for now, we still need to specify the start iteration in config.
        )
        return learning_rate

    if lr_config.schedule == "constant":
        learning_rate = init_constant_lr(lr_config)
    elif lr_config.schedule == "cosine":
        learning_rate = init_cosine_lr(lr_config)
    elif lr_config.schedule == "linear":
        learning_rate = init_linear_lr(lr_config)
    elif lr_config.schedule == "piecewise_linear":
        learning_rate = init_piecewise_linear_lr(lr_config)
    else:
        raise ValueError(f"schedule type {lr_config.schedule} is not supported.")
    return learning_rate


def wrap_scheduler(
    learning_rate: optax.ScalarOrSchedule,
    logger: None,
    schedule_title: str="schedule",
):
    """Returns a wrapped scheduler that logs current learning rate."""
    def wrapper(schedule, count, logger, title):
        if callable(schedule):
            lr = schedule(count)
        else:
            lr = schedule
        if logger is not None:
            jax.experimental.io_callback(logger, None, {f"lr/{title}": lr}, commit=False)
        return lr
    return jtu.Partial(wrapper, learning_rate, logger=logger, title=schedule_title)


def init_optimizer(
    model: eqx.Module,
    config: DictConfig,
    logger: None,
):
    """Construct optimizer from model and training config.

    Returns:
        Initial optimizer and opt_state.
    """
    # Initialize base optimizer / online learner.
    name = config.optimizer.name
    max_steps = config.train.max_steps

    def init_adamw(config: DictConfig, **kwargs):
        """use kwargs to pass down optional arguments (e.g., schedule_title)"""
        learning_rate = wrap_scheduler(
            init_scheduler(config.lr_config), logger=logger, **kwargs)
        return benchmark.adamw(
            learning_rate=learning_rate,
            beta1=config.beta1,
            beta2=config.beta2,
            weight_decay=config.weight_decay,
            debias_beta1=config.debias_beta1,
            debias_beta2=config.debias_beta2,
            use_momentum=config.use_momentum,
            use_preconditioning=config.use_preconditioning,
            decouple_weight_decay=config.decouple_weight_decay,
        )

    def init_sgdm(config: DictConfig, **kwargs):
        learning_rate = wrap_scheduler(
            init_scheduler(config.lr_config), logger=logger, **kwargs)
        return benchmark.sgdm(
            learning_rate=learning_rate,
            beta=config.beta,
            weight_decay=config.weight_decay
        )

    opt_config = config.optimizer
    if name == "adamw":
        optimizer = init_adamw(config=opt_config)
    elif name == "sgdm":
        optimizer = init_sgdm(config=opt_config)
    elif name == "polar":
        optimizer = optim.polar_descent(
            direction_optim=init_adamw(config=opt_config.direction),
            magnitude_optim=init_adamw(config=opt_config.magnitude, schedule_title="schedule_2"),
        )
    elif name == "jump":
        optimizer = optim.jump_trajectory(
            normal_optim=init_adamw(config=opt_config.normal),
            jump_optim=init_adamw(config=opt_config.jump, schedule_title="schedule_2"),
            normal_steps=opt_config.normal_steps,
            jump_steps=opt_config.jump_steps,
        )
    elif name == "ogd_md":
        learning_rate = wrap_scheduler(
            init_scheduler(opt_config.lr_config, max_steps=max_steps), logger=logger)
        optimizer = ol.ogd_mirror_descent(
            learning_rate=learning_rate,
            beta=opt_config.beta,
            mu=opt_config.mu,
        )

    # Wrap online to non-convex.
    if name in ["ogd_md"]:
        wrap_o2nc = True
    elif name in ["adamw", "sgdm", "polar", "jump"]:
        wrap_o2nc = False
    else:
        wrap_o2nc = config.train.wrap_o2nc
    if wrap_o2nc:
        optimizer = deterministic_online_nonconvex(optimizer)

    # Wrap random scaling.
    optimizer = wrap_random_scaling(
        gradient_transformation=optimizer,
        random_scaling=config.train.random_scaling,
        use_importance_sampling=config.train.use_importance_sampling,
        seed=config.train.random_scaling_seed   # TODO: deprecate. use PRNGKey passed from argument instead of random seed.
    )
    
    # Gradient clipping and finite gradient wrapper.
    grad_clip = optax.clip_by_global_norm(config.train.gradient_clip_val)
    optimizer = optax.chain(
        grad_clip,
        optax.apply_if_finite(optimizer, 15)
    )

    # Initialize opt_state
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    return optimizer, opt_state


def init_aux_state(config: DictConfig, model: eqx.Module, opt_state: optax.OptState) -> AuxState:
    """Initializes aux_state from confg."""
    if not config.log_callback_data:
        return None
    opt_loggings = utils.merge_dicts(*logstate.list_of_logs(opt_state))
    if "update/random_scaling" not in opt_loggings:
        warnings.warn("Optimizer has no key named 'update/random_scaling,",
                      "and random scaling is default to one.",
                      "Wrap your optimizer with o2nc.wrap_random_scaling for correct logging.")
        random_scalar = jnp.ones([])
    else:
        random_scalar = opt_loggings["update/random_scaling"]
    if "update/importance_sampling" not in opt_loggings:
        warnings.warn("Optimizer has no key named 'update/importance_sampling,",
                      "and importance sampling is default to one.",
                      "Wrap your optimizer with o2nc.wrap_random_scaling for correct logging.")
        importance_sampling = jnp.ones([])
    else:
        importance_sampling = opt_loggings["update/importance_sampling"]
    loggings = {
        "grads/norm": jnp.zeros([]),
        "grads/l1-norm": jnp.zeros([]),
        "update/<gn, Delta(n)>": jnp.zeros([]),
        "update/<gn, Delta(n)>_sum": jnp.zeros([]),
        "update/<g(n-1), Delta(n)>": jnp.zeros([]),
        "update/<g(n-1), Delta(n)>_sum": jnp.zeros([]),
        "update/cos(g(n-1), Delta(n))": jnp.zeros([]),
        "update/wn*<gn, Delta(n)>": jnp.zeros([]),
        "update/wn*<gn, Delta(n)>_sum": jnp.zeros([]),
        "update/fn-f(n-1)": jnp.zeros([]),
        "update/fn-f(n-1)_sum": jnp.zeros([]),
        "update/<gn, xn-x(n-1)>": jnp.zeros([]),
        "update/<gn, xn-x(n-1)>_sum": jnp.zeros([]),
        "grads/<gn, g(n-1)>": jnp.zeros([]),
        "grads/<gn, g(1:n-1)>": jnp.zeros([]),
        "grads/cos(gn, g(n-1))": jnp.zeros([]),
        "grads/cos(gn, g(1:n-1))": jnp.zeros([]),
        "grads/inf_grads": jnp.zeros([], jnp.int32),
    }
    loggings.update(opt_loggings)
    zeros = utils.zero_tree(eqx.filter(model, eqx.is_array))
    return AuxState(
        params_diff = zeros if config.store_last_params else None,
        last_grads = zeros if config.store_last_grads else None,
        past_grads = zeros if config.store_past_grads else None,
        random_scalar = random_scalar,
        importance_sampling = importance_sampling,
        loggings = loggings,
    )


def update_aux_state(
    train_state: TrainState,
    updates: optax.Updates,
    grads: optax.Updates,
    batches: MiniBatch,
    loss: Array,
    config: DictConfig,
) -> TrainState:
    """Updates aux_state. config: global config.
    Note: train_state.model uses new_model, i.e., x_(n+1).
    """
    global_config = config
    config = config.logging
    if not config.log_callback_data:
        return None
    
    model = eqx.apply_updates(
        train_state.model, utils.negative_tree(updates))    # x_n
    opt_state = train_state.opt_state
    aux_state = train_state.aux_state
    dynamic_scaler_state = train_state.dynamic_scaler_state
    key, new_key = jr.split(train_state.train_key)
    batches = jnp.array(batches)
    
    base_loggings = {
        "grads/norm": utils.tree_l2_norm(grads),
        "grads/l1-norm": utils.tree_l1_norm(grads),
    }
    opt_loggings = utils.merge_dicts(*logstate.list_of_logs(opt_state))
    base_loggings.update(opt_loggings)

    def update_nan(state, base_loggings, dynamic_scaler_state):
        loggings = state.loggings
        loggings.update({
            "grads/inf_grads": optax.safe_int32_increment(loggings["grads/inf_grads"])
        })
        loggings.update(base_loggings)
        return state._replace(loggings=loggings), dynamic_scaler_state
    
    def update_finite(state, base_loggings, dynamic_scaler_state):
        loggings = state.loggings
        if config.store_last_params:
            inner_g_dx = utils.tree_inner_product(grads, state.params_diff)
            inner_g_Delta = inner_g_dx / state.random_scalar
            inner_g_wDelta = inner_g_Delta * state.importance_sampling
            loggings.update({
                "update/<gn, xn-x(n-1)>": inner_g_dx,
                "update/<gn, xn-x(n-1)>_sum": loggings["update/<gn, xn-x(n-1)>_sum"]+inner_g_dx,
                "update/<gn, Delta(n)>": inner_g_Delta,
                "update/<gn, Delta(n)>_sum": loggings["update/<gn, Delta(n)>_sum"]+inner_g_Delta,
                "update/wn*<gn, Delta(n)>": inner_g_wDelta,
                "update/wn*<gn, Delta(n)>_sum": loggings["update/wn*<gn, Delta(n)>_sum"]+inner_g_wDelta,
            })
        if config.store_last_params and config.store_last_grads:
            inner_g_last_Delta = utils.tree_inner_product(state.last_grads, state.params_diff)
            loggings.update({
                "update/<g(n-1), Delta(n)>": inner_g_last_Delta,
                "update/<g(n-1), Delta(n)>_sum": loggings["update/<g(n-1), Delta(n)>_sum"]+inner_g_last_Delta,
                "update/cos(g(n-1), Delta(n))": utils.tree_cosine_similarity(state.last_grads, state.params_diff),
            })
        if config.store_last_params and config.compute_last_loss:
            last_model = eqx.apply_updates(
                model, utils.negative_tree(state.params_diff))
            def compute_last_loss(i, val):
                batch = batches[i]
                if global_config.train.use_amp:
                    amp_loss_fn = amp(loss_fn, compute_dtype=get_dtype(global_config.train.precision))
                    last_loss_, _ = amp_loss_fn(last_model, batch, key=key)
                else:
                    last_loss_, _ = loss_fn(last_model, batch, key=key)
                return val + last_loss_
            last_loss = jax.lax.fori_loop(
                0, len(batches), compute_last_loss, init_val=0
            )
            last_loss /= len(batches)   # average last loss over all batches
            df = loss - last_loss
            loggings.update({
                "update/fn-f(n-1)": df,
                "update/fn-f(n-1)_sum": loggings["update/fn-f(n-1)_sum"]+df,
            })
        if config.store_last_grads:
            loggings.update({
                "grads/<gn, g(n-1)>": utils.tree_inner_product(grads, state.last_grads),
                "grads/cos(gn, g(n-1))": utils.tree_cosine_similarity(grads, state.last_grads),
            })
        if config.store_past_grads:
            loggings.update({
                "grads/<gn, g(1:n-1)>": utils.tree_inner_product(grads, state.past_grads),
                "grads/cos(gn, g(1:n-1))": utils.tree_cosine_similarity(grads, state.past_grads),
            })
        loggings.update(base_loggings)
        if "update/random_scaling" in opt_loggings:
            random_scalar = opt_loggings["update/random_scaling"]
        else:
            random_scalar = state.random_scalar
        if "update/importance_sampling" in opt_loggings:
            importance_sampling = opt_loggings["update/importance_sampling"]
        else:
            importance_sampling = state.importance_sampling
        return state._replace(
            params_diff = updates if config.store_last_params else None,
            last_grads = grads if config.store_last_grads else None,
            past_grads = utils.tree_add(state.past_grads, grads) if config.store_past_grads else None,
            random_scalar = random_scalar,
            importance_sampling = importance_sampling,
            loggings = loggings,
        ), dynamic_scaler_state
    
    aux_state, dynamic_scaler_state = jax.lax.cond(
        utils.is_finite_tree(grads), update_finite, update_nan, aux_state, base_loggings, dynamic_scaler_state)
    
    return train_state._replace(
        dynamic_scaler_state = dynamic_scaler_state,
        train_key = new_key,
        aux_state = aux_state
    )


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
    grads = utils.tree_scalar_multiply(grads, 1/num_batches)

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
