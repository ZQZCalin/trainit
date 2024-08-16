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
from omegaconf import OmegaConf, DictConfig

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


class AuxState(NamedTuple):
    """Auxiliary states stored for additional loggings."""
    params_diff: Optional[optax.Updates]        # x_n - x_{n-1} = s_n * Delta_n
    last_grads: Optional[optax.Updates]         # grad_{n-1}
    past_grads: Optional[optax.Updates]         # sum_{i=1}^{n-1} grad_i
    random_scalar: Optional[Array]              # s_n
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
        if config.shuffle_buffer_size > 0:
            loader = chunk_shuffle(loader, chunk_size=config.shuffle_buffer_size, length=max_steps, seed=seed)
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
    # Overwrites default config.
    config = {
        'lr': 0,
        'schedule': 'constant',
        'warmup': 0,
        'max_steps': 0
    }
    config.update(lr_config),
    config.update(kwargs)
    config = OmegaConf.create(config)

    use_warmup = type(config.warmup)==int and (config.warmup > 0)
    if config.schedule == "constant":
        learning_rate = config.lr
    elif config.schedule == "cosine":
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
    elif config.schedule == "linear":
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
    else:
        raise ValueError(f"schedule type {config.schedule} is not supported.")
    return learning_rate


# TODO: deprecate logging wrapper. instead, we can log learning rate inside corresponding optimizer that calls it.
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
    return jtu.Partial(
        wrapper, learning_rate, logger=logger, title=schedule_title)


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
    if "update/random_scaling" not in opt_loggings.keys():
        warnings.warn("Optimizer has no key named 'update/random_scaling,",
                      "so random scaling is not recognized in logging.",
                      "Wrap your optimizer with o2nc.wrap_random_scaling for correct logging.")
        random_scalar = jnp.ones([])
    else:
        random_scalar = opt_loggings["update/random_scaling"]
    loggings = {
        "grads/norm": jnp.zeros([]),
        "grads/l1-norm": jnp.zeros([]),
        "update/<gn, Delta(n)>": jnp.zeros([]),
        "update/<gn, Delta(n)>_sum": jnp.zeros([]),
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
        loggings = loggings,
    )


def update_aux_state(
    train_state: TrainState,
    updates: optax.Updates,
    grads: optax.Updates,
    batches: List[Tuple[Array, Array]],
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
            loggings.update({
                "update/<gn, Delta(n)>": inner_g_Delta,
                "update/<gn, Delta(n)>_sum": loggings["update/<gn, Delta(n)>_sum"]+inner_g_Delta,
                "update/<gn, xn-x(n-1)>": inner_g_dx,
                "update/<gn, xn-x(n-1)>_sum": loggings["update/<gn, xn-x(n-1)>_sum"]+inner_g_dx,
            })
        if config.store_last_params and config.compute_last_loss:
            last_model = eqx.apply_updates(
                model, utils.negative_tree(state.params_diff))
            last_loss = 0
            for batch in batches:
                if global_config.train.use_amp:
                    amp_loss_fn = amp(loss_fn, compute_dtype=get_dtype(global_config.train.precision))
                    last_loss_, _ = amp_loss_fn(last_model, batch, key=key)
                else:
                    last_loss_, _ = loss_fn(last_model, batch, key=key)
                last_loss += last_loss_
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
        if "update/random_scaling" in opt_loggings.keys():
            random_scalar = opt_loggings["update/random_scaling"]
        else:
            random_scalar = state.random_scalar
        return state._replace(
            params_diff = updates if config.store_last_params else None,
            last_grads = grads if config.store_last_grads else None,
            past_grads = utils.tree_add(state.past_grads, grads) if config.store_past_grads else None,
            random_scalar = random_scalar,
            loggings = loggings,
        ), dynamic_scaler_state
    
    aux_state, dynamic_scaler_state = jax.lax.cond(
        utils.is_finite_tree(grads), update_finite, update_nan, aux_state, base_loggings, dynamic_scaler_state)
    
    return TrainState(
        model = train_state.model,
        opt_state = opt_state,
        dynamic_scaler_state = dynamic_scaler_state,
        iteration = train_state.iteration,
        train_key = new_key,
        aux_state = aux_state
    )


def back_prop(
    train_state: TrainState,
    batches: List[Tuple[Array, Array]],
    config: DictConfig,
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
    dynamic_scaler_state = train_state.dynamic_scaler_state
    current_key, new_key = jr.split(train_state.train_key)
    num_batches = len(batches)
    keys = jr.split(current_key, num_batches)

    # Compute f(x_n, z_n) and g(x_n, z_n) for multi-batches.
    loss = 0
    accuracy = 0
    grads = jtu.tree_map(jnp.zeros_like, eqx.filter(model, eqx.is_array))
    for batch, key in zip(batches, keys):
        if config.train.use_amp:
            dynamic_scaler_state, ((loss_, logits_), grads_) = value_and_grad_fn(
                model, batch, key=key, dynamic_scaler_state=dynamic_scaler_state
            )
        else:
            (loss_, logits_), grads_ = value_and_grad_fn(model, batch, key=key)
        loss += loss_
        accuracy += get_accuracy(logits_, batch)
        grads = utils.tree_add(grads, grads_)
    loss /= num_batches
    accuracy /= num_batches
    grads = utils.tree_scalar_multiply(grads, 1/num_batches)

    train_state = TrainState(
        model=model,
        opt_state=train_state.opt_state,
        dynamic_scaler_state=dynamic_scaler_state,
        iteration=train_state.iteration,
        train_key=new_key,
        aux_state=train_state.aux_state,
    )

    return train_state, loss, accuracy, grads


def train_step(
    train_state: TrainState,
    batches: List[Tuple[Array, Array]],
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
    train_state = TrainState(
        model=new_model,
        opt_state=opt_state,
        dynamic_scaler_state=train_state.dynamic_scaler_state,
        iteration=train_state.iteration + 1,
        train_key=train_state.train_key,
        aux_state=train_state.aux_state,
    )

    # Update aux_state and related loggings.
    train_state = update_aux_state(
        train_state, updates, grads, batches, loss, config=config)
    log_data = train_state.aux_state.loggings
    return loss, accuracy, log_data, train_state


def train_loop(
    train_state: TrainState,
    optimizer: optax.GradientTransformation,
    dataloader: Any,
    config: DictConfig,
    time_keeper: TimeKeeper,
    logger: RateLimitedWandbLog,
) -> TrainState:
    # [CHECKPOINT]: Handling restarting from checkpoints.
    do_save_checkpoint = config.checkpoint.save
    checkpoint_path = config.checkpoint.save_path
    num_steps = config.train.max_steps
    if do_save_checkpoint:
        if checkpoint_path is None:
            raise ValueError("checkpoint.save_path cannot be empty.")
        # checkpoint_path = os.path.join(os.getcwd(), "saved_checkpoints", checkpoint_path)
        if not os.path.exists(checkpoint_path):
            raise ValueError(f"checkpoint path {checkpoint_path} does not exist.")
        if config.checkpoint.num_steps is not None:
            num_steps = config.checkpoint.num_steps

    num_batches = config.dataset.total_batch_size // config.dataset.batch_size   # number of mini-batches per iter
    start_steps = train_state.iteration                 # 0 if not loading from checkpoint
    end_steps = start_steps + num_steps
    # dataloader = dataloader[start_steps:end_steps]      # get the subset for this checkpoint
    # pbar = tqdm.tqdm(enumerate(dataloader), total=num_steps)  #NOTE
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

    # for it, batch in pbar:    #NOTE
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
        # [CHECKPOINT]: Saving checkpoint.
        if do_save_checkpoint and train_state.iteration % config.checkpoint.save_steps == 0:
            serializer.save(os.path.join(checkpoint_path, f"iter_{train_state.iteration}.ckpt"), train_state)

    return train_state


def train(config: DictConfig):
    # Some san check of config.
    
    # TODO: this is temporary
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

    # Initialize random keys
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
        checkpoint_path = config.checkpoint.load_path
        if not os.path.exists(checkpoint_path):
            raise ValueError(f"checkpoint path {checkpoint_path} does not exist.")
        train_state = serializer.load(checkpoint_path, train_state)

    time_keeper = TimeKeeper()

    train_loop(
        train_state,
        optimizer,
        train_loader,
        config,
        logger=limited_log,
        time_keeper=time_keeper
    )


def init_config(config: DictConfig) -> DictConfig:
    """Pre-process config"""
    # ======================================================================
    # Pre-process pile dataset.
    # If using loadit data, turn on shift_labels and fix batch_size=2.
    if config.dataset.name == "pile":
        if config.dataset.use_loadit:
            config.dataset.batch_size = 2
            config.dataset.shift_labels = True
        else:
            config.dataset.shift_labels = False

    # Pre-process total_batch_size.
    if not config.dataset.total_batch_size:
        config.dataset.total_batch_size = config.dataset.batch_size

    # ======================================================================
    # [CHECKPOINT]: Pre-process config for saving checkpoint.
    # Upon loading config, if checkpoint.save is not None, will process the config in the following way:
    # - A new checkpoint directory will be created if checkpoint.path doesn't exist.
    # - If config.yaml already exists, it will be loaded and will overwrite the config template beside the checkpoint section.
    # - If config.yaml already exists and you want to overwrite it, you can turn on checkpoint.overwrite=True. 
    #   Use this with caution since this overwrites the existing config file.
    # - The updated config will be saved to config.yaml.

    # Save new config file if checkpoint.save is true (i.e., we need to save checkpoint).
    if config.checkpoint.save:
        checkpoint_path = config.checkpoint.save_path
        if checkpoint_path is None:
            raise ValueError("checkpoint.save_path cannot be empty.")
        config_path = os.path.join(checkpoint_path, 'config.yaml')
        if not os.path.exists(checkpoint_path):
            # Create checkpoint directory.
            os.makedirs(checkpoint_path)
            print(f"Directory {checkpoint_path} created.")
        elif os.path.exists(config_path):
            if config.checkpoint.overwrite:
                warnings.warn("checkpoint.overwrite is true. this will overwrite existing config.yaml.")
            else:
                # Load existing config file.
                checkpoint_config = config.checkpoint
                config = OmegaConf.load(config_path)
                config.checkpoint = checkpoint_config
        config.checkpoint.overwrite = False     # manually turn off checkpoint.overwrite
        # Update config file.
        with open(config_path, 'w') as f:
            OmegaConf.save(config, f)
        print(f"Config file {config_path} updated.")

    logging.info(OmegaConf.to_yaml(config))
    return config


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(config: DictConfig) -> None:
    config = init_config(config)
    train(config)


if __name__ == "__main__":
    main()
