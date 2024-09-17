# A beta version for train_jax.py,
# includes experimental features
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


def train_step(
    train_state: TrainState,
    batches: List[Tuple[Array, Array]],
    optimizer: optax.GradientTransformation,
    config: DictConfig,
):
    """Modified train step for interpolate-O2NC:
    
    $$x_n = x_{n-1} + \Delta_n, w_n = x_{n-1} + s_n\Delta_n$$

    Evaluate loss at F(xn) and gradient at \nabla F(wn), fix random scaling with s_n=Unif(0,1).
    """
    # Compute interpolate parameter w_n.
    model = train_state.model                                       # x_n
    opt_state = train_state.opt_state                               # opt_state of x_n
    aux_state = train_state.aux_state
    key = train_state.train_key
    
    key, new_key = jr.split(key)
    rs_warmup = config.experimental.rs_warmup
    # if type(rs_warmup) == int and train_state.iteration < rs_warmup:
    #     random_scalar = jnp.ones([])                                # set s_n = 1 during warmup
    # else:
    #     random_scalar = jr.uniform(key, minval=0, maxval=1)         # s_n
    use_rs = config.experimental.use_interpolate_o2nc
    use_rs = use_rs and (type(rs_warmup) == int and train_state.iteration >= rs_warmup)
    random_scalar = jax.lax.cond(
        use_rs,
        lambda _: jr.uniform(key, minval=0, maxval=1),
        lambda _: jnp.ones([]),
        operand = None
    )                                                               # s_n
    params_diff = aux_state.params_diff                             # Delta_n = x_n - x_(n-1)
    interpolate_diff = jtu.tree_map(
        lambda delta: (random_scalar - 1) * delta, params_diff 
    )
    interpolate_model = eqx.apply_updates(model, interpolate_diff)  # w_n = x_n - (1-s_n) * Delta_n
    train_state = train_state._replace(
        train_key = new_key
    )

    # Compute f(x_n, z_n) and g(w_n, z_n). NOTE: we use the same batch for loss and gradient evaluations.
    train_state, loss, accuracy, _ = back_prop(train_state, batches, config, no_grads=True) # f(x_n, z_n)
    train_state = train_state._replace(model=interpolate_model)
    train_state, _, _, grads = back_prop(train_state, batches, config)                      # g(w_n, z_n)

    # Apply one-step update.
    updates, opt_state = optimizer.update(
        grads, opt_state, eqx.filter(model, eqx.is_array)
    )                                                               # Delta_(n+1) = x_(n+1) - x_n
    new_model = eqx.apply_updates(model, updates)                   # x_(n+1)

    # Update new train_state.
    train_state = train_state._replace(
        model = new_model,
        opt_state = opt_state,
        iteration = train_state.iteration + 1
    )

    # Update aux_state and related loggings.
    train_state = update_aux_state(
        train_state, updates, grads, batches, loss, config=config)
    log_data = train_state.aux_state.loggings

    # log actual random_scaling:
    log_data.update({
        "update/interpolate_RS": random_scalar,
    })
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


def init_experimental_config(config: DictConfig) -> DictConfig:
    """Further pre-process config based on experimental.yaml."""
    # Interpolate O2NC
    if config.experimental.use_interpolate_o2nc:
        config.train.random_scaling = None
    return config


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(config: DictConfig) -> None:
    config = init_config(config)
    config = init_experimental_config(config)   # further pre-processing
    train(config)


if __name__ == "__main__":
    main()
