"""Train loop for lm tasks."""

import jax
import jax.tree_util as jtu
import jax.random as jr
import jax.numpy as jnp

import equinox as eqx
import optax
from optax import GradientTransformation

from typing import List, Tuple
from jaxtyping import Array
from omegaconf import DictConfig, ListConfig

import os, logging
from tqdm import tqdm

from serialize import serializer
from utils import tree_utils
from utils import TimeKeeper, RateLimitedWandbLog
from dataloaders import DataBatch, DataLoader
from dataloaders import shift_labels
from loggers import Logger, LogMetrics
from loggers import get_internal_logs
from losses import ObjectiveFn
from _src.train.base import TrainState
from _src.train.base import forward_prop, back_prop


# TODO: maybe a better design is to instantiate the Logger class in each train_loop function
# so that each train_loop is coupled with one unique logger. This way, we don't need to worry
# about manually changing codes when a different logger is used.
# - Cons: only one logger is allowed, which violates my initial design.
def train_step(
        train_state: TrainState,
        batches: List[DataBatch],
        optimizer: GradientTransformation,
        loss_fn: ObjectiveFn,
        logger: Logger,
        config: DictConfig,
) -> Tuple[Array, Array, LogMetrics, TrainState]:
    """Wraps one training step, including back-prop, optimizer update, log update, etc.

    Given a train_state corresponding to model x_n and a mini-batch of data z_n
    in iteration n (starting from n=0), performs one train step and updates to model x_(n+1).
    
    Returns:
        A tuple of (loss, accuracy, log metrics, train state).
    """
    use_amp = config.train.use_amp
    amp_precision = config.train.precision
    use_log_callback = config.logging.wandb_project != None and config.logging.log_callback_data
    use_forward_prev = config.logging.compute_last_loss and use_log_callback
    use_back_prev = config.logging.compute_last_grads and use_log_callback

    model = train_state.model                                       # x_n
    opt_state = train_state.opt_state
    log_state = train_state.log_state
    iteration = train_state.iteration

    # Compute f(x_n, z_n) and g(x_n, z_n).
    loss, accuracy, grads, train_state = back_prop(
        *batches, 
        train_state=train_state, 
        loss_fn=loss_fn, 
        use_amp=use_amp, 
        amp_precision=amp_precision,
    )

    # Apply one-step update: x_n -> x_(n+1).
    # NOTE: following the notion of online-to-non-convex, the updates 
    # used for x_(n+1) has iteration index n+1, i.e., Delta_(n+1). 
    # Hence, Delta_(n+1) is dependent on g_t but independent of g_(t+1).
    updates, opt_state = optimizer.update(
        grads, opt_state, eqx.filter(model, eqx.is_array)
    )                                                               # x_(n+1) - x_n = s_(n+1) * Delta_(n+1)
    new_model = eqx.apply_updates(model, updates)                   # x_(n+1)

    # Compute log metrics via logger.
    # NOTE: this part of the code needs manual adaption to logger.update().
    # For now, please be consistent with the logger.
    # TODO: if we want back_prop at x_(n-1), then we also need to store amp_state
    # of the last iteration as well?
    # For full_log, we need to compute an extra forward prop at f(x_(n-1), z_n). 
    if use_forward_prev:
        is_nonarray = lambda x: not eqx.is_array(x)
        train_state = train_state._replace(
            model = eqx.combine(log_state.params_prev, eqx.filter(model, is_nonarray))
        )
        loss_prev, _, _ = forward_prop(
            *batches, 
            train_state=train_state, 
            loss_fn=loss_fn, 
            use_amp=use_amp, 
            amp_precision=amp_precision,
        )

        optim_metrics = get_internal_logs(opt_state)
        random_scaling = optim_metrics["update/random_scaling"] if "update/random_scaling" in optim_metrics else 1.0
        if isinstance(logger, type(None)):
            raise KeyboardInterrupt
        log_state, log_metrics = logger.update(
            log_state, loss=loss, loss_prev=loss_prev, 
            params=eqx.filter(model, eqx.is_array),
            grads=grads, updates=updates,
            random_scaling=random_scaling,
        )
        log_metrics.update(optim_metrics)
    else:
        log_metrics = {}

    # Update new train_state.
    train_state = train_state._replace(
        model = new_model,
        opt_state = opt_state,
        log_state = log_state,
        iteration = optax.safe_int32_increment(iteration),
    )
    return loss, accuracy, log_metrics, train_state


def lm_train_loop(
        config: DictConfig,
        train_state: TrainState,
        optimizer: GradientTransformation,
        dataloader: DataLoader,
        loss_fn: ObjectiveFn,
        logger: Logger,
        time_keeper: TimeKeeper,
        wandb_logger: RateLimitedWandbLog,
        max_nan_loss: int = 5,
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
    pbar = tqdm(enumerate(dataloader_idx), total=num_steps)

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
        loss, accuracy, log_metrics, train_state = train_step_jit(
            train_state, batches, optimizer, loss_fn, logger
        )

        # Auto-terminate if there are too many consecutive nan losses.
        num_nans = train_state.num_nans
        if jnp.isnan(loss):
            if num_nans >= max_nan_loss:
                logging.info(f"iteration {train_state.iteration}: loss = {loss}, training stopped.")
                break
            else:
                train_state = train_state._replace(num_nans=num_nans+1)
        elif num_nans > 0:
            train_state = train_state._replace(num_nans=0)

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
        metrics.update(log_metrics)

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
            wandb_logger(
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
                checkpoint_train_state = os.path.join(config.checkpoint.save_path, f"iter_{it}.ckpt")
                serializer.save(checkpoint_train_state, train_state)
                logging.info(f"Successfully saves checkpoint file to '{checkpoint_train_state}'.")

                checkpoint_model = os.path.join(config.checkpoint.save_path, f"iter_{it}_model.ckpt")
                serializer.save(checkpoint_model, train_state.model)
                logging.info(f"Successfully saves checkpoint model to '{checkpoint_model}'.")

    return train_state