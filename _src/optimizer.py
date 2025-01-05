"""Optimizer algorithms."""

import jax
import jax.tree_util as jtu
import jax.random as jr
import optax
import optimizers
import equinox as eqx
from omegaconf import DictConfig
from typing import Any, Tuple
from jaxtyping import PRNGKeyArray


def init_schedule(lr_config: DictConfig) -> optax.ScalarOrSchedule:
    """Parses the config and initializes a learning rate scheduler.

    Args:
        lr_config: The learning rate config.
        kargs: Additional arguments to overwrite learning rate config.

    Returns:
        A `optax.ScalarOrSchedule` object.
    """
    def use_warmup(warmup: Any) -> bool:
        return isinstance(warmup, int) and (warmup > 0)

    def init_constant_lr(config):
        learning_rate = config.lr
        return learning_rate
    
    def init_cosine_lr(config):
        if use_warmup(config.warmup):
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
        if use_warmup(config.warmup):
            learning_rate = optimizers.warmup_linear_decay_schedule(
                init_value=0.0,
                peak_value=config.lr,
                warmup_steps=config.warmup,
                decay_steps=config.max_steps,
            )
        else:
            learning_rate = optax.linear_schedule(
                init_value=config.lr,
                end_value=0.0,
                transition_steps=config.max_steps,
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
    wandb_log: None,
    schedule_title: str="schedule",
):
    """Returns a wrapped scheduler that logs current learning rate.
    
    The wrapped schedule takes in `learning_rate` as argument and returns a scalar lr.
    """
    def wrapper(schedule, count, wandb_log, title):
        if callable(schedule):
            lr = schedule(count)
        else:
            lr = schedule
        if wandb_log is not None:
            jax.experimental.io_callback(wandb_log, None, {f"lr/{title}": lr}, commit=False)
        return lr
    return jtu.Partial(wrapper, learning_rate, wandb_log=wandb_log, title=schedule_title)


def init_optimizer(
    model: eqx.Module,
    config: DictConfig,
    wandb_log: None,
    *,
    key: PRNGKeyArray,
) -> Tuple[optax.GradientTransformation, optax.OptState]:
    """Construct optimizer from model and training config.

    Args:
        model: an equinox.Module object.
        config: global_config.
        wandb_log: optional logger to handle backend wandb logging while training.
        key: random key for optimizer.

    Returns:
        A tuple of optax.GradientTransofrmation and optax.OptState.
    """
    def init_adamw(config: DictConfig):
        learning_rate = wrap_scheduler(
            init_schedule(config.lr_config), wandb_log=wandb_log)
        return optimizers.adamw(
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

    def init_sgdm(config: DictConfig):
        learning_rate = wrap_scheduler(
            init_schedule(config.lr_config), wandb_log=wandb_log)
        return optimizers.sgdm(
            learning_rate=learning_rate,
            beta=config.beta,
            weight_decay=config.weight_decay
        )

    # Initialize base optimizer.
    name = config.optimizer.name
    opt_config = config.optimizer
    if name == "adamw":
        optimizer = init_adamw(config=opt_config)
    elif name == "sgdm":
        optimizer = init_sgdm(config=opt_config)

    # Wrap online-to-nonconvex.
    if name in ["ogd_md"]:
        wrap_o2nc = True
    elif name in ["adamw", "sgdm", "polar", "jump"]:
        wrap_o2nc = False
    else:
        wrap_o2nc = config.train.wrap_o2nc
    if wrap_o2nc:
        optimizer = optimizers.online_to_gradient_transformation(optimizer)

    # Wrap random scaling.
    random_scaling_key, key = jr.split(key)
    optimizer = optimizers.wrap_random_scaling(
        gradient_transformation=optimizer,
        random_scaling=config.train.random_scaling,
        use_importance_sampling=config.train.use_importance_sampling,
        key=random_scaling_key,
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