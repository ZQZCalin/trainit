"""Initialize the training setup."""

import jax
import jax.tree_util as jtu
import jax.random as jr
import jax.numpy as jnp

import equinox as eqx
from optax import GradientTransformation

from typing import Any, List, Tuple, Optional, NamedTuple, Union, Iterable
from jaxtyping import Array

from jaxamp import amp, DynamicScalerState, dynamic_scale_value_and_grad
from omegaconf import OmegaConf, DictConfig, ListConfig

import os, sys, logging, warnings
import wandb

import numpy as np
import torch
import random

from serialize import serializer

from datasets import DataLoader
from losses import LossFn
from loggers import Logger
from utils import RateLimitedWandbLog
from _src.train.base import TrainState
from _src.model import init_model
from _src.dataset import init_dataloader
from _src.optimizer import init_optimizer
from _src.loss import init_loss_fn
from _src.logger import init_logger


def init_pipeline(
        config: DictConfig
) -> Tuple[
        TrainState, 
        GradientTransformation, 
        DataLoader,
        LossFn,
        Logger,
        RateLimitedWandbLog,
    ]:
    """Initialize all components of the training pipeline.

    If loading checkpoint is true, loads train_state from the checkpoint.

    Args:
        config: global_config; usually pre-processed by `init_config`.
    
    Returns:
        A tuple of (
            train state, optimizer, dataloader, 
            loss function, logger, wandb log function
        ).
    """
    use_wandb = config.logging.wandb_project is not None
    to_load_checkpoint = config.checkpoint.load

    # Initialize random keys.
    seed = config.random_seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    key = jr.PRNGKey(seed)
    model_key, optim_key, train_key = jr.split(key, 3)

    # Initialize wandb logger.
    if use_wandb:
        limited_log = RateLimitedWandbLog(config.logging.wandb_logs_per_sec)
    else:
        limited_log = None

    # Initialize model.
    model = init_model(config, key=model_key)

    # Initialize dataloader.
    train_loader = init_dataloader(config)

    # Initialize optimizer and opt_state.
    optimizer, opt_state = init_optimizer(model, config, wandb_log=limited_log, key=optim_key)

    # Initialize loss function.
    loss_fn = init_loss_fn(config)

    # Initialize logger function.
    # NOTE: right now the logger function still requires some manual efforts.
    # Users need to parse additional arguments to `logger.init()` and `logger.update()`
    # if they customize their own logger function that takes more arguments than default ones.
    # Parsing unnecessary arguments, on the other hand, is safe as they will just be dropped.
    logger = init_logger(config)
    log_state = logger.init(
        params=eqx.filter(model, eqx.is_array),
        # parse extra arguments if needed...
    )

    # Initialize amp state.
    # NOTE: I personally believe using DynamicScalarState() for all cases
    # makes management more convenient. for now, I still leave it as is.
    amp_state = DynamicScalerState() if config.train.use_amp else None

    # Initialize train state.
    train_state = TrainState(
        model = model,
        opt_state = opt_state,
        log_state = log_state,
        dynamic_scaler_state = amp_state,
        epoch = jnp.zeros([], dtype=jnp.int32),
        iteration = jnp.zeros([], dtype=jnp.int32),
        train_key = train_key,
    )

    # If loading is true, loads train state from checkpoint.
    if to_load_checkpoint:
        checkpoint_file = os.path.join(config.checkpoint.load_path, config.checkpoint.load_file)
        checkpoint_config = OmegaConf.load(
            os.path.join(config.checkpoint.load_path, 'config.yaml'))
        
        to_reload_opt_state = config.checkpoint.overwrite_optimizer
        to_reload_amp_state = config.train.use_amp != checkpoint_config.train.use_amp
        # TODO: maybe also add a special case for changing logger function?
        # NOTE: I think opt_state (and amp_state) should be the only state(s) subject to change
        # upon loading checkpoint. All other attributes in train_state must remain the same.
        
        # We need to make sure the current train_state has the same structure as the checkpoint.
        # Otherwise, we cannot load train_state because of unmatched train_state structure.
        if to_reload_opt_state:                             # opt_state
            ckpt_optimizer, ckpt_opt_state = init_optimizer(model, checkpoint_config, logger=None)
            train_state = train_state._replace(
                opt_state=ckpt_opt_state
            )
        if to_reload_amp_state:                             # dynamic_scaler_state
            train_state = train_state._replace(
                dynamic_scaler_state=DynamicScalerState() if checkpoint_config.train.use_amp else None
            )
        
        # Load train_state from checkpoint.
        train_state = serializer.load(checkpoint_file, train_state)

        # Undo previous changes and replace with the current opt_state and dynamic_scaler_state.
        if to_reload_opt_state:                             # initialize opt_state
            train_state = train_state._replace(
                opt_state=opt_state
            )
        if to_reload_amp_state:                             # change dynamic_scaler_state
            train_state = train_state._replace(
                dynamic_scaler_state=amp_state
            )
        logging.info(f"Successfully loaded checkpoint file from '{checkpoint_file}'.")

    return train_state, optimizer, train_loader, loss_fn, logger, limited_log


def init_config(
        config: DictConfig,
        verbose: int = 2,
    ) -> DictConfig:
    """Pre-process config files.

    `init_config` takes the loaded hydra config and processes it to handle special cases, 
    e.g., checkpoint saving and loading, and dataset batch size correction (for LoadIt datasets).

    Users don't need to modify this function unless they need to manually handle
    new special cases in their customized components, such as models, datasets, optimziers, etc.
    
    Args:
        config: global_config.
        verbose: whether to print the configs.
            0: prints nothing;
            1: prints output config.
            2: prints both input and output configs.
    """

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
    
    if verbose not in range(3):
        raise ValueError("invalid argument: verbose cannot be '{verbose}'.")
    
    if verbose == 2:
        logging.info(">>> Loaded config:")
        logging.info(OmegaConf.to_yaml(config))

    config = init_config_dataset(config)
    config = init_config_load_ckpt(config)
    config = init_config_save_ckpt(config)

    if verbose == 1 or verbose == 2:
        logging.info(">>> Pre-processed config:")
        logging.info(OmegaConf.to_yaml(config))
        
    return config


def init_wandb(config: DictConfig) -> None:
    """Initialize wandb logging.
    
    Args:
        config: global_config.
    """
    if config.logging.wandb_project is not None:
        wandb.init(project=config.logging.wandb_project, name=config.logging.wandb_name)
        wandb.config.update(OmegaConf.to_container(config))