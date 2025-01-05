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

import os

import serialize.serializer as serializer
from datasets import shift_labels
from loggers import LogMetrics
from utils import TimeKeeper, RateLimitedWandbLog
from _src import MiniBatch, TrainState
from _src import init_pipeline
from _src import init_config
from _src import init_wandb
from _src import train_loop


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(config: DictConfig) -> None:
    # NOTE: set verbose=0 to print no config; 
    # verbose=1 to print the final config; 
    # verbose=2 to print both initial and final config.
    config = init_config(config, verbose=2)
    
    # NOTE: customize each component in `_src` if needed.
    train_state, optimizer, train_loader, loss_fn, logger, limited_log = init_pipeline(config)

    time_keeper = TimeKeeper()

    init_wandb(config)

    # NOTE: customize your own train_loop in `_src/train.py`.
    train_loop = init_train_loop(config)
    train_loop(
        train_state = train_state,
        optimizer = optimizer,
        dataloader = train_loader,
        config = config,
        logger = limited_log,
        time_keeper = time_keeper
    )


if __name__ == "__main__":
    main()
