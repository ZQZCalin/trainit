"""Tests learning rate schedules."""

import jax
from jax import numpy as jnp
from jax import random as jr
from jax import tree_util as jtu
import chex
import optax
from optax import Updates, Params, OptState, ScalarOrSchedule, GradientTransformation
from typing import Any, Tuple, NamedTuple, Optional, Union, Callable, Protocol
from utils import tree_utils, log_utils
import optimizers.schedule as schedule


def test_scheduler():
    return

if __name__ == "__main__":
    test_scheduler()