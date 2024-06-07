"""Learning rate schedulers."""

import jax
from jax import numpy as jnp
from jax import random as jr
from jax import tree_util as jtu
import chex
import optax
from optax import Updates, Params, OptState, ScalarOrSchedule, GradientTransformation
from typing import Any, Tuple, NamedTuple, Optional, Union, Callable, Protocol
import sys
sys.path.append('../trainit')
import utils
import logstate


def get_current_lr(
    learning_rate: ScalarOrSchedule,
    count: chex.Array,
):
    """Returns the current learning rate."""    
    if callable(learning_rate):
        return learning_rate(count)
    else:
        return learning_rate
    

def linear_decay_schedule(
    init_value: float,
    decay_steps: int,
    end_value: float = 0.0,
) -> optax.Schedule:
    """Linear decay schedule.

    Args:
        init_value: Initial value for the scalar to be annealed.
        peak_value: Peak value for scalar to be annealed at end of warmup.
        end_value: End value of the scalar to be annealed. Defaults to 0.

    Returns:
        A schedule function.
    """
    return optax.linear_schedule(
        init_value=init_value,
        end_value=end_value,
        transition_steps=decay_steps,
    )


def warmup_linear_decay_schedule(
    init_value: float,
    peak_value: float,
    warmup_steps: int,
    decay_steps: int,
    end_value: float = 0.0,
) -> optax.Schedule:
    """Linear warmup followed up linear decay.

    Args:
        init_value: Initial value for the scalar to be annealed.
        peak_value: Peak value for scalar to be annealed at end of warmup.
        warmup_steps: Positive integer, the length of the linear warmup.
        decay_steps: Positive integer, the total length of the schedule. Note that
            this includes the warmup time, so the number of steps during which linear
            decay is applied is ``decay_steps - warmup_steps``.
        end_value: End value of the scalar to be annealed. Defaults to 0.

    Returns:
        A schedule function.
    """

    schedules = [
        optax.linear_schedule(
            init_value=init_value,
            end_value=peak_value,
            transition_steps=warmup_steps,
        ),
        linear_decay_schedule(
            init_value=peak_value,
            decay_steps=decay_steps-warmup_steps,
            end_value=end_value,
        ),
    ]
    return optax.join_schedules(schedules, [warmup_steps])