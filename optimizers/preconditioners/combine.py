"""Implements common combine methods for preconditioner algorithms."""

import jax
import jax.tree_util as jtu
import optax
from typing import Any, Union, Mapping, Hashable, Callable, NamedTuple
from jaxtyping import PyTree
from optimizers.base import adamw
from optimizers.combine import multi_transform


def adamw_2dmask(
        optimizer: optax.GradientTransformation,
        adam_lr: optax.ScalarOrSchedule = 3e-4,
        adam_beta1: float = 0.95,
        adam_beta2: float = 0.95,
        adam_eps: float = 1e-8,
        adam_wd: float = 0.0,
        adam_nesterov: bool = False,
) -> optax.GradientTransformation:
    """The adamw mask wrapper.
    
    Applies input optimizer update on 2d parameters and
    applies adamw update on the rest.

    Args:
        optimizer: the target optimizer.
        adam_lr: adam learning rate.
        adam_beta1: adam beta1.
        adam_beta2: adam beta2.
        adam_eps: adam eps.
        adam_wd: adam weight decay.
        adam_nesterov: adam nesterov update.
    """
    adamw_optim = adamw(
        learning_rate=adam_lr,
        beta1=adam_beta1,
        beta2=adam_beta2,
        eps=adam_eps,
        weight_decay=adam_wd,
        use_nesterov=adam_nesterov,
    )
    transforms = {
        "optim": optimizer,
        "adamw": adamw_optim,
    }
    def label_params(params):
        return jtu.tree_map(
            lambda p: "optim" if p.ndim == 2 else "adamw", params
        )
    return multi_transform(transforms, label_params)