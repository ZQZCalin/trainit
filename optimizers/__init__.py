"""Extends optax optimizers."""

from optimizers.base import sgdm
from optimizers.base import adamw
from optimizers.online_nonconvex import deterministic_online_nonconvex
from optimizers.online_nonconvex import wrap_random_scaling