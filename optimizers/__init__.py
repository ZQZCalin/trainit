"""The optimizer subfolder.

Extends/re-implements existing algorithms in optax.
"""

from optimizers.base import (
    adam_base,
    adam,
    adamw,
    nadam,
    rmsprop,
    sgdm,
)
from optimizers.online_nonconvex import wrap_random_scaling
from optimizers.online_nonconvex import online_to_gradient_transformation
# from optimizers.online_nonconvex import online_to_non_convex
from optimizers.schedule import get_current_lr
from optimizers.schedule import warmup_linear_decay_schedule
from optimizers.muon.muon import scale_by_muon
from optimizers.muon.muon import muon