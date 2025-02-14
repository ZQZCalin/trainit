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
from optimizers.combine import (
    multi_transform
)
from optimizers.online_nonconvex import wrap_random_scaling
from optimizers.online_nonconvex import online_to_gradient_transformation
# from optimizers.online_nonconvex import online_to_non_convex
from optimizers.schedule import get_current_lr
from optimizers.schedule import (
    warmup_linear_decay_schedule,
    warmup_const_linear_decay_schedule,
    trapezoid_schedule,
)
from optimizers.muon.muon import scale_by_muon
from optimizers.muon.muon import (
    muon,
    muon_og,
)
from optimizers.muon.muon_laprop import (
    muon_laprop,
    muon_adamw,
)
from optimizers.muon.mango import (
    mango,
    mango_v2,
    mango_v3,
    visualize_norm,
)
from optimizers.preconditioners.combine import (
    adamw_2dmask,
)
from optimizers.preconditioners.normalized_sgdm import normalized_sgdm
from optimizers.optim_test import test_optimizer