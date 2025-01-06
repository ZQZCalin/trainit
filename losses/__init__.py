"""The loss subfolder.

Implements the Loss class and common loss functions.
"""

from losses.base import (
    LossFn,
    ObjectiveFn,
)
from losses.base import loss_to_objective
from losses.loss import softmax_cross_entropy