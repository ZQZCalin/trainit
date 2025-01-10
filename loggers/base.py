"""A stateless log function"""

import optax
from typing import Any, Dict, Tuple, NamedTuple, Protocol
from jaxtyping import Array, PyTree
from utils import merge_dicts, list_of_logs


LogState = NamedTuple
LogMetrics = dict[Array]


class LoggerInitFn(Protocol):
    def __call__(self, **extra_args: Any) -> Tuple[LogState, LogMetrics]:
        """The `init` function."""


class LoggerUpdateFn(Protocol):
    def __call__(self, state: LogState, **extra_args: Any) -> Tuple[LogState, LogMetrics]:
        """The `update` function."""


class Logger(NamedTuple):
    """A stateless log function implemented by a tuple of `init_fn` and `update_fn`.
    
    Log functions should only compute metrics based on input arguments and
    may *not* incur any extra computations, including forward and backward props.
    Such computations should be called outside the `update_fn`, whose results are then
    passed to the log function to compute associated metrics.

    For convenience of use, please include a use example of both `init_fn` and `update_fn`
    when customizing your own log function.

    Examples:
        >>> from loggings import base
        >>> def example_log() -> base.Logger:
        ...  '''An example log function.'''
        ...  # logger = example_log()
        ...  # log_state, log_metrics = logger.init(params=...)
        ...  # log_state, log_metrics = logger.update(log_state, params=..., grads=...)
    """
    # NOTE: for users who want to customize their own logger functions,
    # they need to manually change the lines (e.g., `train_step` and `train_loop`) in `_src/train`.
    init: LoggerInitFn
    update: LoggerUpdateFn


def get_internal_logs(opt_state: optax.OptState) -> dict[Array]:
    """Fetchs `utils.Log` objects, the internal logging metrics, from an `optax.OptState`."""
    return merge_dicts(*list_of_logs(opt_state))