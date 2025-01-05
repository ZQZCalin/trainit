"""A stateless log function"""

import optax
from typing import Any, Dict, Tuple, NamedTuple, Protocol
from jaxtyping import Array, PyTree
from utils import merge_dicts, list_of_logs


LogState = PyTree
LogMetrics = PyTree


class LoggerInitFn(Protocol):
    def __call__(self, **extra_args: Any) -> LogState:
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
        ...  # log_state = logger.init(params=...)
        ...  # log_state, log_metrics = logger.update(log_state, params=..., grads=...)
    """
    # TODO: this is the only part I don't like about LogFn: users need to write different lines
    # in the train() function to update different log functions. Ideally, we'd want one uniform
    # line to achieve this.
    init: LoggerInitFn
    update: LoggerUpdateFn


def get_internal_logs(opt_state: optax.OptState) -> Dict[Array]:
    """Fetchs `utils.Log` objects, the internal logging metrics, from an `optax.OptState`."""
    return merge_dicts(*list_of_logs(opt_state))