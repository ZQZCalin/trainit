"""The metrics subfolder.

Implements the LogFn class that computes metrics during training.
"""

from loggers.registry import init_logger
from loggers.base import Logger, LogState, LogMetrics
from loggers.base import get_internal_logs
from loggers.default import (
    minimal_logger,
    default_logger,
)