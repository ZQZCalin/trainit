"""A stateless logger function."""

from omegaconf import DictConfig
import loggers


def init_logger(config: DictConfig) -> loggers.Logger:
    """Initializes the logger function.
    
    Args:
        config: global_config
    """
    logging_config = config.logging
    # NOTE: we always use full log for now.
    # name = config.logging.log_fn
    name = "full_log"

    if name == "simple_log":
        return loggers.simple_log()
    if name == "full_log":
        return loggers.full_log(logging_config)
    raise ValueError(f"invalid config: logging.log_fn '{name}' is not supported.")