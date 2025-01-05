"""A stateless logger function."""

from omegaconf import DictConfig
import loggers


def init_logger(config: DictConfig) -> loggers.Logger:
    """Initializes the logger function.
    
    Args:
        config: global_config.logging
    """
    logging_config = config.logging
    name = config.logging.log_fn

    if name == "simple_log":
        return loggers.simple_log()
    if name == "full_log":
        return loggers.full_log(logging_config)
    raise ValueError(f"invalid config: logging.log_fn '{name}' is not supported.")