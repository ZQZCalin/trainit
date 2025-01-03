"""A stateless log function"""

from omegaconf import DictConfig
import loggings


def init_log(config: DictConfig) -> loggings.LogFn:
    """Initializes the log function.
    
    Args:
        config: global_config.logging
    """
    logging_config = config.logging
    name = config.logging.log_fn

    if name == "simple_log":
        return loggings.simple_log()
    if name == "full_log":
        return loggings.full_log(logging_config)
    raise ValueError(f"invalid config: logging.log_fn '{name}' is not supported.")