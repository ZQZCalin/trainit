from omegaconf import DictConfig
from loggers.base import Logger
from loggers.default import minimal_logger, default_logger, empty_logger
from loggers.meta_grad import meta_grad_logger


def init_logger(config: DictConfig) -> Logger:
    """Initializes the logger function.
    
    Args:
        config: global_config
    """
    name = config.logger.logger_name

    if name == "empty" or not config.logging.log_callback_data:
        return empty_logger()
    if name == "minimal":
        return minimal_logger()
    if name == "default":
        return default_logger(config.logger)
    if name == "meta_grad":
        return meta_grad_logger(config.logger)
    raise ValueError(f"invalid config: logging.log_fn '{name}' is not supported.")