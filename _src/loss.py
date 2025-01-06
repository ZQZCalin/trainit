"""Loss functions."""

from omegaconf import DictConfig
from losses import ObjectiveFn
from losses import loss_to_objective
from losses import softmax_cross_entropy
from _src.base import classification_tasks


def init_loss_fn(
        config: DictConfig
) -> ObjectiveFn:
    """Initialize the optimization objective function.
    
    Args:
        config: global_config.
    
    Returns:
        An `ObjectiveFn` object.
    """
    # NOTE: For now, loss function should be only dependent on datasets.
    # For example, classification datasets like c4 and pile use cross-entropy.
    dataset_name = config.dataset.name
    if dataset_name in classification_tasks:
        return loss_to_objective(softmax_cross_entropy)
    else:
        raise ValueError("invalid config: cannot initialize loss function", 
                         f"because of unknown dataset '{dataset_name}'.")