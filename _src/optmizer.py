"""Optimizer algorithms."""

import optax
import optimizers
import equinox as eqx
from omegaconf import DictConfig
from typing import Tuple





def init_optimizer(
    model: eqx.Module,
    config: DictConfig,
    logger: None,
) -> Tuple[optax.GradientTransformation, optax.OptState]:
    """Construct optimizer from model and training config.

    Returns:
        Initial optimizer and opt_state.
    """
    # Initialize base optimizer / online learner.
    name = config.optimizer.name
    max_steps = config.train.max_steps

    def init_adamw(config: DictConfig, **kwargs):
        """use kwargs to pass down optional arguments (e.g., schedule_title)"""
        learning_rate = wrap_scheduler(
            init_scheduler(config.lr_config), logger=logger, **kwargs)
        return optimizers.adamw(
            learning_rate=learning_rate,
            beta1=config.beta1,
            beta2=config.beta2,
            weight_decay=config.weight_decay,
            debias_beta1=config.debias_beta1,
            debias_beta2=config.debias_beta2,
            use_momentum=config.use_momentum,
            use_preconditioning=config.use_preconditioning,
            decouple_weight_decay=config.decouple_weight_decay,
        )

    def init_sgdm(config: DictConfig, **kwargs):
        learning_rate = wrap_scheduler(
            init_scheduler(config.lr_config), logger=logger, **kwargs)
        return optimizers.sgdm(
            learning_rate=learning_rate,
            beta=config.beta,
            weight_decay=config.weight_decay
        )

    opt_config = config.optimizer
    if name == "adamw":
        optimizer = init_adamw(config=opt_config)
    elif name == "sgdm":
        optimizer = init_sgdm(config=opt_config)

    # Wrap online to non-convex.
    if name in ["ogd_md"]:
        wrap_o2nc = True
    elif name in ["adamw", "sgdm", "polar", "jump"]:
        wrap_o2nc = False
    else:
        wrap_o2nc = config.train.wrap_o2nc
    if wrap_o2nc:
        optimizer = optimizers.deterministic_online_nonconvex(optimizer)

    # Wrap random scaling.
    optimizer = optimizers.wrap_random_scaling(
        gradient_transformation=optimizer,
        random_scaling=config.train.random_scaling,
        use_importance_sampling=config.train.use_importance_sampling,
        seed=config.train.random_scaling_seed   # TODO: deprecate. use PRNGKey passed from argument instead of random seed.
    )
    
    # Gradient clipping and finite gradient wrapper.
    grad_clip = optax.clip_by_global_norm(config.train.gradient_clip_val)
    optimizer = optax.chain(
        grad_clip,
        optax.apply_if_finite(optimizer, 15)
    )

    # Initialize opt_state
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    return optimizer, opt_state