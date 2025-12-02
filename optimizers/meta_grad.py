import optax
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from typing import NamedTuple, Optional, Protocol
from jaxtyping import Array, PyTree
from utils import tree_utils, log_utils
from optimizers.schedule import get_current_lr, init_schedule
from omegaconf import OmegaConf, DictConfig


class MetaGradientRegularization(Protocol):
    """Gradient function for meta-gradient regularization.
    
    Requires to implement the forward() function.
    __call__ function defaults to output the gradient of forward() wrt current lr.
    """

    def forward(self, lr: Array, *args: Array, **kwargs: Array) -> Array:
        """The gradient function that maps (lr, *args) into
        the gradient of the regularization function at the current lr.
        """
        raise NotImplementedError

    def __call__(self, lr: Array, *args: Array, **kwargs: Array) -> Array:
        """Autodiff that outputs the gradient at current lr."""
        grad_fn = jax.grad(self.forward, argnums=0)
        return grad_fn(lr, *args, **kwargs)


@jax.jit
def _safe_log_tanh(lr, lam, omega):
    lr = jnp.clip(lr, jnp.finfo(lr.dtype).tiny, None)
    lam = jnp.asarray(lam, dtype=lr.dtype)
    omega = jnp.asarray(omega, dtype=lr.dtype)
    s = omega * jnp.log10(lr)
    return -lam * jnp.tanh(s)


class LogTanhRegularization(MetaGradientRegularization):
    """R_t(lr) = lam_t * tanh(-omega_t * log(lr))"""

    def __init__(
            self,
            regularization: optax.ScalarOrSchedule,
            decay: optax.ScalarOrSchedule,
    ):
        self.regularization = regularization
        self.decay = decay

    def forward(self, lr, count, *args, **kwargs):
        lam = get_current_lr(self.regularization, count)
        omega = get_current_lr(self.decay, count)
        return _safe_log_tanh(lr, lam, omega)


class MetaGradState(NamedTuple):
    """wrap_with_meta_gradient state."""
    lr: Array
    prev_lr: Array
    count: Array
    prev_updates: PyTree
    momentum_buffer: PyTree
    inner_state: optax.OptState
    log_state: log_utils.Log


def wrap_with_meta_gradient(
        inner: optax.GradientTransformation,
        base_lr: float,
        clip_min: float = 1e-8,
        clip_max: float = None,
        momentum: float = 0.0,
        learning_rate: optax.ScalarOrSchedule = 1e-6,
        regularization_fn: MetaGradientRegularization = None,
        aux_output_lr: optax.ScalarOrSchedule = None,
) -> optax.GradientTransformation:
    """Wrap an optimizer with meta-gradient for lr update.

    Given any regularization function R(eta), regularized meta-gradient updates LR as
        
        eta_{t+1} = eta_t - beta_t * (<u_{t-1}, g_t> + \\nabla R(eta_t)),

    where beta_t is the stepsize for meta-gradient, u_t denotes update at step t, and g_t is the gradient.
    
    This should be wrapped around any optimizer whose lr should be set to constantly one.
    The wrapper will automatically applies a lr scaling according to regularized meta gradient.

    There are different modes of aggregating meta gradient with windowed EMA of gradients

    Args:
        inner: inner optimizer to be wrapped
        base_lr: base learning rate to start from
        clip_min: minimum value of lr
        clip_max: maximum value of lr
        momentum: momentum for SGDM of meta-grad
        learning_rate: lr schedule for meta-gradient descent
        regularization_fn: a regularization function that maps (lr, count, prev_lr) into gradient of lr. 
            If None, no regularization and becomes vanilla meta gradient.
        momentum_mode:
            There are different modes of aggregating meta gradient with windowed EMA of gradients.
            Defaults to None, where no momentum buffer is used.
            If set to "total_grad", computes momentum buffer of the total regularized gradient
                m_t = b * m_{t-1} + (1-b) * (<u_{t-1}, g_t> + \\nabla R(eta_t)),
                \\hat m_t = m_t / (1 - beta**t)
            and updates
                eta_{t+1} = eta_t - beta_t * \\hat m_t
            If set to "update", computes momentum buffer on previous updates
                m_t = b * m_{t-1} + (1-b) * u_{t-1},
                \\hat m_t = m_t / (1 - beta**t)
            and updates
                eta_{t+1} = eta_t - beta_t * (<\\hat u_{t-1}, g_t> + \\nabla R(eta_t)),
        aux_output_lr: defaults to None. if specified, aux_output_lr will overwrite the meta-gradient method.

    ..Example::

        # TBD
    """

    def get_log_state(lr, meta_grad, reg_grad):
        return log_utils.Log({
            "meta_grad_lr": lr,
            "meta_grad": meta_grad,
            "reg_grad": reg_grad,
        })

    def init_fn(params: optax.Params):
        if aux_output_lr is None:
            lr = jnp.asarray(base_lr, dtype=jnp.float32) 
        else: 
            lr = get_current_lr(aux_output_lr, 0)
        return MetaGradState(
            lr=lr,
            prev_lr=lr,
            count=jnp.zeros([], jnp.int32),
            prev_updates=jtu.tree_map(jnp.zeros_like, params),
            momentum_buffer=jnp.zeros([]),
            inner_state=inner.init(params),
            log_state=get_log_state(base_lr, 0.0, 0.0),
        )
    
    def update_fn(updates: optax.Updates, state: MetaGradState, params: optax.Params):
        lr = state.lr
        count = state.count
        count_inc = optax.safe_int32_increment(count)
        momentum_buffer = state.momentum_buffer

        meta_grad = tree_utils.inner(updates, state.prev_updates)
        reg_grad = regularization_fn(lr=lr, count=count, prev_lr=state.prev_lr)
        if aux_output_lr is None:
            beta = get_current_lr(learning_rate, count)
            momentum_buffer = momentum * momentum_buffer + (1-momentum_buffer) * (meta_grad + reg_grad)
            new_lr = lr - beta * momentum_buffer
            new_lr = jnp.clip(new_lr, min=clip_min, max=clip_max)
        else:
            # If aux_output_lr is provided, uses the provided schedule as the output lr.
            new_lr = get_current_lr(aux_output_lr, count)

        updates, inner_state = inner.update(
            updates, state.inner_state, params)
        updates = tree_utils.scalar_dot(updates, new_lr)

        return updates, MetaGradState(
            lr=new_lr,
            prev_lr=lr,
            count=count_inc,
            prev_updates=updates,
            momentum_buffer=momentum_buffer,
            inner_state=inner_state,
            log_state=get_log_state(new_lr, meta_grad, reg_grad),
        )
    
    return optax.GradientTransformation(init_fn, update_fn)


def init_meta_gradient_regularizer(config: DictConfig) -> MetaGradientRegularization:
    def init_log_tanh(config):
        return LogTanhRegularization(
            regularization=init_schedule(config.regularization),
            decay=init_schedule(config.decay)
        )

    if config.name == "log_tanh":
        return init_log_tanh(config)
    else:
        raise ValueError(f"unsupported meta-grad regularizer '{config.name}'")


def init_wrap_with_meta_gradient(
        inner: optax.GradientTransformation, 
        config: DictConfig,
) -> optax.GradientTransformation:
    return wrap_with_meta_gradient(
        inner=inner,
        base_lr=config.base_lr,
        clip_min=config.clip_min,
        clip_max=config.clip_max,
        momentum=config.momentum,
        learning_rate=init_schedule(config.learning_rate),
        regularization_fn=init_meta_gradient_regularizer(config.regularizer),
        aux_output_lr=None if config.aux_output_lr is None else init_schedule(config.aux_output_lr),
    )


#TODO: add double EMA to incorporate the idea of segment/windowed average of regularized loss