"""Online convex optimization algorithms."""

import jax
from jax import numpy as jnp
from jax import random as jr
from jax import tree_util as jtu
import chex
import optax
from optax import Updates, Params, OptState, ScalarOrSchedule, GradientTransformation
from typing import Any, Tuple, NamedTuple, Optional, Union, Callable, Protocol
from utils import tree_utils, log_utils
import optimizers.schedule as schedule


class OnlineLearnerInitFn(Protocol):
    def __call__(self, params: Params) -> OptState:
        """The `init` function.

        Args:
            params: The initial parameters of the online learner. 
            Note: this can be different from the model parameter if the online learner is used as the subroutine
            of the online-to-non-convex conversion.

        Returns:
            The initial state of the online learner.
        """


class OnlineLearnerUpdateFn(Protocol):
    def __call__(
        self, 
        grads: Updates, 
        state: OptState,
        params: Optional[Params] = None,
    ) -> Tuple[Params, OptState]:
        """The `update` function.

        Args:
            grads: A tree of gradients.
            state: The state of the online learner.
            params: (Optionally) the current parameters w_t.

        Returns:
            The new parameter w_{t+1}, and the updated state.
        """


class OnlineLearnerUpdateExtraArgsFn(Protocol):
    def __call__(
        self,
        grads: Updates,
        state: OptState,
        params: Optional[Params] = None,
        **extra_args: Any,
    ) -> Tuple[Params, OptState]:
        """OnlineLearner update function with extra arguments (e.g., for FTRL)."""


class OnlineLearner(NamedTuple):
    """A pair of init and update functions implementing online learners.

    In our context, online learners aim to minimize the "anytime weighted and regularized regret":
        Regret_T(u) = sum_{t=1}^T beta^{T-t} * (<g_t, w_t - u> + mu/2*|w_t|^2),
    where beta in [0,1] denotes the "exponentiated gradient constant" (which can be equivalently viewed as a momentum constant)
    and mu > 0 denotes the l2 regularization constant.

    For beta = 1 and mu = 0, the above definition just recovers the standard definition of regret. In general, given T
    one can easily convert a "standard" online learner into a "modified" online learner by changing the loss from g_t to 
    tilde g_t := beta^{T-t} * (g_t + mu*w_t). When T is unknown a priori, we can still try to minimize
        beta^T * sum_{t=1}^T beta^-t * (<g_t, w_t - u> + mu/2*|w_t|^2),
    by changing the loss to tilde g_t := beta^-t * (g_t + mu*w_t). However, note a caveat that the Lipschitz constant of 
    tilde g_t grows exponentially, so the algorithm design needs to be modified as well to compensate this growth.

    As an example, consider Online Gradient Descent (OGD) with non-increasing learning rates eta_t, which guarantees
        Regret_T(u) <= D^2 / eta_{T+1} + sum_{t=1}^T eta_t * |g_t|^2.
    With g_t replaced by tilde g_t (with mu=0 for simplicity), we can check that eta_t = eta * beta^t
    actually finds the optimal "modified" regret of order O(D^2/eta + eta/(1-beta)) for any unknown T.
    If we combine the learning rate with the gradient tilde g_t, note that the exponentials cancel each out and the update
    remains to be w_{t+1} = w_t - eta * (g_t + mu*w_t), which is exactly the standard OGD (with weight decay).

    Unlike typical optax GradientTransformations, upon calling the update function, an OnlineLearner returns the 
    new parameter (instead of an update) together with the updated state. 
    The motivation is that most online learners (mirror descent, FTRL, parameter-free algorithms, etc.) do not have 
    an iterative update expression such as w_{t+1} = w_t + eta * g_t. Hence, it would be easier to directly return the
    new parameter instead of new_params - params.
    """
    init: OnlineLearnerInitFn
    update: OnlineLearnerUpdateFn


class OnlineLearnerExtraArgs(OnlineLearner):
    """Online learner with extra arguments (e.g., FTRL).
    
    Overwrites the update function in base OnlineLearner class.
    """
    update: OnlineLearnerUpdateExtraArgsFn




"""NOTE: 
- Updated on 2025/01/02:
    Everything below this line is still under dev.
    I still need to adapt everything to the new framework, fix some import issues, 
    and correct some implementation errors. 
    For now, I put a NotImplementedError on all algorithms.
"""


class ScaleByOnlineLearnerState(NamedTuple):
    """Scale by Online Learner State"""
    Delta: Updates
    opt_state: OptState


# Note: this can be freely replaced with any online learner that returns Delta in each iteration.
# This only serves as an easy wrapper for popular online learners; and I can imagine for parameter-free
# algorithms, implemeting them from scratch would be easier.
def scale_by_online_learner(
    ol_optimizer: GradientTransformation,
    projection_norm: Optional[float] = None,
    exponentiated_gradient: bool = False,
) -> GradientTransformation:
    """Updates online learner and returns the updated parameter Delta_t.

    Args:
        ol_optimizer (GradientTransformation): The online learner optimizer used to update Delta (i.e., returning Delta_n-Delta_{n-1}).
        projection_norm (Optional[float]): If not None, clip the parameter to global norm <= `projection`. Defaults to None.
        exponentiated_gradient (bool): Whether it is used in standard O2NC (False) or Exponentiated O2NC (True). Defaults to False.

    Returns:
        A `GradientTransformation` object.
    """
    raise NotImplementedError
    
    def init_fn(params):
        Delta = jtu.tree_map(jnp.zeros_like, params)
        opt_state = ol_optimizer.init(Delta)
        return ScaleByOnlineLearnerState(
            Delta=Delta, opt_state=opt_state)
    
    def update_fn(updates, state, params=None):
        # Performs a one-step update of the online learner:
        #   Updates the params (Delta) and opt_state of the online learner.
        # 
        # Here updates can be either a pytree of g_t (in standard O2NC)
        # or a pytree of tuple (g_t, beta_t, mu_t) (in Exponentiated O2NC).
        # ====================================================================================
        del params
        # Compute gradient of the exponentiated and l2-regularized loss
        def linearize(updates_, params_):
            g, beta, mu = updates_
            return beta*g + mu*params_
        if exponentiated_gradient:
            updates = jtu.tree_map(
                linearize, updates, state.Delta)
        # Update Delta.
        Delta_updates, opt_state = state.ol_optimizer.update(
            updates, state.opt_state, state.Delta)
        Delta = optax.apply_updates(state.Delta, Delta_updates)
        # Optional: project Delta into a constrained domain.
        if projection_norm:
            clip_norm = jnp.minimum(1, projection_norm/optax.global_norm(Delta))
            Delta = tree_utils.scalar_dot(Delta, clip_norm)
        # TODO: return state.Delta or new Delta?
        # need to check which one adheres to the notion in the references.
        return Delta, ScaleByOnlineLearnerState(
            Delta=Delta, opt_state=opt_state)
    
    return GradientTransformation(init_fn, update_fn)


# Next, we also provide a more general wrapper for online learners.
# If you want to use an online learner as part of a larger optimizer (say O2NC), just
# wrap the online learner with this wrapper.
class WrapOnlineLearnerState(NamedTuple):
    """online learner wrapper state."""
    params: Updates 
    state: OptState


def wrap_online_learner(
    online_learner: OnlineLearner
) -> OnlineLearner:
    """Wraps an online learenr.

    Automatically stores the params of the online learner, which may be different from the
    model parameter. This wrapper can be useful if the online learner is used as a subroutine
    in the online-to-non-convex conversion.

    Args:
        online_learner: An `OnlineLearner` object to be wrapped.

    Returns:
        A wrapped `OnlineLearner` object.
    """
    raise NotImplementedError

    def init_fn(params):
        state = online_learner.init(params)
        return WrapOnlineLearnerState(
            params=params, state=state)
    
    def update_fn(updates, state, params=None):
        del params
        new_params, state = online_learner.update(updates, state.state, state.params)
        return new_params, WrapOnlineLearnerState(params=new_params, state=state)

    return OnlineLearner(init_fn, update_fn)


# ======================================================================
# Below implements popular online learners.
# ======================================================================

class OGDState(NamedTuple):
    """ogd state."""
    count: chex.Array


def ogd(
    learning_rate: optax.ScalarOrSchedule,
    weight_decay: float = 0.0,
) -> OnlineLearner:
    """Online Gradient Descent (OGD).

    Updates w_{t+1} = w_t + eta_t * g_t. 
    See previous explanation for why vanilla OGD adapts to any "modified regret" with any beta.

    Args:
        learning_rate: OGD learning rate.
        weight_decay: l2 regularization constant. Defaults to 0.0 (no regularization).
    """
    raise NotImplementedError

    def init_fn(params=None):
        del params
        return OGDState(count=jnp.zeros([], jnp.int32))
    
    def update_fn(updates, state, params):
        # l2 regularization / weight decay
        grads = jtu.tree_map(
            lambda g, w: g + weight_decay*w, updates, params)
        # gradient descent
        count_inc = optax.safe_int32_increment(state.count)
        eta = schedule.get_current_lr(learning_rate, state.count)
        new_params = jtu.tree_map(
            lambda w, g: w - eta*g, params, grads)
        return new_params, OGDState(count=count_inc)
    
    return OnlineLearner(init_fn, update_fn)


class OGDMirrorDescentState(NamedTuple):
    """ogd_mirror_descent state."""
    count: chex.Array
    

def ogd_mirror_descent(
    learning_rate: optax.ScalarOrSchedule,
    beta: float = 1.0,
    mu: float = 0.0,
) -> OnlineLearner:
    """A variant of OGD derived from online mirror descent.

    Updates w_{t+1} = (w_t - eta_t * g_t) * [beta / (1 + eta_t*mu)].

    Args:
        learning_rate: Learning rate scheduler.
        beta: Momentum constant. Defaults to 1.0.
        mu: Weight decay constant (although implemented in an implicit way). Defaults to 0.0.

    Returns:
        OnlineLearner: _description_
    """
    raise NotImplementedError

    def init_fn(params=None):
        del params
        return OGDMirrorDescentState(count=jnp.zeros([], jnp.int32))
    
    def update_fn(updates, state, params):
        count_inc = optax.safe_int32_increment(state.count)
        eta = schedule.get_current_lr(learning_rate, state.count)
        new_params = jtu.tree_map(
            lambda w, g: (w - eta*g) * beta/(1+eta*mu), params, updates)
        return new_params, OGDMirrorDescentState(count=count_inc)
    
    return OnlineLearner(init_fn, update_fn)


# TODO: 
# SGDM with constant beta schedule.
# 1. ogd_md with hints 
# 2. ogd_md with different learning rates: adaptive lr, other schedulers...
# Q: is cosine scheduler important to achieve a good performance? If so, why is the reason; if not, can we replace with other lr schedulers?
# and with optimistic online learners.
# 3. FTRL



class UnconstrainedOGDState(NamedTuple):
    """Unconstrained OGD State."""
    count: chex.Array
    Delta: Updates


# TODO: deprecate unconstrained_ogd and use ogd_mirror_descent instead.
def unconstrained_ogd(
    learning_rate: optax.ScalarOrSchedule,
    beta: float = 0.99,
    mu: float = 100.0,
) -> GradientTransformation:
    """Unconstrained OGD as implemented in [Exponentiated O2NC](XXX).

    Given learning rate eta, exponentiate constant beta, and regularzation factor mu, updates
    Delta <- beta/(1+eta*mu) * (Delta - eta*grad).

    Note that this is equivalent to weight decay, with slightly different weight_decay constant.

    Set beta = 1 and mu = 0 recovers standard OGD.

    Args:
        learning_rate (optax.ScalarOrSchedule): _description_
        beta (float, optional): _description_. Defaults to 0.99.
        mu (float, optional): _description_. Defaults to 100.

    Returns:
        A `GradientTransformation` object.
    """
    raise NotImplementedError
    
    def init_fn(params):
        return UnconstrainedOGDState(
            count=jnp.zeros([], jnp.int32),
            Delta=jtu.tree_map(jnp.zeros_like, params))
    
    def update_fn(updates, state, params=None):
        del params
        count_inc = optax.safe_int32_increment(state.count)
        if callable(learning_rate):
            # eta = learning_rate(count_inc)
            eta = learning_rate(state.count)
        else:
            eta = learning_rate
        new_Delta = jtu.tree_map(
            lambda Delta, g: beta/(1+eta*mu) * (Delta-eta*g),
            state.Delta, updates
        )
        return new_Delta, UnconstrainedOGDState(
            count=count_inc, Delta=new_Delta)
    
    return GradientTransformation(init_fn, update_fn)


class FTRLState(NamedTuple):
    """ftrl state."""


def ftrl() -> OnlineLearnerExtraArgs:
    """Family of Follow-the-regularized-leader algorithms.

    Updates w_{t+1} = argmin_w psi_t(w) + sum_{i=1}^t ell_i(w),
        where psi_t is a linearized regularizer of form <r_t, w> and ell_t is a linearized loss of form <g_t, w>.
    """
    raise NotImplementedError

    def init_fn(params):
        return FTRLState()
    
    def update_fn(grads, state, params):
        return new_params, FTRLState()
    
    return OnlineLearnerExtraArgs(init_fn, update_fn)


class AdaFTRLState(NamedTuple):
    """Adaptive FTRL State."""
    count: chex.Array
    mu: Updates
    nu: Updates


# TODO: add regularization (mu).
def ada_ftrl(
    learning_rate: ScalarOrSchedule,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    scale_exponential: bool = False,
    scale_lr: bool = False,
    scale_eps: bool = False,
) -> GradientTransformation:
    """Ada-FTRL.

    See notes for the update.

    **Note:** with scale_exponential = False, this algorithm is almost Adam, up to two differences:
        - mu and nu are not debiased;
        - eps is scaled by an exponential decay beta2**(t/2).

    Args:
        learning_rate (ScalarOrSchedule): _description_
        beta1 (float, optional): _description_. Defaults to 0.9.
        beta2 (float, optional): _description_. Defaults to 0.999.
        eps (float, optional): _description_. Defaults to 1e-8.
        scale_exponential (bool, optional): If true, scale the update by (sqrt(beta2)/beta1)**t. Defaults to False.
        scale_lr (bool, optional): If true, scale the learning rate by sqrt(1-beta2)/(1-beta1). Defaults to False.
        scale_eps (bool, optional): If true, scale eps by sqrt(1-beta2). Defaults to False.

    Returns:
        A `GradientTransformation` object.
    """
    raise NotImplementedError

    if scale_lr:
        scale_lr_const = (1-beta2)**.5 / (1-beta1)
        if callable(learning_rate):
            learning_rate = lambda n: scale_lr_const * learning_rate(n)
        else:
            learning_rate *= scale_lr_const
    
    if scale_eps:
        eps *= (1-beta2)**.5

    def init_fn(params):
        return AdaFTRLState(
            count=jnp.zeros([], jnp.int32),
            mu=jtu.tree_map(jnp.zeros_like, params),
            nu=jtu.tree_map(jnp.zeros_like, params)
        )
    
    def update_fn(updates, state, params=None):
        del params
        mu = jtu.tree_map(
            lambda m, g: beta1*m + (1-beta1)*g, state.mu, updates)
        nu = jtu.tree_map(
            lambda v, g: beta2*v + (1-beta2)*g**2, state.nu, updates)
        if scale_exponential:
            scalar = (beta2**.5/beta1)**state.count
        else:
            scalar = 1
        if callable(learning_rate):
            eta = learning_rate(state.count)
        else:
            eta = learning_rate
        Delta = jtu.tree_map(
            lambda m, v: -scalar * eta * m / (beta2**(state.count/2)*eps + jnp.sqrt(v)),
            mu, nu)
        return Delta, AdaFTRLState(
            count=optax.safe_int32_increment(state.count), mu=mu, nu=nu)
    
    return GradientTransformation(init_fn, update_fn)


class KTBettorState(NamedTuple):
    """KT coin better state."""
    sum_grad: Updates
    wealth: Updates
    count: chex.Array
    logging: Optional[log_utils.Log]


# TODO: support Pytree argument for eps
def kt_bettor(
    eps: float = 1e2,
    G: float = 1.0,
    log_reward: bool = False,
) -> OnlineLearner:
    """KT Coin Bettor.

    If dimension is higher than 1, then implements per-coordinate KT coin bettor.
    Unlike other online learners, the initial parameter should be set to zeros in most cases.

    References:
        [Orabona, 2019, Alg. 9.2](https://arxiv.org/abs/1912.13213)

    Args:
        eps (float or Pytree): Initial wealth between 1 and sqrt(T). Defaults to 100.
            Currently only supports float32 type argument.
        G: Estimated Lipschitz constant.
        log_reward: If true, log cumulative reward to wandb. Defaults to False.

    Returns:
        A `GradientTransformation` object.
    """
    raise NotImplementedError

    class KTBettorLog():
        def __call__(self, params, wealth) -> Optional[logstate.Log]:
            if not log_reward:
                return None
            total_wealth = utils.tree_l1_norm(wealth)
            return logstate.Log({
                "KT/params": utils.tree_l1_norm(params),
                "KT/reward": eps + total_wealth,
                "KT/net_wealth": G * total_wealth,
            })
        
    log = KTBettorLog()

    def init_fn(params):
        sum_grad = jtu.tree_map(jnp.zeros_like, params)
        wealth = jtu.tree_map(jnp.zeros_like, params)
        return KTBettorState(
            sum_grad=sum_grad,
            wealth=wealth,
            count=jnp.ones([], jnp.int32),
            logging=log(params, wealth)
        )
    
    def update_fn(updates, state, params):
        # NOTE: gradient is scaled down by Lipschitz constant,
        # i.e., sn -> sn/G.
        updates = tree_scalar_multiply(updates, 1/G)
        count_inc = optax.safe_int32_increment(state.count)
        sum_grad = tree_add(state.sum_grad, updates)
        wealth = tree_subtract(state.wealth, tree_multiply(updates, params))
        new_params = jtu.tree_map(
            lambda St, Wt: - St / count_inc * (eps + Wt), sum_grad, wealth)
        return new_params, KTBettorState(
            sum_grad=sum_grad, wealth=wealth, count=count_inc, logging=log(new_params, wealth))
    
    return OnlineLearner(init_fn, update_fn)


class BlackboxFTRLState(NamedTuple):
    """FTRL (for blackbox reduction) state."""
    momentum: Updates


def blackbox_ftrl(
    beta: float = 1.0
) -> OnlineLearner:
    """Implements FTRL projected on a unit ball with exponentiated gradients
    equal to beta**-t * gt. (Note that gt is automatically regularized by blackbox reduction.) 

    Args:
        beta: Exponentiation constant between 0 and 1. Defaults to 1.0 (no exponentiation).
    """
    raise NotImplementedError
    
    assert beta >= 0 and beta <= 1, "beta must be between 0 and 1."

    def init_fn(params):
        return BlackboxFTRLState(momentum=jtu.tree_map(jnp.zeros_like, params))
    
    def update_fn(updates, state, params=None):
        del params
        if beta == 1.0:
            momentum = jtu.tree_map(
                lambda m, g: m - g, state.momentum, updates)
        else:
            momentum = jtu.tree_map(
                lambda m, g: beta*m - (1-beta)*g, state.momentum, updates)
        return tree_normalize(momentum), BlackboxFTRLState(momentum)

    return OnlineLearner(init_fn, update_fn)


# TODO: add sign(s1:t) for output.
class BlackboxReductionState(NamedTuple):
    """Black box reduction state."""
    magnitude_params: Params
    direction_params: Params
    magnitude_state: OptState
    direction_state: OptState


# TODO: implement different scaling for each parameter.
def blackbox_reduction(
    magnitude_learner: OnlineLearner,
    direction_learner: OnlineLearner,
    weight_decay: float = 0.0,
) -> OnlineLearner:
    """Black-box reduction algorithm.

    References:
        [Cutkosky & Orabona, 2018](https://arxiv.org/abs/1802.06293)

    To adapt to exponentiated and regularized loss, we slightly modify the blackbox reduction algorithm.
    Specifically, given exp-reg gradient gt_tilde = beta**-t * (gt + mu*wt), we send gt_tilde to the direction learner,
    but we send the non-exponentiated scalar gradient, namely <gt + mu*wt, wt>, to the parameter-free 1d learner.
    The intuition is that we want the 1d learner to learn the optimal sequence of beta**t * lr. A theoretical support for 
    this modification is yet to be established.

    For computation efficiency, blackbox reduction is automatically wrapped and params are stored.

    Args:
        magnitude_learner: Online learner (typically 1D parameter-free algorithms) for magnitude; learns |xt|.
        direction_learner: Online learner for direction; learns xt/|xt|.
        weight_decay: Regularization constant. Defaults to 0.0 (no regularization).
    """
    raise NotImplementedError
    
    def init_fn(params):
        zt, xt = utils.tree_norm_direction_decomposition(params)
        magnitude_state = magnitude_learner.init(zt)
        direction_state = direction_learner.init(xt)
        return BlackboxReductionState(
            magnitude_params=zt,
            direction_params=xt,
            magnitude_state=magnitude_state,
            direction_state=direction_state
        )
    
    def update_fn(updates, state, params=None):
        del params
        zt, xt = state.magnitude_params, state.direction_params
        params = tree_scalar_multiply(xt, zt)
        st = utils.tree_inner_product(updates, xt) + weight_decay * zt
        gt_tilde = jtu.tree_map(
            lambda g, w: g + weight_decay*w, updates, params)
        new_zt, magnitude_state = magnitude_learner.update(
            st, state.magnitude_state, zt)
        new_xt, direction_state = direction_learner.update(
            gt_tilde, state.direction_state, xt)
        new_params = tree_scalar_multiply(new_xt, new_zt)

        return new_params, BlackboxReductionState(
            magnitude_params=new_zt,
            direction_params=new_xt,
            magnitude_state=magnitude_state,
            direction_state=direction_state
        )
    
    return OnlineLearner(init_fn, update_fn)


class NormalizedBlackboxState(NamedTuple):
    """normalized 1d to dimension-free reduction state."""
    sum_st: Updates
    sum_gt: Updates
    base_params: Params
    base_state: OptState
    key: jax.Array


def normalized_blackbox(
    base_learner: OnlineLearner,
    beta: float = 1.0,
    weight_decay: float = 0.0,
    seed: int = 0,
    per_layer: bool = False,
) -> OnlineLearner:
    """One-dimension to dimension-free reduction.

    Args:
        base_learner: 1d learner.
        bet: Exponentiated gradient constant. Defaults to 1.0.
        weight_decay: l2 regularization constant. Defaults to 0.0.
        seed: PRNGKey seed. Defaults to 0.
        per_layer: If true, updates each node of the Pytree using coordinate-wise base learner.
    """
    raise NotImplementedError

    def init_fn(params):
        # For now, always initialize the 1d learner with zt=0.
        if per_layer:
            zero = jtu.tree_map(lambda _: jnp.zeros([], jnp.float32), params)
        else:
            zero = jnp.zeros([], jnp.float32)
        jax.debug.print(">>>zero = {x}", x=zero)
        base_state = base_learner.init(zero)
        return NormalizedBlackboxState(
            sum_st=zero,
            sum_gt=jtu.tree_map(jnp.zeros_like, params),
            base_params=zero,
            base_state=base_state,
            key=jr.PRNGKey(seed),
        )
    
    def update_fn(updates, state, params):
        gt = jtu.tree_map(
            lambda g, w: g + weight_decay*w, updates, params)
        # compute s1 = <gt, v> for some random unit vector v in the first iteration
        # and compute st = sign(s_{1:t-1}) * <gt, g_{1:t-1}/|g_{1:t-1}|>.
        def inner_product(arr1, arr2):
            return jnp.dot(arr1.ravel(), arr2.ravel())
        def true_fun(_):
            key, v = utils.random_unit_vector(state.key, gt)
            # st = util.tree_inner_product(gt, v)
            st = jtu.tree_map(inner_product, gt, v)
            return key, st
        def false_fun(_):
            # st = jnp.sign(state.sum_st) * util.tree_inner_product(gt, tree_normalize(state.sum_gt))
            st = jtu.tree_map(
                lambda sum_si, gi, sum_gi: jnp.sign(sum_si) * inner_product(gi, sum_gi) / jnp.linalg.norm(sum_gi),
                state.sum_st, gt, state.sum_gt
            )
            return state.key, st
        key, st = jax.lax.cond(
            utils.is_zero_tree(state.sum_gt), true_fun, false_fun, operand=None)
        zt, base_state = base_learner.update(st, state.base_state, state.base_params)
        sum_st = tree_add(state.sum_st, st)
        if beta == 1.0:
            sum_gt = tree_add(state.sum_gt, gt)
        else:
            # Since we only use normalized sum_gt, it's ok to use the biased aggregation.
            sum_gt = jtu.tree_map(
                lambda m, g: beta*m + (1-beta)*g, state.sum_gt, gt)
        # xt = tree_scalar_multiply(tree_normalize(sum_gt), zt*jnp.sign(sum_st))
        xt = jtu.tree_map(lambda z, sum_g, sum_s: jnp.sign(sum_s)*z*sum_g/jnp.linalg.norm(sum_g), zt, sum_gt, sum_st)
        return xt, NormalizedBlackboxState(
            sum_st=sum_st,
            sum_gt=sum_gt,
            base_params=zt,
            base_state=base_state,
            key=key
        )
    
    return OnlineLearner(init_fn, update_fn)


class ParameterFreeMirrorDescentState(NamedTuple):
    """parameter_free_mirror_descent state"""
    V: Updates
    params: list[Params]


def parameter_free_mirror_descent(
    G: float,
    eps: float,
    num_grids: int=1,
) -> OnlineLearner:
    raise NotImplementedError
    
    # TODO: add an option for a list of customized schedules.
    etas = [2**-k / G for k in range(num_grids)]
    eps = eps / num_grids
    
    def init_fn(params):
        return ParameterFreeMirrorDescentState(
            V=jnp.array([4*G**2]),
            params=[utils.zero_tree(params) for _ in range(num_grids)]
        )
    
    def update_fn(updates, state, params=None):
        del params
        grads_sqnorm = utils.tree_l2_norm(updates)**2
        new_V = state.V + grads_sqnorm
        alpha = eps * G**2 / (new_V * jnp.log(new_V/G**2)**2)

        new_params = []
        for eta, w_eta in zip(etas, state.params):
            w_norm = utils.tree_l2_norm(w_eta)
            theta = jax.lax.cond(
                w_norm == 0,
                lambda _: utils.negative_tree(updates),
                lambda _: jtu.tree_map(
                    lambda w, g: 2*(w/w_norm)*jnp.log(w_norm/alpha)/eta - g, w_eta, updates),
                operand=None
            )
            theta_norm = utils.tree_l2_norm(theta)
            new_w_eta = jtu.tree_map(
                lambda t: alpha*theta/theta_norm * (jnp.exp(eta/2*jnp.maximum(theta_norm-2*eta*grads_sqnorm, 0))-1), theta)
            new_params.append(new_w_eta)
            # jax.debug.print('eta={x}, theta={t}, w={w}, w/alpha={y}', x=eta, t=theta, w=new_w_eta, y=w_norm/alpha)
        sum_new_params = jtu.tree_map(
            lambda x: jnp.sum(x, axis=0),
            jtu.tree_map(lambda *xs: jnp.stack(xs), *new_params)
        )
        # jax.debug.print('debug: V={v}, alpha={a}', v=new_V, a=alpha)
        return sum_new_params, ParameterFreeMirrorDescentState(V=new_V, params=new_params)
    
    return OnlineLearner(init_fn, update_fn)


# ======================================================================
# Below implements online learner conversion algorithms.
# ======================================================================

class ImperfectHintsState(NamedTuple):
    """imperfect_hints state."""
    sum_lam: chex.Array         # regularization constant
    sum_sigma: chex.Array
    r_square: chex.Array        # sum of square of negative correlation
    last_hint: Updates
    ol_state: OptState


def imperfect_hints(
    online_learner: OnlineLearner,
    mu: float = 1.0,
) -> OnlineLearner:
    raise NotImplementedError
    
    def init_fn(params):
        return ImperfectHintsState(
            sum_lam=jnp.Array([1/mu]),
            sum_square=jnp.zeros([]),
            r_square=jnp.ones([]),
            last_hint=utils.zero_tree(params),
            ol_state=online_learner.init(params),
        )
    
    def update_fn(grads, state, params):
        hint_grad_inner = utils.tree_inner_product(grads, state.last_hint)
        r_square = jax.lax.cond(
            hint_grad_inner < 0, 
            lambda _: state.r_square - hint_grad_inner, 
            lambda _: state.r_square, 
            operand=None
        )
        sigma = jax.abs(hint_grad_inner) * mu / state.r_square
        
        return new_params, ImperfectHintsState()
    
    return OnlineLearner(init_fn, update_fn)