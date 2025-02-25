# Modified from Jiacheng You's code:
# 
# https://gist.github.com/YouJiacheng/393c90cbdc23b09d5688815ba382288b

from functools import partial

import jax
import jax.numpy as jnp
import optax

import matplotlib.pyplot as plt


def poly(x: jnp.ndarray, w: jnp.ndarray):
    assert w.shape == (3,)
    w = w.astype(jnp.float32)
    return w[0] * x + w[1] * x**3 + w[2] * x**5


def poly_chain(x: jnp.ndarray, w_seq: jnp.ndarray):
    y = [x]
    for w in w_seq:
        y.append(poly(y[-1], w))
    return y


def min_of_polys(x: jnp.ndarray, w_seq: jnp.ndarray):
    y = jnp.full_like(x, jnp.inf)
    for w in w_seq:
        y = jnp.minimum(y, poly(x, w))
    return y


@partial(jax.jit, static_argnums=(2, 3, 4, 5))
def optimize_w(
    w_seq: jnp.ndarray,
    lr: float,
    k: float,
    b: float,
    n: int,
    debug: bool = False,
):
    def loss(w_seq: jnp.ndarray):
        xs = (jnp.arange(2048) + 1) / 2048
        # Compute the poly chain outputs: the list returned is [x, f_1(x), f_2(x), ..., f_6(x)]
        *zs, ys = jax.vmap(poly_chain, in_axes=(0, None))(xs, w_seq)
        
        # Compute statistics on the final output f(x)=ys:
        y_max = jnp.amax(ys)
        y_min = jnp.amin(jnp.where(xs > 1/128, ys, jnp.inf))
        diff_ratio = (y_max - y_min) / jnp.clip(y_max, a_min=1e-3)
        
        # For controlling the slope, use a sample of points:
        slope_xs = (jnp.arange(320) + 1) / 256
        min_ps = jax.vmap(min_of_polys, in_axes=(0, None))(slope_xs, w_seq)
        min_slope = jnp.amin(min_ps / slope_xs)
        
        # For controlling the transitions between layers:
        z_max_seq = [jnp.amax(z) for z in zs]
        max_next_excess = sum(
            jnp.clip(poly(z + 1/16, w) - z, a_min=0)
            for z, w in zip(z_max_seq, w_seq)
        )
        
        # Original objectives:
        # (i) Encourage a high start: we want f_0(x)=x to be a good fraction of the maximum.
        obj_0 = ys[0] / y_max
        # (ii) Ensure y_max is close to 1.
        obj_1 = y_max
        # (iii) Penalize large variation in the output.
        obj_2 = jnp.log2(diff_ratio)
        # (iv) Ensure the chain has a minimum slope.
        obj_3 = min_slope
        # (v) Control overshooting between layers.
        obj_4 = max_next_excess
        
        # New approximation objective: mean squared error between f(x) and target g(x)=exp(-kx)
        target = (1-b) * (1-jnp.exp(-k * xs))
        obj_1_max = (1-b) * (1-jnp.exp(-k))
        mse = jnp.mean((ys - target) ** 2)
        
        # Combine them with weights.
        # loss = mse \
        #       - 4.0 * obj_0 \
        #       + 16.0 * jnp.square(obj_1 - b) \
        #       + 2.0 * jnp.clip(obj_2, a_min=-10) \
        #       - 4.0 * jnp.clip(obj_3, a_max=1/2) \
        #       + 64.0 * obj_4
        loss = mse \
              + 1.0 * jnp.square(obj_1 - obj_1_max)
        objectives = (mse, obj_0, obj_1, obj_2, obj_3, obj_4)
        return loss, objectives

    loss_and_grad_fn = jax.value_and_grad(loss, argnums=0, has_aux=True)
    optimizer = optax.chain(
        optax.adam(learning_rate=lr),
        optax.clip_by_global_norm(1.0),
    )
    opt_state = optimizer.init(w_seq)

    def body_fn(carry: tuple[jnp.ndarray, optax.OptState], _):
        w_seq, opt_state = carry
        (_, objectives), grad = loss_and_grad_fn(w_seq)
        updates, opt_state = optimizer.update(grad, opt_state)
        w_seq = optax.apply_updates(w_seq, updates)
        return (w_seq, opt_state), objectives

    (w_seq, _), objectives = jax.lax.scan(body_fn, (w_seq, opt_state), length=n)
    return w_seq, objectives


def main():
    BASE = 128
    N = 4
    w_seq = jnp.array([[3.5, -6.04444444444, 2.84444444444]] * N)
    k, b = (20, 0.01)
    n = 10000
    verbose = True
    for i in range(5):
        w_seq, objectives = optimize_w(w_seq, k=k, b=b, lr=2e-3, n=n)
        if verbose:
            print(w_seq.astype(jnp.bfloat16) * BASE)
            print(i, [obj[-1].item() for obj in objectives])
    for i in range(5):
        w_seq, objectives = optimize_w(w_seq, k=k, b=b, lr=1e-3, n=n)
        if verbose:
            print(w_seq.astype(jnp.bfloat16) * BASE)
            print(i, [obj[-1].item() for obj in objectives])
    for i in range(5):
        w_seq, objectives = optimize_w(w_seq, k=k, b=b, lr=5e-4, n=n)
        if verbose:
            print(w_seq.astype(jnp.bfloat16) * BASE)
            print(i, [obj[-1].item() for obj in objectives])
    for i in range(20):
        w_seq, objectives = optimize_w(w_seq, k=k, b=b, lr=1e-4, n=n)
        if verbose:
            print(w_seq.astype(jnp.bfloat16) * BASE)
            print(i, [obj[-1].item() for obj in objectives])

    # xs = jnp.linspace(0, 1, 256)
    # plt.plot(xs, poly_chain(xs, w_seq)[-1])
    # plt.show()


if __name__ == "__main__":
    main()
