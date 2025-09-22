"""Learning rate schedules that complements optax.schedule."""

import optax
from jaxtyping import Array
import jax.numpy as jnp


def get_current_lr(
    learning_rate: optax.ScalarOrSchedule,
    count: Array,
):
    """Returns the current learning rate."""    
    if callable(learning_rate):
        return learning_rate(count)
    else:
        return learning_rate
    

def linear_decay_schedule(
    init_value: float,
    decay_steps: int,
    end_value: float = 0.0,
) -> optax.Schedule:
    """Linear decay schedule.

    Args:
        init_value: Initial value for the scalar to be annealed.
        peak_value: Peak value for scalar to be annealed at end of warmup.
        end_value: End value of the scalar to be annealed. Defaults to 0.

    Returns:
        A schedule function.
    """
    return optax.linear_schedule(
        init_value=init_value,
        end_value=end_value,
        transition_steps=decay_steps,
    )


def warmup_linear_decay_schedule(
    init_value: float,
    peak_value: float,
    warmup_steps: int,
    decay_steps: int,
    end_value: float = 0.0,
) -> optax.Schedule:
    """Linear warmup followed up linear decay.

    Args:
        init_value: Initial value for the scalar to be annealed.
        peak_value: Peak value for scalar to be annealed at end of warmup.
        warmup_steps: Positive integer, the length of the linear warmup.
        decay_steps: Positive integer, the total length of the schedule. Note that
            this includes the warmup time, so the number of steps during which linear
            decay is applied is ``decay_steps - warmup_steps``.
        end_value: End value of the scalar to be annealed. Defaults to 0.

    Returns:
        A schedule function.
    """

    schedules = [
        optax.linear_schedule(
            init_value=init_value,
            end_value=peak_value,
            transition_steps=warmup_steps,
        ),
        optax.linear_schedule(
            init_value=peak_value,
            end_value=end_value,
            transition_steps=decay_steps-warmup_steps,
        )
    ]
    return optax.join_schedules(schedules, [warmup_steps])


def warmup_const_linear_decay_schedule(
        peak_value: float,
        warmup_steps: int,
        const_steps: int,
        total_steps: int,
        init_value: float = 0.0,
        end_value: float = 0.0,
) -> optax.Schedule:
    """The 3-phase linear decay schedule.

    The first phase is warmup from init_value to peak_value,
    the second phase stays constant at peak_value, and
    the third phase decays from peak_value to end_value.

    Args:
        peak_value: maximum lr.
        warmup_steps: number of warmup steps.
        const_steps: number of constant steps.
        total_steps: total number of steps, including warmup and constant.
        init_value: starting lr.
        end_value: final lr.

    Returns:
        An `optax.Schedule` object
    """
    
    schedules = [
        optax.linear_schedule(
            init_value=init_value,
            end_value=peak_value,
            transition_steps=warmup_steps,
        ),
        optax.linear_schedule(
            init_value=peak_value,
            end_value=end_value,
            transition_steps=total_steps-warmup_steps-const_steps,
            transition_begin=const_steps,
        ),
    ]
    return optax.join_schedules(schedules, [warmup_steps])


def trapezoid_schedule(
        peak_value: float,
        total_steps: int,
        warmup_steps: int = 0,
        decay_steps: int = 0,
) -> optax.Schedule:
    """Implements the trapezoid schedule
    
    https://arxiv.org/pdf/2405.18392

    Args:
        peak_value: maximum lr.
        total_steps: total number of steps, including warmup and decay.
        warmup_steps: number of warmup steps.
        decay_steps: number of decay steps.

    Returns:
        An `optax.Schedule` object
    """

    schedules = [
        optax.linear_schedule(
            init_value=0.0,
            end_value=peak_value,
            transition_steps=warmup_steps,
        ),
        optax.linear_schedule(
            init_value=peak_value,
            end_value=peak_value,
            transition_steps=total_steps - warmup_steps - decay_steps,
        ),
        optax.linear_schedule(
            init_value=peak_value,
            end_value=0.0,
            transition_steps=decay_steps,
        )
    ]
    return optax.join_schedules(schedules, [warmup_steps, total_steps - decay_steps])


def quadratic_schedule(
        peak_value: float,
        total_steps: int,
) -> optax.Schedule:
    """Downwards parabola schedule.

    Args:
        peak_value: highest lr
        total_steps: total number of steps
    """
    assert total_steps > 0, f"total_steps must be > 0, got {total_steps}."
    a = total_steps / 2
    def schedule(count: Array) -> Array:
        t = count.astype(jnp.float32)
        lr = 1 - (t/a - 1)**2
        lr = jnp.clip(lr, min=0.0) * peak_value
        return lr.astype(jnp.float32)
    return schedule


def semi_local_schedule(
        peak_value: float,
        total_steps: int,
) -> optax.Schedule:
    """Scaling of semi-local search optimal schedule.

    Checkpoint:
    /projectnb/aclab/qinziz/trainit/scheduler_outputs/2025-09-17/step2k_seg10_lr1e-3_grid20_eps0.48decay_15be1a

    Args:
        peak_value: highest lr
        total_steps: total number of steps
    """
    checkpoints = [
        (0, 0.0),
        (1, 0.1),
        (2, 0.2),
        (3, 0.4),
        (4, 0.8),
        (5, 1.0),
        (6, 1.0),
        (7, 6/7),
        (8, 4/7),
        (9, 2/7),
        (10, 0.0),
    ]
    schedules = []
    for i, (_, value) in enumerate(checkpoints[:-1]):
        _, next_value = checkpoints[i+1]
        schedules.append(
            optax.linear_schedule(
                init_value=value,
                end_value=next_value,
                transition_steps=1,
            )
        )
    boundaries = [step for step, _ in checkpoints]
    boundaries = boundaries[1:-1]
    base_schedule = optax.join_schedules(schedules, boundaries)

    def schedule(count: Array) -> Array:
        lr = base_schedule(10*count/total_steps) * peak_value
        return lr
    return schedule


if __name__ == "__main__":
    # testing
    schedule = semi_local_schedule(peak_value=2.0, total_steps=30)
    for t in range(-3, 23):
        print(t, schedule(jnp.array(t)))