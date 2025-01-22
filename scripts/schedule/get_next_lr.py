"""A python script to get (lr1, lr2_candidates) for the next segment.

You can customize your learning rate choosing logic here.
"""

import argparse
import json
import wandb
import pandas as pd
import numpy as np
from typing import Any


# Type of smoothing. you can implement your own way of smoothing.
SMOOTHING = "EMA"
EMA_WINDOW_SIZE = 10

# Next lr methods.
NEXT_LR1 = "eps_greedy"
EPS_GREEDY_VAL = 0.01

NEXT_LR2 = "log"
LOG_GRID_MULTI = 2
LOG_GRID_SIZE = 2       # additional lrs on each side

# Other global variables
WANDB_ENTITY = "optimizedlearning"


def EMA(losses: list) -> list:
    """Applies EMA smoothing."""
    alpha = 1 / EMA_WINDOW_SIZE
    return pd.Series(losses).ewm(alpha=alpha).mean().to_list()


# Customize your own smoothing if needed.
def customized_smoothing(loss: list) -> list:
    raise NotImplementedError


def smoothing(losses: list) -> list:
    """Wraps all smoothing mechanisms."""
    if SMOOTHING == "EMA":
        return EMA(losses)
    # Add your customized smoothing below.
    raise ValueError(f"unsupport smoothing = '{SMOOTHING}'.")


def get_run_info(run: Any) -> tuple[float, float]:
    """Extracts (lr, last_loss) from a run with specific id."""
    # Fetch associated lr2.
    lr2 = run.config["optimizer"]["lr_config"]["lr2"]

    # Fetch last loss after smoothing
    history = run.scan_history(keys=["loss"])
    losses = [row["loss"] for row in history]
    last_loss = smoothing(losses)[-1]

    return lr2, last_loss


def greedy_lr1(arr: np.ndarray) -> float:
    """Returns lr with lowest loss value."""
    i = np.argmin(arr[:, 1])
    return arr[i, 0]


def eps_greedy_lr1(arr: np.ndarray) -> float:
    """Returns largest lr such that loss <= loss_min + eps."""
    loss_min = np.min(arr[:, 1])
    arr_filtered = arr[arr[:, 1] <= loss_min + EPS_GREEDY_VAL]
    return np.max(arr_filtered[:, 0])


# Customize your own lr mechanism if needed.
def customized_lr1(arr: np.ndarray) -> float:
    raise NotImplementedError


def log_lr2(val: float) -> list:
    """list of lr2 with log-grid search, including 0."""
    res = [val, 0.0]
    lr = val
    for _ in range(LOG_GRID_SIZE):
        lr *= LOG_GRID_MULTI
        res.append(lr)
    lr = val
    for _ in range(LOG_GRID_SIZE):
        lr /= LOG_GRID_MULTI
        res.append(lr)
    return sorted(res)


def get_next_lrs(arr: np.ndarray) -> tuple[float, list]:
    """Wraps all lr methods."""
    if NEXT_LR1 == "greedy":
        lr1 = greedy_lr1(arr)
    elif NEXT_LR1 == "eps_greedy":
        lr1 = eps_greedy_lr1(arr)
    # Add your customized methods below.
    else:
        raise ValueError(f"unsupport lr1 mechanism = '{NEXT_LR1}'.")
    if NEXT_LR2 == "log":
        lr2_candidates = log_lr2(lr1)
    # Add your customized methods below.
    else:
        raise ValueError(f"unsupport lr2 mechanism = '{NEXT_LR2}'.")
    return lr1, lr2_candidates


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project",
        type=str,
    )
    parser.add_argument(
        "--job_ids",
        nargs="+",
        type=int,
    )
    args = parser.parse_args()

    # Fetch losses using WandB API.
    api = wandb.Api()

    entity = WANDB_ENTITY
    project = args.project

    arr = []
    for run_id in args.job_ids:
        run = api.run(f"{entity}/{project}/{run_id}")
        arr.append(get_run_info(run))

    # Customized method to decide lrs in the next segment.
    lr1, lr2_candidates = get_next_lrs(np.array(arr))

    # Return data as a JSON string
    result = {"lr1": lr1, "lr2_candidates": lr2_candidates}
    print(json.dumps(result))



if __name__ == "__main__":
    main()
    # arr = np.array([
    #     [1,2],
    #     [2,1],
    #     [3,4]
    # ])
    # print(eps_greedy_lr1(arr))