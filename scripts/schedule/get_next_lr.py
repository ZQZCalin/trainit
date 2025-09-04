"""A python script to get (lr1, lr2_candidates) for the next segment.

You can customize your learning rate choosing logic here.
"""

import argparse
import json
import wandb
from wandb.errors import CommError
import logging
import pandas as pd
import numpy as np
from typing import Any
import os
from pathlib import Path


# =========================================================
# >>> CONFIGS OF LR MECHANISMS
# =========================================================

# ---------------------------------------------------------
# Type of smoothing. you can implement your own way of smoothing.
SMOOTHING_LIST = [
    "EMA",
]
SMOOTHING = "EMA"
assert SMOOTHING in SMOOTHING_LIST

# >> EMA
EMA_WINDOW_SIZE = 10


# ---------------------------------------------------------
# Default lrs (first segment).
DEFAULT_LR1 = 0.0                                           # we always fix the initial lr1 to 0

DEFAULT_LR2_DICT = {
    "log_grid": [1e0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5],    # default log grid
    "baseline": [1e-3],                                 # hard code first segment to match baseline
    "baseline_better": [2e-3],                          # hard code baseline, but better
    "test": [0.1, 0.01],                                # for testing
}
DEFAULT_LR2_KEY = "baseline_better"                         # specify lr2 for the first segment
assert DEFAULT_LR2_KEY in DEFAULT_LR2_DICT
DEFAULT_LR2 = DEFAULT_LR2_DICT[DEFAULT_LR2_KEY]


# ---------------------------------------------------------
# Next lr1 methods.
NEXT_LR1_LIST = [
    "greedy",
    "eps_greedy",
]
NEXT_LR1 = "eps_greedy"
assert NEXT_LR1 in NEXT_LR1_LIST

# >> greedy mechanism
# ...   # it takes no hyper-parameter, so it's empty here

# >> epsilon-greedy mechanism for lr1
EPS_GREEDY_VAL = 0.240                                      # CHANGE THIS
EPS_GREEDY_ABSOLUTE = True                                  # CHANGE THIS; if true, use absolute eps, otherwise use relative eps
EPS_GREEDY_DECAY = False                                     # CHANGE THIS; if true, adds a linear decay to eps.

# >> potentially other mechanism
# ...


# ---------------------------------------------------------
# Next lr2 methods.
NEXT_LR2_LIST = [
    "log",
    "linear",
]
NEXT_LR2 = "linear"
assert NEXT_LR2 in NEXT_LR2_LIST

# >> logarithmic grid for lr2
LOG_GRID_MULTI = 2                                          # size of logarithmic grids (multiplicative)
LOG_GRID_SIZE = 2                                           # additional lrs on each side

# >> linear grid for lr2
LINEAR_GRID_LOWER_SIZE = 10                                 # CHANGE THIS; should be equal to num_segs
LINEAR_GRID_UPPER_COEF = [1, 1.25, 1.5, 2]                  # CHANGE THIS if needed
# LINEAR_GRID_LOWER_SIZE = 2      # testing
# LINEAR_GRID_UPPER_COEF = [1, 2] # testing


# ---------------------------------------------------------
# Other global variables
# >> Wandb team/organization name.
WANDB_ENTITY = "optimizedlearning"

# >> local json name
DATA_FNAME = "data.json"

# =========================================================
# >>> SMOOTHING METHODS
# =========================================================

def EMA(losses: list) -> list:
    """Applies EMA smoothing."""
    alpha = 1 / EMA_WINDOW_SIZE
    return pd.Series(losses).ewm(alpha=alpha).mean().to_list()


# Customize your own smoothing if needed.
def customized_smoothing(loss: list) -> list:
    raise NotImplementedError


# =========================================================
# >>> DEFAULT LR SETTUP
# =========================================================

def get_default_lr1() -> float:
    """Returns default value of lr1."""
    return DEFAULT_LR1


def get_default_lr2() -> list:
    """Returns default value of lr2."""
    return DEFAULT_LR2


# =========================================================
# >>> LR1 NEXT SEGMENT
# =========================================================

def greedy_lr1(losses: np.ndarray) -> int:
    """Returns the run index corresponding to the lowest last-iterate loss value.
    
    Args:
        losses: [k,] array of last losses of k runs.
    """
    return np.argmin(losses)


def eps_greedy_lr1(losses: np.ndarray, lrs: np.ndarray, eps: float=0.0, use_abs: bool=True) -> int:
    """Returns the run index corresponding to the 
    largest lr such that:
    - loss <= loss_min + eps for absolute eps;
    - loss <= loss_min * (1+eps) for relative eps.
    
    Args:
        losses: [k,] array of last losses.
        lrs: [k,] array of lrs.
        eps: value of epsilon
        abs: whether to use absolute or relative eps for threshold.
    """
    loss_min = np.min(losses)
    if use_abs:
        threshold = loss_min + eps
    else:
        threshold = loss_min * (1 + eps)
        # threshold = loss_min + EPS_GREEDY_VAL * abs(initial_loss - loss_min)
    # 03/31: fixed an issue where eps-greedy is not correctly returning the index
    # of the best run; this only affects runs in v0.0.3.
    ## lrs_filtered = lrs[losses[:, -1] <= threshold]    # wrong implementation
    lrs_filtered = np.where(losses <= threshold, lrs, -np.inf)
    return np.argmax(lrs_filtered)


# Customize your own lr mechanism if needed.
def customized_lr1(arr: np.ndarray) -> float:
    raise NotImplementedError


# =========================================================
# >>> LR2 NET SEGMENT
# =========================================================

def loggrid_lr2(val: float) -> list:
    """list of lr2 with log-grid search, including 0."""
    # Special case: if val == 0, returns default setting
    if val == 0:
        # Edge case: return a log grid.
        return [0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0]
    
    # Standard case.
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


def linear_grid_lr2(val: float) -> list:
    """linear grid of form i/(i+1)."""
    if val == 0:
        # Edge case: return a log grid.
        return [0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0]
    
    coefs = [i/(i+1) for i in range(LINEAR_GRID_LOWER_SIZE)]
    coefs += LINEAR_GRID_UPPER_COEF
    return sorted([val * coef for coef in coefs])


# =========================================================
# >>> WRAPPER FUNCTIONS
# =========================================================

def smoothing(losses: list) -> list:
    """Wraps all smoothing mechanisms."""
    if SMOOTHING == "EMA":
        return EMA(losses)
    # Add your customized smoothing below.
    raise ValueError(f"unsupport smoothing = '{SMOOTHING}'.")


def get_run_info(run: Any) -> tuple[float, list]:
    """Extracts (lr, smoothed_losses) from a run with specific id."""
    # Fetch associated lr2.
    lr2 = run.config["optimizer"]["lr_config"]["lr2"]

    # Fetch last loss after smoothing
    history = run.scan_history(keys=["loss"])
    loss = [row["loss"] for row in history]
    return lr2, smoothing(loss)


def get_best_run(candidates: list, seg: int, num_segs: int, use_greedy: bool = False) -> Any:
    """Given a list of valid runs, return the optimal run.

    Args:
        candidates: list of wandb runs.
        use_greedy: defaults to False; if True, use greedy methods.

    Returns:
        optimal run in the last segment.
    """
    if len(candidates) == 0:
        raise RuntimeError("Candiates cannot be an empty list.")
    
    # Get losses and lrs.
    last_losses = np.zeros(len(candidates))
    lrs = np.zeros(len(candidates))
    for i, run in enumerate(candidates):
        lr2, loss = get_run_info(run)
        last_losses[i] = loss[-1]
        lrs[i] = lr2
    
    # Wrap methods.
    if use_greedy or NEXT_LR1 == "greedy":
        idx = greedy_lr1(last_losses)
    elif NEXT_LR1 == "eps_greedy":
        eps = EPS_GREEDY_VAL
        if EPS_GREEDY_DECAY:
            eps *= (num_segs - seg) / (num_segs - 1)
        idx = eps_greedy_lr1(last_losses, lrs, eps, EPS_GREEDY_ABSOLUTE)
    # Add your customized methods below.
    else:
        raise ValueError(f"unsupport lr1 mechanism = '{NEXT_LR1}'.")
    return candidates[idx]


def get_next_lrs(run: Any) -> tuple[float, list]:
    """Wrap lr2_candidates methods."""
    # Get lr1.
    lr1 = run.config["optimizer"]["lr_config"]["lr2"]
    # Get lr2.
    if NEXT_LR2 == "log":
        lr2_candidates = loggrid_lr2(lr1)
    if NEXT_LR2 == "linear":
        lr2_candidates = linear_grid_lr2(lr1)
    # Add your customized methods below.
    else:
        raise ValueError(f"unsupport lr2 mechanism = '{NEXT_LR2}'.")
    return lr1, lr2_candidates


def update_local_data(run: Any) -> None:
    """Store data in this segment locally."""
    # Fetch data from the new run.
    keys = ["loss", "accuracy", "iterations", "lr/schedule"]
    history = run.scan_history(keys=keys)
    new_data = {key: [row[key] for row in history] for key in keys}

    # Fetch local data.
    ckpt_path = run.config["checkpoint"]["save_path"]
    fname = os.path.join(Path(ckpt_path).parent.parent, DATA_FNAME)
    if os.path.exists(fname):
        with open(fname, 'r') as f:
            data = json.load(f)
    else:
        # Initialize an empty structure
        data = {key: [] for key in keys}

    # Update and store new data.
    for key in keys:
        data[key] += new_data[key]
    with open(fname, 'w') as f:
        json.dump(data, f, indent=2)  # indent=2 for readability


def bash_format(lr1, lr2_candidates):
    """Format wrapper."""
    result = {"lr1": lr1, "lr2_candidates": lr2_candidates}
    return json.dumps(result)   # Return data as a JSON string


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=str)
    parser.add_argument("--job_ids", nargs="*", type=int)
    parser.add_argument("--seg", type=int)          # 1-index, except for the initial call where seg=0
    parser.add_argument("--num_segs", type=int)
    args = parser.parse_args()

    is_first = args.seg == 0
    is_last  = args.seg == args.num_segs

    if is_first:
        return bash_format(get_default_lr1(), get_default_lr2())

    # Fetch losses using WandB API.
    api = wandb.Api()

    entity = WANDB_ENTITY
    project = args.project

    candidates = []
    for run_id in args.job_ids:
        # Added an error catcher for any failed runs
        try:
            run = api.run(f"{entity}/{project}/{run_id}")
            # Add a safe-check: check if ckpt_path contains any .ckpt file
            ckpt_path = run.config["checkpoint"]["save_path"]
            if os.path.isdir(ckpt_path) and any(filename.endswith(".ckpt") for filename in os.listdir(ckpt_path)):
                candidates.append(run)
        except CommError as e:
            logging.info(f"- Update: failed to fetch run {run_id}.")
            logging.error(f"Failed to fetch run {run_id}:\n{e}")

    # Customized method to decide lrs in the next segment.
    best_run = get_best_run(candidates, seg=args.seg, num_segs=args.num_segs, use_greedy=is_last)
    lr1, lr2_candidates = get_next_lrs(best_run)

    # Store current best loss and lrs locally.
    update_local_data(best_run)

    return bash_format(lr1, lr2_candidates)


if __name__ == "__main__":
    result = main()
    print(result)