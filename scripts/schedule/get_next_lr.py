"""A python script to get (lr1, lr2_candidates) for the next segment.

You can customize your learning rate choosing logic here.
"""

import argparse
import wandb
import pandas as pd
import numpy as np


# Type of smoothing. you can implement your own way of smoothing.
SMOOTHING = "EMA"

# Smoothing parameters.
EMA_WINDOW_SIZE = 10


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--segment",
        type=int,
    )
    parser.add_argument(
        "--path",
        type=str,
    )
    args = parser.parse_args()

    window_size = 10
    alpha = 1 / window_size

    api = wandb.Api()

    entity = "optimizedlearning"
    project = "pile_baseline"
    run_id = "599oovpq"

    run = api.run(f"{entity}/{project}/{run_id}")

    # Fetch losses.
    # NOTE: do not use run.history() as it downsamples the data.
    history = run.scan_history(keys=["loss"])
    losses = [row["loss"] for row in history]
    
    last_loss = smoothing(losses)[-1]
    print(last_loss)