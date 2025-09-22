# Usage:
#   cd /projectnb/aclab/qinziz/trainit/scripts/plot
#   module load python3/3.10.12 cuda/12.2
#   source /projectnb/aclab/qinziz/trainit/env/bin/activate
#   python plot.py


from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from scipy.ndimage import gaussian_filter1d
import json
import wandb
import os


def _gaussian_smooth(y, sigma=5, truncate=3.0, mode="nearest"):
    y = np.asarray(y, dtype=float)
    if sigma is None or sigma <= 0:
        return y

    # NaN-aware normalized convolution:
    mask = ~np.isnan(y)
    y_filled = np.where(mask, y, 0.0).astype(float)

    num = gaussian_filter1d(y_filled, sigma=sigma, mode=mode, truncate=truncate)
    den = gaussian_filter1d(mask.astype(float), sigma=sigma, mode=mode, truncate=truncate)

    # avoid divide-by-zero; where den==0, keep NaN
    with np.errstate(invalid="ignore", divide="ignore"):
        smoothed = num / den
    smoothed[den == 0] = np.nan
    return smoothed


def plot_losses_and_schedules(name_to_data, loss_sigma=5, loss_truncate=3.0, loss_mode="nearest"):
    figsize = (10, 4)
    dpi = 120
    fontsize = 10

    fig, axes = plt.subplots(1, 2, figsize=figsize, dpi=dpi)
    ax_loss, ax_sched = axes

    for name, d in name_to_data.items():
        print(f"Plotting {name}...")

        it = np.asarray(d["iterations"], dtype=float)
        loss = np.asarray(d["loss"], dtype=float)
        sched = np.asarray(d["lr/schedule"], dtype=float)

        # sort by iteration
        order = np.argsort(it)
        it, loss, sched = it[order], loss[order], sched[order]

        # subplot 1: loss (raw + smoothed via SciPy)
        loss_sm = _gaussian_smooth(loss, sigma=loss_sigma, truncate=loss_truncate, mode=loss_mode)
        # ax_loss.plot(it, loss, alpha=0.25, linewidth=1.0, label=f"{name} (raw)")
        ax_loss.plot(it, loss_sm, linewidth=2.0, label=name)

        # subplot 2: schedule (no smoothing)
        ax_sched.plot(it, sched, linewidth=2.0, label=name)

    # ax_loss.set_title("Iteration vs Loss")
    ax_loss.set_xlabel("Iteration", fontsize=fontsize)
    ax_loss.set_ylabel("Loss", fontsize=fontsize)
    ax_loss.grid(True, alpha=0.3)
    ax_loss.legend(ncols=1, fontsize=fontsize)

    # ax_sched.set_title("Iteration vs Learning-Rate Schedule")
    ax_sched.set_xlabel("Iteration", fontsize=fontsize)
    ax_sched.set_ylabel("LR Schedule", fontsize=fontsize)
    ax_sched.grid(True, alpha=0.3)
    ax_sched.legend(fontsize=fontsize)

    fig.tight_layout()
    return fig, (ax_loss, ax_sched)


def load_json(path: str) -> dict:
    print(f"loading json from {path}...")
    with open(os.path.join(path, "checkpoint/data.json"), "r") as f:
        data = json.load(f)
    return data


def load_wandb(path: str) -> dict:
    print(f"loading wandb data from {path}...")
    api = wandb.Api()
    run = api.run(path)
    df = run.history(samples=None) 
    data = {
        key: df.get(key).to_numpy() for key in ["iterations", "loss", "lr/schedule"]
    }
    return data


if __name__ == "__main__":
    runs = {
        "trapezoid": load_wandb("optimizedlearning/test3/d2a77cfe-d24d-4f0a-95e5-61f50aca514f"),
        "cosine": load_wandb("optimizedlearning/test3/097e95e1-a4e2-400c-8fd3-7a5305b5316e"),
        "semi-local": load_json("/projectnb/aclab/qinziz/trainit/scheduler_outputs/2025-09-17/step2k_seg10_lr1e-3_grid20_eps0.48decay_15be1a"),
        "local": load_json("/projectnb/aclab/qinziz/trainit/scheduler_outputs/2025-09-19/step2k_seg10_lr1e-3_grid20_eps0.0_9e290b"),
    }
    fig, _ = plot_losses_and_schedules(runs, loss_sigma=20)
    fig.savefig("results/1.png", dpi=300, bbox_inches="tight")