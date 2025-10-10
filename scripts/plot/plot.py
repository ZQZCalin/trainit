# Usage:
#   cd /projectnb/aclab/qinziz/trainit/scripts/plot
#   module load python3/3.10.12 cuda/12.2
#   source /projectnb/aclab/qinziz/trainit/env/bin/activate
#   python plot.py


from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd
import numpy as np
from scipy.ndimage import gaussian_filter1d
import json
import wandb
import os


DEFAULT_COLOR = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#a562e3",
    "#c76dac", "#17becf", "#c47841", "#F0E442", "#5a128e",
    "#835757", "#377473", "#bcbd22", "#001F3F", "#56B4E9", 
]

def load_json(*path: str) -> dict:
    print(f"loading json from {path}...")
    data = {}
    for p in path:
        with open(os.path.join(p, "checkpoint/data.json"), "r") as f:
            data.update(json.load(f))
    return data


def load_wandb(*path: str) -> dict:
    print(f"loading wandb data from {path}...")
    api = wandb.Api()
    keys=["iterations", "loss", "lr/schedule"]
    data = { key: [] for key in keys }

    for p in path:
        run = api.run(p)
        rows = run.scan_history(keys=keys)
        for r in rows:
            for key in keys:
                data[key].append(r.get(key))
    return data


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


def filled_loss_diff(it1, loss1, it2, loss2, fill_before_first=np.nan):
    # for each iters1[k], find index of the rightmost iters2 <= iters1[k]
    idx = np.searchsorted(it2, it1, side="right") - 1  # -1 means "no past value"

    loss2_ffill = np.empty_like(loss1, dtype=float)
    mask = idx >= 0
    loss2_ffill[mask]  = loss2[idx[mask]]
    loss2_ffill[~mask] = fill_before_first  # e.g., np.nan or a constant like loss2[0]

    return loss1 - loss2_ffill, loss2_ffill  # return diff and the aligned loss2 for inspection


def get_sorted_data(data):
    it = np.asarray(data["iterations"], dtype=float)
    loss = np.asarray(data["loss"], dtype=float)
    sched = np.asarray(data["lr/schedule"], dtype=float)
    # sort by iteration
    # order = np.argsort(it)
    # it, loss, sched = it[order], loss[order], sched[order]
    return it, loss, sched


def adjust_color(color, alpha=0.0):
    """
    alpha = fraction toward target (white/black/any color).
    alpha=0.2 toward white ≈ 20% lighter; toward black ≈ 20% darker.
    """
    if alpha > 0:
        target = "#ffffff"
    else:
        target = "#000000"
    alpha = abs(alpha)
    c = np.array(mcolors.to_rgb(color))
    t = np.array(mcolors.to_rgb(target))
    out = (1 - alpha) * c + alpha * t
    return mcolors.to_hex(out)


def plot_losses_and_schedules(
        name_to_data, 
        loss_sigma=5, 
        loss_truncate=3.0, 
        loss_mode="nearest",
        baseline=None,
        log_scale=False,
    ):
    figsize = (10, 4)
    dpi = 120
    fontsize = 10

    fig, axes = plt.subplots(1, 2, figsize=figsize, dpi=dpi)
    ax_loss, ax_sched = axes

    # Compute baseline if necessary
    baseline_it, baseline_loss = None, None
    if baseline is not None:
        _, func, *path = name_to_data[baseline]
        baseline_it, baseline_loss, baseline_sched = get_sorted_data(func(*path))

    color_index = 5

    for name, (_cfg, func, *path) in name_to_data.items():
        print(f"Plotting {name}...")

        # Copy from baseline
        if name == baseline:
            it, loss, sched = baseline_it, baseline_loss, baseline_sched
        else:
            it, loss, sched = get_sorted_data(func(*path))
        
        # Read plot configs
        color, alpha, ls = _cfg or (None, None, None)
        alpha = alpha or 0.0
        ls = ls or "-"

        # Rescale colors
        if color is not None:
            c = DEFAULT_COLOR[color]
        else:
            c = DEFAULT_COLOR[color_index]
            color_index += 1
        c = adjust_color(c, alpha)

        # subplot 1: loss (raw + smoothed via SciPy)
        if baseline:
            loss_diff, _ = filled_loss_diff(it, loss, baseline_it, baseline_loss)
            loss_diff_sm = _gaussian_smooth(loss_diff, sigma=loss_sigma, truncate=loss_truncate, mode=loss_mode)
            ax_loss.plot(it, loss_diff_sm, linewidth=2.0, c=c, ls=ls, label=name)
        else:
            loss_sm = _gaussian_smooth(loss, sigma=loss_sigma, truncate=loss_truncate, mode=loss_mode)
            # ax_loss.plot(it, loss, alpha=0.25, linewidth=1.0, label=f"{name} (raw)")
            ax_loss.plot(it, loss_sm, linewidth=2.0, c=c, ls=ls, label=name)

        # subplot 2: schedule (no smoothing)
        ax_sched.plot(it, sched, linewidth=2.0, c=c, ls=ls, label=name)

    # ax_loss.set_title("Iteration vs Loss")
    ax_loss.set_xlabel("Iteration", fontsize=fontsize)
    ax_loss.set_ylabel("Loss Diff" if baseline else "Loss", fontsize=fontsize)
    ax_loss.grid(True, alpha=0.3)
    ax_loss.legend(ncols=1, fontsize=fontsize)

    # ax_sched.set_title("Iteration vs Learning-Rate Schedule")
    ax_sched.set_xlabel("Iteration", fontsize=fontsize)
    ax_sched.set_ylabel("LR Schedule", fontsize=fontsize)
    if log_scale:
        ax_sched.set_yscale("log")
    ax_sched.grid(True, alpha=0.3)
    ax_sched.legend(fontsize=fontsize)

    fig.tight_layout()
    return fig, (ax_loss, ax_sched)


if __name__ == "__main__":
    BLUE = 0            # WSD
    RED = 3             # semi-local decay
    ORANGE = 1          # semi-local const
    GREEN = 2           # local
    PURPLE = 4          # other
    LIGHT = +0.4
    DARK = -0.4
    _ = None

    runs = {
        "intro": {
            "WSD": ((BLUE,_,_),  load_wandb, "optimizedlearning/test3/d2a77cfe-d24d-4f0a-95e5-61f50aca514f"),
            "cosine":    (_,     load_wandb, "optimizedlearning/test3/097e95e1-a4e2-400c-8fd3-7a5305b5316e"),
            "semi-local":((RED,_,_),   load_json,  "/projectnb/aclab/qinziz/trainit/scheduler_outputs/2025-09-17/step2k_seg10_lr1e-3_grid20_eps0.48decay_15be1a"),
            "local":     ((GREEN,_,_), load_json,  "/projectnb/aclab/qinziz/trainit/scheduler_outputs/2025-09-19/step2k_seg10_lr1e-3_grid20_eps0.0_9e290b"),
        },
        "seg10": {
            "eps=0.0": ( _,    load_json, "/projectnb/aclab/qinziz/trainit/scheduler_outputs/2025-09-19/step2k_seg10_lr1e-3_grid20_eps0.0_9e290b"),
            "eps=0.06":( _,    load_json, "/projectnb/aclab/qinziz/trainit/scheduler_outputs/2025-09-14/step2k_seg10_lr1e-3_grid20_eps0.06_1df4fd"),
            "eps=0.12":( _,    load_json, "/projectnb/aclab/qinziz/trainit/scheduler_outputs/2025-09-15/step2k_seg10_lr1e-3_grid20_eps0.12_b66742"),
            "eps=0.24":( _,    load_json, "/projectnb/aclab/qinziz/trainit/scheduler_outputs/2025-09-16/step2k_seg10_lr1e-3_grid20_eps0.24_96bb10"),
            "eps=0.48":( _,    load_json, "/projectnb/aclab/qinziz/trainit/scheduler_outputs/2025-09-21/step2k_seg10_lr1e-3_grid20_eps0.48_f3f724"),
        },
        "seg4": {
            "eps=0.0": ( _,    load_json, "/projectnb/aclab/qinziz/trainit/scheduler_outputs/2025-09-12/step2k_seg4_lr1e-3_grid20_eps0.0_70e394"),
            "eps=0.06":( _,    load_json, "/projectnb/aclab/qinziz/trainit/scheduler_outputs/2025-09-11/step2k_seg4_lr1e-3_grid20_eps0.06_09278e"),
            "eps=0.12":( _,    load_json, "/projectnb/aclab/qinziz/trainit/scheduler_outputs/2025-09-12/step2k_seg4_lr1e-3_grid20_eps0.12_b147e3"),
            "eps=0.24":( _,    load_json, "/projectnb/aclab/qinziz/trainit/scheduler_outputs/2025-09-13/step2k_seg4_lr1e-3_grid20_eps0.24_f01f5d"),
        },
        "seg10_decay": {
            "eps=0.0": ( _,    load_json, "/projectnb/aclab/qinziz/trainit/scheduler_outputs/2025-09-19/step2k_seg10_lr1e-3_grid20_eps0.0_9e290b"),
            "eps=0.24":( _,    load_json, "/projectnb/aclab/qinziz/trainit/scheduler_outputs/2025-09-16/step2k_seg10_lr1e-3_grid20_eps0.24decay_d0f502"),
            "eps=0.48":( _,    load_json, "/projectnb/aclab/qinziz/trainit/scheduler_outputs/2025-09-17/step2k_seg10_lr1e-3_grid20_eps0.48decay_15be1a"),
            "eps=0.96":( _,    load_json, "/projectnb/aclab/qinziz/trainit/scheduler_outputs/2025-09-18/step2k_seg10_lr1e-3_grid20_eps0.96decay_71b436"),
        },
        "seg20": {
            "eps=0.0":   ( _,    load_json, "/projectnb/aclab/alee12/trainit3/trainit_project/trainit/scheduler_outputs/2025-09-22/seg20_2k_eps0_417bb2"),
            "eps=0.015": ( _,    load_json, "/projectnb/aclab/alee12/trainit3/trainit_project/trainit/scheduler_outputs/2025-09-21/seg20_2k_eps0.015_7804b1"),
            "eps=0.03":  ( _,    load_json, "/projectnb/aclab/alee12/trainit3/trainit_project/trainit/scheduler_outputs/2025-09-20/seg20_2k_eps0.03_20e645"),
            "eps=0.06":  ( _,    load_json, "/projectnb/aclab/alee12/trainit3/trainit_project/trainit/scheduler_outputs/2025-09-15/seg20_2k_eps0.06_40fc39"),
            "eps=0.12":  ( _,    load_json, "/projectnb/aclab/alee12/trainit3/trainit_project/trainit/scheduler_outputs/2025-09-16/seg20_2k_eps0.12_93da8e"),
            "eps=0.24":  ( _,    load_json, "/projectnb/aclab/alee12/trainit3/trainit_project/trainit/scheduler_outputs/2025-09-19/seg20_2k_eps0.24_aaa453"),
            "eps=0.48":  ( _,    load_json, "/projectnb/aclab/alee12/trainit3/trainit_project/trainit/scheduler_outputs/2025-09-19/seg20_2k_eps0.48_11daa5"),
        },
        "local_suboptimal": {
            "WSD":      ((BLUE,_,_),  load_wandb, "optimizedlearning/test3/d2a77cfe-d24d-4f0a-95e5-61f50aca514f"),
            # "cosine":      (_,    load_wandb, "optimizedlearning/test3/097e95e1-a4e2-400c-8fd3-7a5305b5316e"),
            "seg=4":    ((GREEN, LIGHT, "-."), load_json,  "/projectnb/aclab/qinziz/trainit/scheduler_outputs/2025-09-12/step2k_seg4_lr1e-3_grid20_eps0.0_70e394"),
            "seg=10":   ((GREEN,_,_), load_json,  "/projectnb/aclab/qinziz/trainit/scheduler_outputs/2025-09-19/step2k_seg10_lr1e-3_grid20_eps0.0_9e290b"),
            "seg=20":   ((GREEN, DARK, "--"), load_json,  "/projectnb/aclab/alee12/trainit3/trainit_project/trainit/scheduler_outputs/2025-09-22/seg20_2k_eps0_417bb2"),
        },
        "seg10_tune_eps": {
            "WSD": ((BLUE,_,_), load_wandb, "optimizedlearning/test3/d2a77cfe-d24d-4f0a-95e5-61f50aca514f"),
            "eps=0.0":   ( _,   load_json,  "/projectnb/aclab/qinziz/trainit/scheduler_outputs/2025-09-19/step2k_seg10_lr1e-3_grid20_eps0.0_9e290b"),
            "eps=0.06":  ( _,   load_json,  "/projectnb/aclab/qinziz/trainit/scheduler_outputs/2025-09-14/step2k_seg10_lr1e-3_grid20_eps0.06_1df4fd"),
            "eps=0.12":  ( _,   load_json,  "/projectnb/aclab/qinziz/trainit/scheduler_outputs/2025-09-15/step2k_seg10_lr1e-3_grid20_eps0.12_b66742"),
            "eps=0.24":  ( _,   load_json,  "/projectnb/aclab/qinziz/trainit/scheduler_outputs/2025-09-16/step2k_seg10_lr1e-3_grid20_eps0.24_96bb10"),
            "eps=0.48":  ( _,   load_json,  "/projectnb/aclab/qinziz/trainit/scheduler_outputs/2025-09-21/step2k_seg10_lr1e-3_grid20_eps0.48_f3f724"),
        },
        "seg10_eps_decay": {
            "WSD":         ((BLUE,_,_),   load_wandb, "optimizedlearning/test3/d2a77cfe-d24d-4f0a-95e5-61f50aca514f"),
            "eps=0.12 (const)":  ((ORANGE,_,_), load_json,  "/projectnb/aclab/qinziz/trainit/scheduler_outputs/2025-09-15/step2k_seg10_lr1e-3_grid20_eps0.12_b66742"),
            "eps=0.48 (decay)":  ((RED,_,_),    load_json,  "/projectnb/aclab/qinziz/trainit/scheduler_outputs/2025-09-17/step2k_seg10_lr1e-3_grid20_eps0.48decay_15be1a"),
        },
        "semi_local_segments": {
            "WSD": ((BLUE,_,_),   load_wandb, "optimizedlearning/test3/d2a77cfe-d24d-4f0a-95e5-61f50aca514f"),
            "seg=4":    ((ORANGE, LIGHT, "-."), load_json,  "/projectnb/aclab/qinziz/trainit/scheduler_outputs/2025-09-13/step2k_seg4_lr1e-3_grid20_eps0.24_f01f5d"),
            "seg=10":   ((ORANGE,_,_), load_json,  "/projectnb/aclab/qinziz/trainit/scheduler_outputs/2025-09-15/step2k_seg10_lr1e-3_grid20_eps0.12_b66742"),
            "seg=20":   ((ORANGE, DARK, "--"), load_json,  "/projectnb/aclab/alee12/trainit3/trainit_project/trainit/scheduler_outputs/2025-09-19/seg20_2k_eps0.48_11daa5"),
        },
        "inspired_schedules": {
            "WSD":  ((BLUE,_,_),   load_wandb, "optimizedlearning/test3/d2a77cfe-d24d-4f0a-95e5-61f50aca514f"),
            "semi-local": ((RED,_,_),    load_json,  "/projectnb/aclab/qinziz/trainit/scheduler_outputs/2025-09-17/step2k_seg10_lr1e-3_grid20_eps0.48decay_15be1a"),
            "quadratic":  ((GREEN,_,_), load_wandb, "optimizedlearning/greedy_lr_schedule/fb34fbd5-4fbc-4cd4-b30b-5e9e259c65b5"),
            "triangle":   ((PURPLE,_,_),      load_wandb, "optimizedlearning/greedy_lr_schedule/2c0eacdb-1e41-4ff8-912c-0cca0c81a54c"),
        },
        "semi_local_steps10k": {
            "WSD":              ((BLUE,_,_), load_json,  "/projectnb/aclab/qinziz/trainit/scripts/plot/local_data/baseline_step10k_trapezoid_lr3.33e-4"),
            "semi-local (2ksteps)":   ((RED,_,_),  load_wandb, "optimizedlearning/greedy_lr_schedule/9e378f19-099a-4c85-a441-d22f02052ac7"),
            "semi-local (eps=0.12)":  ( _,   load_json,  "/projectnb/aclab/qinziz/trainit/scripts/plot/local_data/semi-local_step10k_eps0.12"),
            "semi-local (eps=0.24)":  ( _,   load_json,  "/projectnb/aclab/qinziz/trainit/scripts/plot/local_data/semi-local_step10k_eps0.24"),
        },
        "seg20_tune_eps": {
            "WSD":         ((BLUE,_,_),   load_wandb, "optimizedlearning/test3/d2a77cfe-d24d-4f0a-95e5-61f50aca514f"),
            "eps=0.0":   ( _,    load_json, "/projectnb/aclab/alee12/trainit3/trainit_project/trainit/scheduler_outputs/2025-09-22/seg20_2k_eps0_417bb2"),
            "eps=0.015": ( _,    load_json, "/projectnb/aclab/alee12/trainit3/trainit_project/trainit/scheduler_outputs/2025-09-21/seg20_2k_eps0.015_7804b1"),
            "eps=0.03":  ( _,    load_json, "/projectnb/aclab/alee12/trainit3/trainit_project/trainit/scheduler_outputs/2025-09-20/seg20_2k_eps0.03_20e645"),
            "eps=0.06":  ( _,    load_json, "/projectnb/aclab/alee12/trainit3/trainit_project/trainit/scheduler_outputs/2025-09-15/seg20_2k_eps0.06_40fc39"),
            "eps=0.12":  ( _,    load_json, "/projectnb/aclab/alee12/trainit3/trainit_project/trainit/scheduler_outputs/2025-09-16/seg20_2k_eps0.12_93da8e"),
            "eps=0.24":  ( _,    load_json, "/projectnb/aclab/alee12/trainit3/trainit_project/trainit/scheduler_outputs/2025-09-19/seg20_2k_eps0.24_aaa453"),
            "eps=0.48":  ( _,    load_json, "/projectnb/aclab/alee12/trainit3/trainit_project/trainit/scheduler_outputs/2025-09-19/seg20_2k_eps0.48_11daa5"),
        },
    }

    folder = "new_results"

    # for name in [
    #     "intro", "seg10", "seg4", "seg10_decay", "seg20"
    # ]:
    #     baseline = None
    #     save_path = f"{name}.png"
    #     fig, _ = plot_losses_and_schedules(runs[name], loss_sigma=20, baseline=baseline)
    #     fig.savefig(f"{folder}/{save_path}", dpi=300, bbox_inches="tight")

    for name in [
        "local_suboptimal", "seg10_tune_eps", "seg10_eps_decay", "seg20_tune_eps",
        "semi_local_segments", "inspired_schedules", "semi_local_steps10k",
    ]:
        baseline = "WSD"
        save_path = f"{name}_baseline.png"
        fig, _ = plot_losses_and_schedules(runs[name], loss_sigma=20, baseline=baseline)
        fig.savefig(f"{folder}/{save_path}", dpi=300, bbox_inches="tight")

    # for name in [
    #     "local_suboptimal", "seg10_tune_eps", "seg10_eps_decay",
    #     "semi_local_segments", "inspired_schedules", "semi_local_steps10k",
    # ]:
    #     baseline = "WSD"
    #     save_path = f"{name}_baseline_logscale.png"
    #     fig, _ = plot_losses_and_schedules(runs[name], loss_sigma=20, baseline=baseline, log_scale=True)
    #     fig.savefig(f"{folder}/{save_path}", dpi=300, bbox_inches="tight")