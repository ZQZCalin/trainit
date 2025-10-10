# Usage:
#   cd /projectnb/aclab/qinziz/trainit/scripts/plot
#   module load python3/3.10.12 cuda/12.2
#   source /projectnb/aclab/qinziz/trainit/env/bin/activate
#   python smooth_loss.py


from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from scipy.ndimage import gaussian_filter1d
import json
import wandb
from omegaconf import OmegaConf
import os
from typing import NamedTuple


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


def load_json(path: str) -> dict:
    print(f"loading json from {path}...")
    with open(os.path.join(path, "checkpoint/data.json"), "r") as f:
        data = json.load(f)
    return data


def load_wandb(path: str) -> dict:
    print(f"loading wandb data from {path}...")
    # For some reason, there are different folder structures...
    conf_path = os.path.join(path, "config.yaml")
    if not os.path.exists(conf_path):
        conf_path = os.path.join(path, "checkpoint/config.yaml")
    conf = OmegaConf.load(conf_path)

    wandb_path = f"optimizedlearning/{conf.logging.wandb_project}/{conf.logging.wandb_runid}"
    api = wandb.Api()
    run = api.run(wandb_path)
    df = run.history(samples=None) 
    data = {
        key: df.get(key).to_numpy() for key in ["iterations", "loss", "lr/schedule"]
    }
    return data


class SmoothConfig(NamedTuple):
    func: callable
    path: str
    steps: int = 2000
    sigma: float = 10


def check_path(runs: dict[str, SmoothConfig]) -> None:
    for run, cfg in runs.items():
        if not os.path.isdir(cfg.path):
            raise ValueError(f"run {run} and path {cfg.path} does not exist.")


def main(runs: dict[str, SmoothConfig], save_path: str) -> None:
    data = {}
    for run, cfg in runs.items():
        d = cfg.func(cfg.path)
        it = np.asarray(d["iterations"], dtype=float)
        loss = np.asarray(d["loss"], dtype=float)

        # sort by iteration
        order = np.argsort(it)
        it, loss = it[order], loss[order]

        smoothed_loss = _gaussian_smooth(loss, sigma=cfg.sigma)
        # wandb sometimes miss iterations, so we pick the largest iter before max_steps
        data[run] = smoothed_loss[it<=cfg.steps][-1].item()
    # save locally
    with open(save_path, "w") as f:
        json.dump(data, f, indent=4)
    print("Saved smoothed losses locally.")


checkpoint_runs = {
    # cosine, tune lr (warmup = 200)
    "baseline_step2k_cosine_lr3.33e-2": SmoothConfig(func=load_wandb, path="/projectnb/aclab/alee12/trainit3/trainit_project/trainit/checkpoint/c2kbaseline3.33e-2"),
    "baseline_step2k_cosine_lr1e-2":    SmoothConfig(func=load_wandb, path="/projectnb/aclab/alee12/trainit3/trainit_project/trainit/checkpoint/c2kbaseline1e-2"),
    "baseline_step2k_cosine_lr3.33e-3": SmoothConfig(func=load_wandb, path="/projectnb/aclab/alee12/trainit3/trainit_project/trainit/checkpoint/c2kbaseline3.33e-3"),
    "baseline_step2k_cosine_lr1e-3":    SmoothConfig(func=load_wandb, path="/projectnb/aclab/alee12/trainit3/trainit_project/trainit/checkpoint/c2kbaseline1e-3"),
    "baseline_step2k_cosine_lr3.33e-4": SmoothConfig(func=load_wandb, path="/projectnb/aclab/alee12/trainit3/trainit_project/trainit/checkpoint/c2kbaseline3.33e-4"),
    "baseline_step2k_cosine_lr1e-4":    SmoothConfig(func=load_wandb, path="/projectnb/aclab/alee12/trainit3/trainit_project/trainit/checkpoint/c2kbaseline1e-4"),
    "baseline_step2k_cosine_lr3.33e-5": SmoothConfig(func=load_wandb, path="/projectnb/aclab/alee12/trainit3/trainit_project/trainit/checkpoint/c2kbaseline3.33e-5"),
    "baseline_step2k_cosine_lr1e-5":    SmoothConfig(func=load_wandb, path="/projectnb/aclab/alee12/trainit3/trainit_project/trainit/checkpoint/c2kbaseline1e-5"),
    
    # cosine, tune warmup (lr = 1e-3)
    "baseline_step2k_cosine_lr1e-3_w0":    SmoothConfig(func=load_wandb, path="/projectnb/aclab/alee12/trainit3/trainit_project/trainit/checkpoint/c2kbaseline1e-3_w0"),
    "baseline_step2k_cosine_lr1e-3_w100":  SmoothConfig(func=load_wandb, path="/projectnb/aclab/alee12/trainit3/trainit_project/trainit/checkpoint/c2kbaseline1e-3_w100"),
    "baseline_step2k_cosine_lr1e-3_w500":  SmoothConfig(func=load_wandb, path="/projectnb/aclab/alee12/trainit3/trainit_project/trainit/checkpoint/c2kbaseline1e-3_w500"),
    "baseline_step2k_cosine_lr1e-3_w1000": SmoothConfig(func=load_wandb, path="/projectnb/aclab/alee12/trainit3/trainit_project/trainit/checkpoint/c2kbaseline1e-3_w1000"),
    
    # trapezoid, tune lr (warmup decay = 200)
    "baseline_step2k_trapezoid_lr3.33e-2": SmoothConfig(func=load_wandb, path="/projectnb/aclab/alee12/trainit3/trainit_project/trainit/checkpoint/2kbaseline3.33e-2"),
    "baseline_step2k_trapezoid_lr1e-2":    SmoothConfig(func=load_wandb, path="/projectnb/aclab/alee12/trainit3/trainit_project/trainit/checkpoint/2kbaseline1e-2"),
    "baseline_step2k_trapezoid_lr3.33e-3": SmoothConfig(func=load_wandb, path="/projectnb/aclab/alee12/trainit3/trainit_project/trainit/checkpoint/2kbaseline3.33e-3"),
    "baseline_step2k_trapezoid_lr1e-3":    SmoothConfig(func=load_wandb, path="/projectnb/aclab/alee12/trainit3/trainit_project/trainit/checkpoint/2kbaseline1e-3"),
    "baseline_step2k_trapezoid_lr3.33e-4": SmoothConfig(func=load_wandb, path="/projectnb/aclab/alee12/trainit3/trainit_project/trainit/checkpoint/2kbaseline3.33e-4"),
    "baseline_step2k_trapezoid_lr1e-4":    SmoothConfig(func=load_wandb, path="/projectnb/aclab/alee12/trainit3/trainit_project/trainit/checkpoint/2kbaseline1e-4"),
    "baseline_step2k_trapezoid_lr3.33e-5": SmoothConfig(func=load_wandb, path="/projectnb/aclab/alee12/trainit3/trainit_project/trainit/checkpoint/2kbaseline3.33e-5"),
    "baseline_step2k_trapezoid_lr1e-5":    SmoothConfig(func=load_wandb, path="/projectnb/aclab/alee12/trainit3/trainit_project/trainit/checkpoint/2kbaseline1e-5"),

    # trapezoid, tune warmup (lr = 1e-3, decay = 200)
    "baseline_step2k_trapezoid_lr1e-3_w0":    SmoothConfig(func=load_wandb, path="/projectnb/aclab/alee12/trainit3/trainit_project/trainit/checkpoint/2kbaseline1e-3_w0"),
    "baseline_step2k_trapezoid_lr1e-3_w100":  SmoothConfig(func=load_wandb, path="/projectnb/aclab/alee12/trainit3/trainit_project/trainit/checkpoint/2kbaseline1e-3_w100"),
    "baseline_step2k_trapezoid_lr1e-3_w500":  SmoothConfig(func=load_wandb, path="/projectnb/aclab/alee12/trainit3/trainit_project/trainit/checkpoint/2kbaseline1e-3_w500"),
    "baseline_step2k_trapezoid_lr1e-3_w1000": SmoothConfig(func=load_wandb, path="/projectnb/aclab/alee12/trainit3/trainit_project/trainit/checkpoint/2kbaseline1e-3_w1000"),

    # trapezoid, tune decay (lr = 1e-3, warmup = 200)
    "baseline_step2k_trapezoid_decay_0":    SmoothConfig(func=load_wandb, path="/projectnb/aclab/qinziz/trainit/scheduler_outputs/2025-09-22/baseline_step2k_trapezoid_decay/decay_0"),
    "baseline_step2k_trapezoid_decay_100":  SmoothConfig(func=load_wandb, path="/projectnb/aclab/qinziz/trainit/scheduler_outputs/2025-09-22/baseline_step2k_trapezoid_decay/decay_100"),
    "baseline_step2k_trapezoid_decay_500":  SmoothConfig(func=load_wandb, path="/projectnb/aclab/qinziz/trainit/scheduler_outputs/2025-09-22/baseline_step2k_trapezoid_decay/decay_500"),
    "baseline_step2k_trapezoid_decay_1000": SmoothConfig(func=load_wandb, path="/projectnb/aclab/qinziz/trainit/scheduler_outputs/2025-09-22/baseline_step2k_trapezoid_decay/decay_1000"),

    # linear, tune warmup (lr = 1e-3)
    "baseline_step2k_linear_lr1e-3_w0_d2000":    SmoothConfig(func=load_wandb, path="/projectnb/aclab/alee12/trainit3/trainit_project/trainit/checkpoint/c2kbaseline1e-3_w0d2000"),
    "baseline_step2k_linear_lr1e-3_w100_d1900":  SmoothConfig(func=load_wandb, path="/projectnb/aclab/alee12/trainit3/trainit_project/trainit/checkpoint/c2kbaseline1e-3_w100d1900"),
    "baseline_step2k_linear_lr1e-3_w200_d1800":  SmoothConfig(func=load_wandb, path="/projectnb/aclab/alee12/trainit3/trainit_project/trainit/checkpoint/c2kbaseline1e-3_w200d1800"),
    "baseline_step2k_linear_lr1e-3_w500_d1500":  SmoothConfig(func=load_wandb, path="/projectnb/aclab/alee12/trainit3/trainit_project/trainit/checkpoint/c2kbaseline1e-3_w500d1500"),
    "baseline_step2k_linear_lr1e-3_w1000_d1000": SmoothConfig(func=load_wandb, path="/projectnb/aclab/alee12/trainit3/trainit_project/trainit/checkpoint/c2kbaseline1e-3_w1000d1000"),

    # quadratic, tune lr
    "baseline_step2k_quadratic_lr3.33e-2": SmoothConfig(func=load_wandb, path="/projectnb/aclab/qinziz/trainit/scheduler_outputs/2025-09-22/baseline_step2k_quadratic/lr_3.33e-2"),
    "baseline_step2k_quadratic_lr1e-2":    SmoothConfig(func=load_wandb, path="/projectnb/aclab/qinziz/trainit/scheduler_outputs/2025-09-22/baseline_step2k_quadratic/lr_1e-2"),
    "baseline_step2k_quadratic_lr3.33e-3": SmoothConfig(func=load_wandb, path="/projectnb/aclab/qinziz/trainit/scheduler_outputs/2025-09-23/baseline_step2k_quadratic/lr_3.33e-3"),
    "baseline_step2k_quadratic_lr1e-3":    SmoothConfig(func=load_wandb, path="/projectnb/aclab/qinziz/trainit/scheduler_outputs/2025-09-22/baseline_step2k_quadratic/lr_1e-3"),
    "baseline_step2k_quadratic_lr3.33e-4": SmoothConfig(func=load_wandb, path="/projectnb/aclab/qinziz/trainit/scheduler_outputs/2025-09-22/baseline_step2k_quadratic/lr_3.33e-4"),
    "baseline_step2k_quadratic_lr1e-4":    SmoothConfig(func=load_wandb, path="/projectnb/aclab/qinziz/trainit/scheduler_outputs/2025-09-23/baseline_step2k_quadratic/lr_1e-4"),

    # triangle (symmetric_linear), tune lr
    "baseline_step2k_triangle_lr3.33e-2": SmoothConfig(func=load_wandb, path="/projectnb/aclab/qinziz/trainit/scheduler_outputs/2025-09-22/baseline_step2k_symmetric_linear/lr_3.33e-2"),
    "baseline_step2k_triangle_lr1e-2":    SmoothConfig(func=load_wandb, path="/projectnb/aclab/qinziz/trainit/scheduler_outputs/2025-09-22/baseline_step2k_symmetric_linear/lr_1e-2"),
    "baseline_step2k_triangle_lr3.33e-3": SmoothConfig(func=load_wandb, path="/projectnb/aclab/qinziz/trainit/scheduler_outputs/2025-09-22/baseline_step2k_symmetric_linear/lr_3.33e-3"),
    "baseline_step2k_triangle_lr1e-3":    SmoothConfig(func=load_wandb, path="/projectnb/aclab/qinziz/trainit/scheduler_outputs/2025-09-22/baseline_step2k_symmetric_linear/lr_1e-3"),
    "baseline_step2k_triangle_lr3.33e-4": SmoothConfig(func=load_wandb, path="/projectnb/aclab/qinziz/trainit/scheduler_outputs/2025-09-22/baseline_step2k_symmetric_linear/lr_3.33e-4"),
    "baseline_step2k_triangle_lr1e-4":    SmoothConfig(func=load_wandb, path="/projectnb/aclab/qinziz/trainit/scheduler_outputs/2025-09-22/baseline_step2k_symmetric_linear/lr_1e-4"),

    # 10k steps: trapezoid, tune lr
    "baseline_step10k_trapezoid_lr1e-3":    SmoothConfig(func=load_wandb, steps=10000, path="/projectnb/aclab/alee12/trainit3/trainit_project/trainit/scheduler_outputs/2025-09-16/iter10k_baseline_trap_warmup0.1_decay0.1/checkpoint/lr_1e-3"),
    "baseline_step10k_trapezoid_lr3.33e-4": SmoothConfig(func=load_wandb, steps=10000, path="/projectnb/aclab/alee12/trainit3/trainit_project/trainit/scheduler_outputs/2025-09-19/iter10k_baseline_trap_warmup0.1_decay0.1/checkpoint/lr_3.33e-4"),
    "baseline_step10k_trapezoid_lr1e-4":    SmoothConfig(func=load_wandb, steps=10000, path="/projectnb/aclab/alee12/trainit3/trainit_project/trainit/scheduler_outputs/2025-09-16/iter10k_baseline_trap_warmup0.1_decay0.1/checkpoint/lr_1e-4"),
    "baseline_step10k_trapezoid_lr3.33e-5": SmoothConfig(func=load_wandb, steps=10000, path="/projectnb/aclab/alee12/trainit3/trainit_project/trainit/scheduler_outputs/2025-09-16/iter10k_baseline_trap_warmup0.1_decay0.1/checkpoint/lr_3.33e-5"),
    "baseline_step10k_trapezoid_lr1e-5":    SmoothConfig(func=load_wandb, steps=10000, path="/projectnb/aclab/alee12/trainit3/trainit_project/trainit/scheduler_outputs/2025-09-16/iter10k_baseline_trap_warmup0.1_decay0.1/checkpoint/lr_1e-5"),

    # 10k steps: stretched semi-local
    "baseline_step10k_semi_local_lr3.33e-4": SmoothConfig(func=load_wandb, steps=10000, path="/projectnb/aclab/qinziz/trainit/scheduler_outputs/2025-09-22/baseline_step10k_semi_local/lr_3.33e-4"),
    "baseline_step10k_semi_local_lr1e-4":    SmoothConfig(func=load_wandb, steps=10000, path="/projectnb/aclab/qinziz/trainit/scheduler_outputs/2025-09-22/baseline_step10k_semi_local/lr_1e-4"),
}

new_runs = {
    "semi_local_step10k_eps0.12": SmoothConfig(func=load_json, steps=10000, path="/projectnb/aclab/qinziz/trainit/scripts/plot/local_data/semi-local_step10k_eps0.12"),
    "semi_local_step10k_eps0.24": SmoothConfig(func=load_json, steps=10000, path="/projectnb/aclab/qinziz/trainit/scripts/plot/local_data/semi-local_step10k_eps0.24"),
}


if __name__ == "__main__":
    # runs = checkpoint_runs
    # path = "checkpoint_smooth.json"
    runs = new_runs
    path = "checkpoint_smooth_new.json"

    check_path(runs)
    main(runs, save_path=path)