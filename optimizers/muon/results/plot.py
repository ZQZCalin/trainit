import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.ticker import FuncFormatter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes,mark_inset

import numpy as np
from scipy.ndimage import gaussian_filter1d
import pandas as pd
import os


data = pd.read_csv("data/loss.csv")


def get_data(
        name: str,
        smoothing: str = "Gaussian",
        smoothing_const: int = 20,
) -> np.ndarray:
    """Fetches data and applies smoothing if necessary."""
    res = data[f"{name} - loss"].values
    if smoothing == "Gaussian":
        res = gaussian_filter1d(res, sigma=smoothing_const)
    return res


def styled_plot(
        *data_series: np.ndarray,
        legends: list[str] | None = None,
        save_path: str | None = None,
        xlabel: str | None = None,
        ylabel: str | None = None,
        title: str | None = None
):
    """Make plots with consistent styles."""
    # Configs
    figsize = (6, 4)
    fontsize = 12
    linewidth = 2
    grid = True
    
    plt.figure(figsize=figsize)
    
    for series in data_series:
        if isinstance(series, tuple) and len(series) == 2:
            plt.plot(series[0], series[1], linewidth=linewidth)
        else:
            plt.plot(series)
    
    if legends:
        plt.legend(legends, fontsize=fontsize)
    if xlabel:
        plt.xlabel(xlabel, fontsize=fontsize)
    if ylabel:
        plt.ylabel(ylabel, fontsize=fontsize)
    if title:
        plt.title(title, fontsize=fontsize)
    if grid:
        plt.grid(True)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()


def plot_baselines():
    """Compares different optimizers with optimal baselines."""
    adamw = get_data("adamw_lr:1e-3")
    muon = get_data("muon_lr:1e-2 (nesterov)")
    muon_laprop = get_data("muon_laprop_same_lr")
    muon_adamw = get_data("muon_adamw")
    mango = get_data("mango_best")

    styled_plot(
        adamw, muon, muon_laprop, muon_adamw, mango,
        legends=["adamw", "muon", "muon-laprop", "muon-adamw", "mango"],
        save_path="plots/baselines"
    )


def plot_normalization_embedding():
    """Embedding layer normalization."""
    baseline = get_data("")


if __name__ == "__main__":
    
    plot_baselines()
