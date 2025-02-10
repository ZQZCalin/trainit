import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.ticker import FuncFormatter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

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
        xrange: np.ndarray | None = None,
        legends: list[str] | None = None,
        save_path: str | None = None,
        xlabel: str | None = None,
        ylabel: str | None = None,
        title: str | None = None
):
    """Make plots with consistent styles."""
    # Configs
    figsize = (6, 4)
    dpi = 300
    bgcolor = "white"
    small_font = 10
    normal_font = 12
    large_font = 14
    linewidth = 2
    grid = True

    # Main plots.
    plt.figure(figsize=figsize, dpi=dpi, facecolor=bgcolor)
    
    for series in data_series:
        if xrange is not None:
            plt.plot(xrange, series[xrange])
        else:
            plt.plot(series)
    
    # Extra info.
    if legends:
        plt.legend(legends)
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)
    if title:
        plt.title(title)
    if grid:
        plt.grid(True)
    
    # Font size.
    ax = plt.gca()
    ax.tick_params(axis='both', labelsize=small_font)
    ax.xaxis.get_offset_text().set_fontsize(small_font)
    ax.yaxis.get_offset_text().set_fontsize(small_font)
    ax.xaxis.label.set_fontsize(normal_font)
    ax.yaxis.label.set_fontsize(normal_font)
    ax.title.set_fontsize(normal_font)
    ax.set_facecolor(bgcolor)

    # Customize ticks
    ax.set_yticks(np.arange(3, 12))

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()


def zoom_plot(
        *data_series: np.ndarray,
        legends: list[str] | None = None,
        save_path: str | None = None,
        xlabel: str | None = None,
        ylabel: str | None = None,
        title: str | None = None,
        zoom_region: tuple[int, int, float, float] | None = None,  # (xmin, xmax, ymin, ymax)
        zoom_ratio: float = 0.3,
):
    """Make plots with consistent styles and an optional zoomed-in subplot."""
    # Configs
    figsize = (6, 4)
    dpi = 300
    bgcolor = "white"
    small_font = 10
    normal_font = 11
    large_font = 14
    linewidth = 2
    grid = True

    # Main figure and axis
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi, facecolor=bgcolor)

    # Plot each data series
    for series in data_series:
        ax.plot(series, linewidth=linewidth)

    # Move legends above the plot
    if legends:
        ncol = min(len(legends), 4)
        ax.legend(legends, loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=ncol, fontsize=normal_font)

    # Labels and grid
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=normal_font)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=normal_font)
    if title:
        ax.set_title(title, fontsize=normal_font)
    if grid:
        ax.grid(True)

    # Font size adjustments
    ax.tick_params(axis='both', labelsize=small_font)
    ax.xaxis.get_offset_text().set_fontsize(small_font)
    ax.yaxis.get_offset_text().set_fontsize(small_font)
    ax.set_facecolor(bgcolor)

    # Customize ticks
    ax.set_yticks(np.arange(3, 12))

    # Zoomed-in subplot
    if zoom_region:
        xmin, xmax, ymin, ymax = zoom_region

        # Fix: Use a proper bbox_to_anchor tuple for relative positioning
        axins = inset_axes(ax, width="100%", height="100%", 
                           bbox_to_anchor=(0.47, 0.5, 0.5, 0.5), bbox_transform=ax.transAxes)

        # Plot the same data in the zoomed inset
        for series in data_series:
            axins.plot(series, linewidth=linewidth/2)

        # Set zoomed-in limits
        axins.set_xlim(xmin, xmax)
        axins.set_ylim(ymin, ymax)

        # Add grid
        axins.grid(grid)

        # Custom ticks
        if ymax - ymin >= 0.4:
            sep = np.round((ymax - ymin) / 4, 1)
            yticks = np.arange(ymin, ymax, sep)[:5]
            axins.set_yticks(yticks)
        axins.tick_params(axis='both', labelsize=small_font - 2)
        
        # Draw rectangle around zoomed-in region
        # mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="red", lw=1.5)

    # plt.tight_layout()
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
        adamw, muon, mango,
        xrange=np.arange(0,2000),
        legends=["adamw", "muon", "mango"],
        save_path="plots/baselines"
    )

    styled_plot(
        adamw, muon, mango,
        xrange=np.arange(1000,2000),
        legends=["adamw", "muon", "mango"],
        save_path="plots/baselines_zoomed"
    )

    zoom_plot(
        adamw, muon, mango,
        legends=["adamw", "muon", "mango"],
        save_path="plots/baselines_combined",
        zoom_region=(1000, 2000, 2.8, 4.0)
    )


def plot_normalization_embedding():
    """Embedding layer normalization."""
    emb_l2col = get_data("mango_baseline")
    emb_null = get_data("mango_emb_null")
    emb_infcol = get_data("mango_emb_infcol")
    emb_ns = get_data("mango_emb_ns")

    styled_plot(
        emb_null, emb_l2col, emb_infcol, emb_ns,
        xrange=np.arange(0,2000),
        legends=["null", "l2_col", "inf_col", "ns"],
        save_path="plots/embedding"
    )

    styled_plot(
        emb_null, emb_l2col, emb_infcol, emb_ns,
        xrange=np.arange(1000,2000),
        legends=["null", "l2_col", "inf_col", "ns"],
        save_path="plots/embedding_zoomed"
    )

    zoom_plot(
        emb_null, emb_l2col, emb_infcol, emb_ns,
        legends=["null", "l2_col", "inf_col", "ns"],
        save_path="plots/embedding_combined",
        zoom_region=(1000, 2000, 2.9, 3.5)
    )


def plot_normalization_head():
    head_ns = get_data("mango_baseline")
    head_l2col = get_data("mango_head_l2col")
    head_null = get_data("mango_head_null")
    zoom_plot(
        head_null, head_l2col, head_ns,
        legends=["null", "l2_col", "ns"],
        save_path="plots/head",
        zoom_region=(1500, 2000, 3.02, 3.18)
    )


def plot_normalization_attn():
    attn_nosplit = get_data("mango_baseline")
    attn_split = get_data("mango_attn-split")
    zoom_plot(
        attn_nosplit, attn_split,
        legends=["no-split", "split"],
        save_path="plots/attn",
        zoom_region=(1800, 2000, 3.02, 3.08)
    )


def plot_normalization_vecw():
    vecw_null = get_data("mango_vecW_null")
    vecw_l2 = get_data("mango_vecW_l2")
    vecw_inf = get_data("mango_baseline")
    zoom_plot(
        vecw_null, vecw_l2, vecw_inf,
        legends=["null", "l2", "inf"],
        save_path="plots/vecw",
        # zoom_region=(1800, 2000, 3.02, 3.09)
        zoom_region=(1500, 2000, 3.02, 3.18)
    )


def plot_normalization_vecb():
    vecb_null = get_data("mango_bias-null-lr0.01")
    vecb_l2 = get_data("mango_baseline")
    vecb_inf = get_data("mango_bias_inf")
    zoom_plot(
        vecb_null, vecb_l2, vecb_inf,
        legends=["null", "l2", "inf"],
        save_path="plots/vecb",
        zoom_region=(1500, 2000, 3.02, 3.18)
    )


def plot_beta2():
    beta2_null = get_data("mango_beta2_null")
    beta2_9 = get_data("mango_beta2_0.9")
    beta2_95 = get_data("mango_best")
    beta2_99 = get_data("mango_beta2_0.99")
    zoom_plot(
        beta2_null, beta2_9, beta2_95, beta2_99,
        legends=["null", "0.9", "0.95", "0.99"],
        save_path="plots/beta2",
        zoom_region=(1500, 2000, 2.9, 3.06)
    )


def plot_offset():
    offset_null = get_data("mango_offset-null")
    # offset_9 = get_data("mango_offset0.9")
    offset_95 = get_data("mango_offset0.95")
    offset_99 = get_data("mango_best")
    zoom_plot(
        offset_null, offset_95, offset_99,
        legends=["null", "0.95", "0.99"],
        save_path="plots/offset",
        zoom_region=(1500, 2000, 2.905, 3.02)
    )


if __name__ == "__main__":
    # plot_baselines()
    # plot_normalization_embedding()
    # plot_normalization_head()
    # plot_normalization_attn()
    # plot_normalization_vecw()
    # plot_normalization_vecb()
    # plot_beta2()
    plot_offset()