import matplotlib.pyplot as plt
import numpy as np

def main():
    figsize = (10, 4)
    dpi = 120
    fontsize = 10

    optimal_color = "#D62728"

    fig, axes = plt.subplots(1, 2, figsize=figsize, dpi=dpi)
    ax_loss, ax_lr = axes

    eps = 1.0
    loss = 10.0
    lr = 0.0

    seg_len = 100

    grid = [0/1, 1/2, 2/3, 1, 3/2, 2/1]

    data = [
        {
            # 1st segment: absolute candidates for both loss & lr
            "lr":   [0.5, 0.75, 1, 1.25, 1.5],
            "loss": [-1.0, -2.0, -2.5, -1.5, -0.5],
            "best": 2,
            "range": set(),
        },
        {
            # 2nd segment: loss deltas (decreases); lr not specified -> carry over
            "loss": [-1.8, -2.6, -2.2, -1.2, -0.6, -0.2],
            "best": 2,
            "range": {0, 1, 2},
        },
        {
            # 3rd segment: another set of loss deltas
            "loss": [-2.6, -2.2, -1.2, -0.8, -0.6, -0.2],
            "best": 1,
            "range": {0, 1},
        },
    ]

    for seg, d in enumerate(data):
        t0, t1 = seg * seg_len, (seg + 1) * seg_len

        loss_cands = loss + np.asarray(d["loss"], dtype=float)
        if seg == 0:
            lr_cands = np.asarray(d["lr"], dtype=float)
        else:
            lr_cands = lr * np.asarray(grid, dtype=float)
        best_idx = d["best"]

        for idx, (_loss, _lr) in enumerate(zip(loss_cands, lr_cands)):
            if idx == best_idx:
                c, ls, linewidth = optimal_color, "-", 2
            else:
                c, ls, linewidth = None, "--", None
                if idx not in d["range"]:
                    c = "grey"
            ax_loss.plot([t0, t1], [loss, _loss], c=c, ls=ls, linewidth=linewidth)
            ax_lr.plot([t0, t1], [lr, _lr], c=c, ls=ls, linewidth=linewidth)

            # add epsilon range
            bar = float(np.nanmin(loss_cands))
            delta = 0.1 if seg == 0 else eps
            ax_loss.fill_between([t0, t1], bar, bar+delta, color="pink", alpha=0.15, linewidth=0, zorder=0)
        loss = loss_cands[best_idx]
        lr = lr_cands[best_idx]

    # ax_loss.set_title("Iteration vs Loss")
    ax_loss.set_xlabel("Iteration", fontsize=fontsize)
    ax_loss.set_ylabel("Loss", fontsize=fontsize)
    ax_loss.grid(True, alpha=0.3)
    # ax_loss.legend(ncols=1, fontsize=fontsize)

    # ax_lr.set_title("Iteration vs Learning-Rate Schedule")
    ax_lr.set_xlabel("Iteration", fontsize=fontsize)
    ax_lr.set_ylabel("LR Schedule", fontsize=fontsize)
    # ax_lr.set_yscale("log")
    ax_lr.grid(True, alpha=0.3)
    # ax_lr.legend(fontsize=fontsize)

    fig.tight_layout()
    fig.savefig(f"results/illustration.png", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()