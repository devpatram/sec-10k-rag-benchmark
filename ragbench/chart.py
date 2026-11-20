"""The comparison chart: one small panel per metric, so 0-1 retrieval scores and 1-5 judge scores never share an axis."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

SURFACE, INK, INK_2, GRID, BAR = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1", "#2a78d6"


def comparison_chart(summary: pd.DataFrame, k: int, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = [("MRR", (0, 1)), (f"Recall@{k}", (0, 1))]
    if "Correctness (1-5)" in summary:
        metrics += [("Correctness (1-5)", (1, 5)), ("Faithfulness (1-5)", (1, 5))]
    names = list(summary.index)[::-1]  # first configuration on top
    fig, axes = plt.subplots(1, len(metrics), figsize=(3.2 * len(metrics) + 1.6, 0.55 * len(names) + 1.4),
                             sharey=True, facecolor=SURFACE)
    for ax, (metric, (lo, hi)) in zip(axes, metrics):
        values = summary.loc[names, metric].astype(float)
        ax.set_facecolor(SURFACE)
        ax.barh(names, values - lo, left=lo, height=0.5, color=BAR)
        for y, v in enumerate(values):
            ax.text(v + (hi - lo) * 0.02, y, f"{v:.2f}", va="center", fontsize=9, color=INK)
        ax.set_xlim(lo, hi + (hi - lo) * 0.18)
        ax.set_title(metric, loc="left", fontsize=10, color=INK)
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(colors=INK_2, labelsize=9, length=0)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
    fig.suptitle("Retrieval and answer quality by configuration", x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)
