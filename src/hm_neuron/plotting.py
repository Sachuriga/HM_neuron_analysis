"""Figure style, saving, and maze drawing."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402

from . import maze as mz  # noqa: E402

RAT_COLORS = {"Rat5": "#1f77b4", "Rat6": "#d62728"}
KIND_COLORS = {"node": "#e69f00", "edge": "#0072b2"}
MM = 1 / 25.4


def style():
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7, "legend.fontsize": 6,
        "xtick.labelsize": 6, "ytick.labelsize": 6, "axes.linewidth": 0.6,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6, "lines.linewidth": 1.0,
        "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42,
        "ps.fonttype": 42, "svg.fonttype": "none", "figure.dpi": 150,
    })


def figure(w_mm=180, h_mm=100, **kw):
    style()
    return plt.subplots(figsize=(w_mm * MM, h_mm * MM), constrained_layout=True, **kw)


def save(fig, out_dir, name, cfg):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in cfg.fig_format:
        p = out / f"{name}.{ext}"
        fig.savefig(p, dpi=cfg.dpi)
        paths.append(p)
    plt.close(fig)
    return paths


def draw_maze(ax, lm: mz.LinearMaze, values=None, cmap="viridis", vmin=None, vmax=None,
              lw=2.5, node_size=12, goal=None, colorbar_label=None):
    """Draw the maze; `values` (len n_bins) colours bins, NaN = grey."""
    b = lm.bins
    xy = b[["x", "y"]].to_numpy()
    cm = plt.get_cmap(cmap)
    if values is not None:
        v = np.asarray(values, float)
        vmin = np.nanmin(v) if vmin is None else vmin
        vmax = np.nanmax(v) if vmax is None else vmax
        norm = plt.Normalize(vmin, vmax)
    segs, cols = [], []
    for (a, c), idx in lm.edge_bins.items():
        pa, pc = np.array(lm.G.nodes[a]["pos"]), np.array(lm.G.nodes[c]["pos"])
        L = np.linalg.norm(pc - pa)
        for i in idx:
            s0, s1 = b.at[i, "s"] - b.at[i, "length"] / 2, b.at[i, "s"] + b.at[i, "length"] / 2
            segs.append([pa + (pc - pa) * s0 / L, pa + (pc - pa) * s1 / L])
            cols.append("0.85" if values is None or not np.isfinite(v[i]) else cm(norm(v[i])))
        # connectors from node edge to first bin
        segs += [[pa, pa + (pc - pa) * lm.r / L], [pc, pc + (pa - pc) * lm.r / L]]
        cols += ["0.85", "0.85"]
    ax.add_collection(LineCollection(segs, colors=cols, linewidths=lw, capstyle="butt"))
    nb = np.where(lm.is_node())[0]
    nc = ["0.7" if values is None or not np.isfinite(v[i]) else cm(norm(v[i])) for i in nb]
    ax.scatter(xy[nb, 0], xy[nb, 1], s=node_size, c=nc, edgecolors="none", zorder=3)
    if goal is not None:
        g = lm.G.nodes[int(goal)]["pos"]
        ax.scatter(*g, s=node_size * 4, facecolors="none", edgecolors="k", linewidths=0.8, zorder=4)
    ax.set_aspect("equal")
    ax.set_xlim(0.5, 9.0)
    ax.set_ylim(0.2, 4.6)
    ax.axis("off")
    if values is not None and colorbar_label:
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cm)
        plt.colorbar(sm, ax=ax, shrink=0.6, label=colorbar_label)
    return ax


def rat_legend(ax, rats):
    from matplotlib.lines import Line2D
    ax.legend([Line2D([], [], color=RAT_COLORS.get(r, "k"), marker="o", ls="") for r in rats],
              rats, frameon=False)
