"""0. Worked examples of the place-field detection.

For one mid-training session per rat, three place cells are taken at the 10th,
50th and 90th percentile of field count. For each cell the figure shows every
step of the detection: spikes on the path, the linearised rate map, the bins
above threshold grouped into fields, and the same cell after a circular spike
shift (surrogate), which is the detector's noise floor.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap

from .. import fields as fl, maze as mz, plotting as P
from . import common

NAME = "field_examples"
TITLE = "How place fields are detected: worked examples"
DEFAULT_QUERY = "repeat > 0"  # repeat 0 = habituation, not training
QUANTILES = (0.10, 0.50, 0.90)


def _surrogate_rate(s, uid, R, cfg, lm, mask, rng):
    """Rate map of `uid` after one circular spike shift within running time."""
    ms = np.where(mask)[0]
    n = len(ms)
    si = fl.spike_samples(s, s.spikes[uid])
    si = si[mask[si]]
    min_shift = int(cfg.fields.shuffle_min_shift_s / s.dt)
    lo, hi = min(min_shift, n // 2), max(n - min_shift, n // 2 + 1)
    b = s.bin[ms[(np.searchsorted(ms, si) + rng.integers(lo, hi)) % n]]
    cnt = np.bincount(b, minlength=lm.n_bins)[None].astype(float)
    K = fl.smoothing_kernel(lm, cfg.fields.smooth_sigma_m)
    return fl.smooth_rate(cnt, R["occ"], K, cfg.fields.min_occ_s)[0][0]


def _mid_session(g, cfg):
    """First usable session of a rat, searching outwards from the middle of training."""
    order = np.argsort(np.abs(np.arange(len(g)) - len(g) // 2), kind="stable")
    for item in common.iter_loaded(g.iloc[order], cfg, verbose=False):
        return item
    return None


def run(sessions, cfg):
    lm = mz.maze_from_cfg(cfg)
    f = cfg.fields
    ex, cells, frows = [], [], []
    for rat, g in sessions.groupby("rat", sort=True):
        item = _mid_session(g, cfg)
        if item is None:
            continue
        row, s, R = item
        print(f"  {rat} {row['date']} (day {row['day']})", flush=True)
        U = R["units"]
        pc = U[U["is_place"]].sort_values(["n_fields", "unit"])
        if pc.empty:
            continue
        mask = fl.running_mask(s, cfg)
        rng = np.random.default_rng(f.seed + 2)
        step = max(1, int(mask.sum() // 6000))
        path = np.c_[s.x[mask], s.y[mask]][::step]
        for q in QUANTILES:
            u = pc.iloc[int(round(q * (len(pc) - 1)))]
            uid = u["unit"]
            k = int(np.where(R["uids"] == uid)[0][0])
            rate = R["rate"][k]
            fd = fl.detect_fields(rate, R["valid"], lm, cfg, goal=s.goal)
            srate = _surrogate_rate(s, uid, R, cfg, lm, mask, rng)
            sfd = fl.detect_fields(srate, R["valid"], lm, cfg)
            si = fl.spike_samples(s, s.spikes[uid])
            si = si[mask[si]]
            peak = float(np.nanmax(np.where(R["valid"], rate, np.nan)))
            above = R["valid"] & (np.nan_to_num(rate) >= f.field_frac * peak)
            in_field = np.zeros(lm.n_bins, bool)
            for d in fd:
                in_field[d["bins"]] = True
            ex.append(dict(rat=rat, date=str(row["date"]), day=int(row["day"]), unit=uid, quantile=q,
                           rate=rate, valid=R["valid"], fields=fd, srate=srate, sfields=sfd,
                           path=path, spikes=np.c_[s.x[si], s.y[si]], peak=peak, goal=s.goal,
                           mean_hz=float(u["mean_hz"]), si=float(u["si"])))
            cells.append(dict(rat=rat, date=str(row["date"]), day=int(row["day"]), unit=uid,
                              percentile=int(100 * q), n_spikes_run=int(u["n_spikes_run"]),
                              mean_hz=float(u["mean_hz"]), peak_hz=peak, threshold_hz=f.field_frac * peak,
                              si_bits_per_spike=float(u["si"]), si_p=float(u["si_p"]),
                              bins_above_threshold=int(above.sum()),
                              bins_rejected=int((above & ~in_field).sum()),
                              n_fields=len(fd), median_field_m=float(np.median([d["size_m"] for d in fd])),
                              n_fields_surrogate=len(sfd)))
            for j, d in enumerate(fd):
                frows.append(dict(rat=rat, date=str(row["date"]), unit=uid, field=j, size_m=d["size_m"],
                                  peak_hz=d["peak_hz"], peak_kind=d["peak_kind"],
                                  peak_element=d["peak_element"], n_bins=d["n_bins"]))
    params = pd.DataFrame([
        dict(parameter="maze bins", value=lm.n_bins),
        dict(parameter="node zones", value=int(lm.is_node().sum())),
        dict(parameter="node_radius_m", value=cfg.lin.node_radius_m),
        dict(parameter="edge_bin_m", value=cfg.lin.edge_bin_m),
        dict(parameter="speed_min_m_s", value=f.speed_min_m_s),
        dict(parameter="smooth_sigma_m", value=f.smooth_sigma_m),
        dict(parameter="min_occ_s", value=f.min_occ_s),
        dict(parameter="field_frac", value=f.field_frac),
        dict(parameter="min_peak_hz", value=f.min_peak_hz),
        dict(parameter="min_field_m", value=f.min_field_m),
        dict(parameter="n_shuffles", value=f.n_shuffles),
        dict(parameter="shuffle_min_shift_s", value=f.shuffle_min_shift_s),
        dict(parameter="alpha", value=f.alpha),
        dict(parameter="min_spikes", value=cfg.units.min_spikes),
    ])
    return dict(cells=pd.DataFrame(cells), fields=pd.DataFrame(frows), params=params, examples=ex)


_FIELD_COLORS = ["#0072b2", "#d55e00", "#009e73", "#cc79a7", "#e69f00", "#56b4e9", "#8c564b", "#7f7f7f", "#bcbd22"]
_FIELD_CMAP = ListedColormap(["k"] + _FIELD_COLORS)


def _field_values(e, lm, frac):
    """NaN below threshold, 0 = above threshold but rejected, 1.. = accepted field (colour index)."""
    v = np.full(lm.n_bins, np.nan)
    above = e["valid"] & (np.nan_to_num(e["rate"]) >= frac * e["peak"])
    v[above] = 0
    for j, d in enumerate(e["fields"]):
        v[d["bins"]] = 1 + j % len(_FIELD_COLORS)
    return v


def _mark_peaks(ax, flds, lm):
    if flds:
        b = [d["peak_bin"] for d in flds]
        ax.scatter(lm.bins["x"].to_numpy()[b], lm.bins["y"].to_numpy()[b], s=7, marker="v",
                   facecolors="w", edgecolors="r", linewidths=0.5, zorder=5)


def plot(res, cfg):
    lm = mz.maze_from_cfg(cfg)
    ex = res["examples"]
    n = len(ex)
    fig, axs = P.figure(180, 27 * n + 6, nrows=n, ncols=4, squeeze=False)
    heads = ["1. spikes (red) on the running path",
             "2. smoothed rate map (triangles: field peaks)",
             f"3. bins ≥ {cfg.fields.field_frac:.0%} of peak → fields",
             "4. same cell, spikes shifted in time"]
    for i, e in enumerate(ex):
        ax = axs[i]
        ax[0].plot(e["path"][:, 0], e["path"][:, 1], ".", ms=0.6, color="0.8", rasterized=True)
        ax[0].plot(e["spikes"][:, 0], e["spikes"][:, 1], ".", ms=0.9, color="r", alpha=0.5, rasterized=True)
        ax[0].set_aspect("equal"); ax[0].set_xlim(0.5, 9.0); ax[0].set_ylim(0.2, 4.6); ax[0].axis("off")
        ax[0].set_title(f"{e['rat']} day {e['day']}, unit {e['unit']}: {e['mean_hz']:.2f} Hz, "
                        f"SI {e['si']:.2f} bit/spk", loc="left", fontsize=5.5)
        vmax = max(e["peak"], np.nanmax(np.where(e["valid"], e["srate"], np.nan)))
        P.draw_maze(ax[1], lm, np.where(e["valid"], e["rate"], np.nan), vmin=0, vmax=vmax, lw=1.6,
                    node_size=5, goal=e["goal"])
        _mark_peaks(ax[1], e["fields"], lm)
        ax[1].set_title(f"peak {e['peak']:.1f} Hz, threshold {cfg.fields.field_frac * e['peak']:.1f} Hz",
                        loc="left", fontsize=5.5)
        P.draw_maze(ax[2], lm, _field_values(e, lm, cfg.fields.field_frac), cmap=_FIELD_CMAP,
                    vmin=0, vmax=len(_FIELD_COLORS), lw=1.6, node_size=5)
        ax[2].set_title(f"{len(e['fields'])} fields (black: rejected, too short or < "
                        f"{cfg.fields.min_peak_hz:g} Hz)", loc="left", fontsize=5.5)
        P.draw_maze(ax[3], lm, np.where(e["valid"], e["srate"], np.nan), vmin=0, vmax=vmax, lw=1.6,
                    node_size=5)
        _mark_peaks(ax[3], e["sfields"], lm)
        ax[3].set_title(f"surrogate: {len(e['sfields'])} fields, same colour scale", loc="left", fontsize=5.5)
        if i == 0:
            for a, h in zip(ax, heads):
                a.text(0.0, 1.32, h, transform=a.transAxes, fontsize=6.5, fontweight="bold", va="bottom")
    return [("field_examples", fig)]


# ----------------------------------------------------------------------------- LaTeX report
CAPTIONS = {"field_examples": (
    r"\textbf{Field detection, step by step.} One mid-training session per rat; rows are place cells at "
    r"the 10th, 50th and 90th percentile of field count in that session. (1) Spikes emitted while running "
    r"inside trials (red) on the animal's path (grey). (2) Rate map on the linearised maze, dark = 0, "
    r"yellow = the cell's peak; open circle = goal node; triangles = peaks of accepted fields. "
    r"(3) Bins at or above the threshold. Each connected group that passes the size and peak criteria is "
    r"one field (colours repeat); black = above threshold but rejected; grey = below threshold or not "
    r"visited. (4) Rate map and detected field peaks of the same cell after circularly shifting its spikes "
    r"in time, which removes spatial tuning but keeps the spike count and the path. Colour scale as in (2).")}


def latex(T):
    from .. import report as R
    p = T["params"].set_index("parameter")["value"]
    cm = lambda k: f"{100 * float(p[k]):g}"
    steps = [
        rf"\textbf{{Linearisation.}} Each position sample is projected onto the maze graph. Samples within "
        rf"{cm('node_radius_m')}\,cm of a node belong to that node's zone ({int(p['node zones'])} zones); the rest "
        rf"fall into {cm('edge_bin_m')}\,cm bins along the edges, giving {int(p['maze bins'])} bins in total.",
        rf"\textbf{{Samples used.}} Only samples inside trials with running speed "
        rf"$\geq {cm('speed_min_m_s')}$\,cm/s; units need $\geq {int(p['min_spikes'])}$ spikes in those samples.",
        rf"\textbf{{Rate map.}} Spike counts and occupancy per bin are each smoothed with a Gaussian of "
        rf"$\sigma = {cm('smooth_sigma_m')}$\,cm over \emph{{track}} distance (no leak between neighbouring "
        rf"corridors) and divided. Bins with smoothed occupancy $< {float(p['min_occ_s']):g}$\,s are invalid.",
        rf"\textbf{{Place cell.}} Skaggs information above the null from {int(p['n_shuffles'])} circular spike "
        rf"shifts ($\geq {float(p['shuffle_min_shift_s']):g}$\,s), $p < {float(p['alpha']):g}$, and at least one field.",
        rf"\textbf{{Field.}} A connected set of valid bins with rate $\geq {100 * float(p['field_frac']):g}\%$ of "
        rf"\emph{{that cell's}} peak rate, with an in-field peak $\geq {float(p['min_peak_hz']):g}$\,Hz and a "
        rf"track extent $\geq {cm('min_field_m')}$\,cm. The number of such sets is the cell's field count; "
        rf"a field's location is its peak bin.",
        r"\textbf{Surrogate.} The same steps on the cell's spikes after a circular shift in running time. "
        r"Any field found there is produced by sampling noise alone.",
    ]
    text = (r"The field counts and densities in the following sections all come from the procedure below "
            r"(Fig.~\ref{fig:field_examples}; the cells shown are listed in Table~\ref{tab:field_examples:cells})."
            "\n" + r"\begin{enumerate}" + "\n" + "\n".join(r"\item " + s for s in steps) + "\n"
            + r"\end{enumerate}" + "\n\n")
    text += R.table(T["cells"], "Example cells of Fig.~\\ref{fig:field_examples}.", "tab:field_examples:cells",
                    cols=["rat", "day", "unit", "percentile", "mean_hz", "peak_hz", "threshold_hz",
                          "si_bits_per_spike", "bins_above_threshold", "bins_rejected", "n_fields",
                          "median_field_m", "n_fields_surrogate"],
                    headers=dict(percentile="pctl.", mean_hz="mean (Hz)", peak_hz="peak (Hz)",
                                 threshold_hz="thr.\\ (Hz)", si_bits_per_spike="SI",
                                 bins_above_threshold="bins $\\geq$ thr.", bins_rejected="rejected",
                                 n_fields="fields", median_field_m="median size (m)",
                                 n_fields_surrogate="surrogate fields"), size=r"\scriptsize")
    return text
