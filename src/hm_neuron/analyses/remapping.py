"""6. How much does the map remap across goals? (population level)

Units are sorted per day, so single cells cannot be followed across goals.
Remapping is therefore measured on POPULATION maps that do not need identity:

* ``density``   field-peak density map: sum of field peaks per bin / bin length /
                n place cells, smoothed over track distance
* ``coverage``  fraction of place cells with a field covering each bin
* ``poprate``   mean peak-normalised rate map over place cells
* ``goal_prof`` field density as a function of track distance to the CURRENT goal
                (0.5 m bins), divided by the uniform-placement expectation

Pairs of days of the same rat are labelled ``same_goal`` (consecutive days, same
goal), ``switch`` (last old-goal day -> first new-goal day) or ``other``.
Correlations over bins valid on both days are compared between pair types.
The same correlation for the running-OCCUPANCY maps is reported as a
behavioural control, and a partial correlation removes it.

Within a day (where cells ARE tracked) a split-half baseline is computed:
rate-map correlation of each place cell between odd and even trials, and the
population-vector correlation between halves.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as st

from .. import fields as fl, maze as mz, plotting as P
from . import common

NAME = "remapping"
TITLE = "Remapping across goals (population maps) with same-goal baseline"
DEFAULT_QUERY = "day >= 0"
PROFILE_EDGES = np.arange(0, 10.5, 0.5)


def _corr(a, b, m, min_n=5):
    m = m & np.isfinite(a) & np.isfinite(b)
    if m.sum() < min_n or np.std(a[m]) == 0 or np.std(b[m]) == 0:
        return np.nan
    return float(np.corrcoef(a[m], b[m])[0, 1])


def _partial(a, b, covs, m):
    """Correlation of a and b after regressing both on the covariate maps."""
    C = np.column_stack(covs)
    m = m & np.isfinite(a) & np.isfinite(b) & np.isfinite(C).all(1)
    if m.sum() < 10:
        return np.nan
    X = np.column_stack([np.ones(m.sum()), C[m]])
    ra = a[m] - X @ np.linalg.lstsq(X, a[m], rcond=None)[0]
    rb = b[m] - X @ np.linalg.lstsq(X, b[m], rcond=None)[0]
    return float(np.corrcoef(ra, rb)[0, 1])


def _maps(s, R, lm, cfg):
    K = fl.smoothing_kernel(lm, cfg.fields.smooth_sigma_m)
    L = lm.bins["length"].to_numpy()
    F = R["fields"]
    F = F[F["is_place"]] if len(F) else F
    pc = R["units"]["is_place"].to_numpy()
    npc = max(int(pc.sum()), 1)
    peaks = np.bincount(F["peak_bin"].to_numpy(int), minlength=lm.n_bins) if len(F) else np.zeros(lm.n_bins)
    density = (K @ peaks) / (K @ L) / npc
    cov = np.zeros(lm.n_bins)
    for u, g in F.groupby("unit"):
        m = np.zeros(lm.n_bins, bool)
        m[np.concatenate(g["bins"].to_list())] = True
        cov += m
    cov /= npc
    rate = R["rate"][pc]
    with np.errstate(invalid="ignore"):
        norm = rate / np.nanmax(rate, axis=1, keepdims=True)
    poprate = np.nanmean(norm, 0) if len(norm) else np.full(lm.n_bins, np.nan)
    d = lm.dist_to_node(s.goal)
    v = R["valid"]
    h_obs = np.histogram(d[F["peak_bin"].to_numpy(int)], PROFILE_EDGES)[0] if len(F) else np.zeros(len(PROFILE_EDGES) - 1)
    h_exp = np.histogram(d[v], PROFILE_EDGES, weights=L[v])[0]
    with np.errstate(invalid="ignore", divide="ignore"):
        prof = (h_obs / max(h_obs.sum(), 1)) / (h_exp / h_exp.sum())
    occ = K @ R["occ"]
    return dict(density=density, coverage=cov, poprate=poprate, goal_prof=prof,
                occ=np.log1p(occ), valid=v)


def _split_half(s, R, lm, cfg):
    pcs = list(R["units"].loc[R["units"]["is_place"], "unit"])
    if len(pcs) < 3:
        return dict(split_cell_r=np.nan, split_pv_r=np.nan)
    tr = s.trials["trial"].to_numpy()
    A = fl.rate_maps(s, pcs, cfg, mask=np.isin(s.trial_id, tr[0::2]), lm=lm)
    B = fl.rate_maps(s, pcs, cfg, mask=np.isin(s.trial_id, tr[1::2]), lm=lm)
    m = A["valid"] & B["valid"]
    cell_r = [_corr(A["rate"][k], B["rate"][k], m) for k in range(len(pcs))]
    pv = [_corr(A["rate"][:, b], B["rate"][:, b], np.ones(len(pcs), bool)) for b in np.where(m)[0]]
    return dict(split_cell_r=float(np.nanmedian(cell_r)), split_pv_r=float(np.nanmean(pv)))


def run(sessions, cfg):
    lm = mz.maze_from_cfg(cfg)
    maps, meta, split = {}, {}, []
    for row, s, R in common.iter_loaded(sessions, cfg):
        sid = common.session_id(row)
        maps[sid] = _maps(s, R, lm, cfg)
        meta[sid] = row
        split.append(dict(session_id=sid, rat=row["rat"], day=row["day"], goal=row["goal"],
                          **_split_half(s, R, lm, cfg)))
    rows = []
    sids = list(maps)
    for i, a in enumerate(sids):
        for b in sids[i + 1:]:
            ra, rb = meta[a], meta[b]
            if ra["rat"] != rb["rat"]:
                continue
            first, second = (ra, rb) if ra["day"] < rb["day"] else (rb, ra)
            consecutive = abs(int(rb["day"]) - int(ra["day"])) == 1
            same = ra["goal"] == rb["goal"]
            kind = ("same_goal" if same and consecutive else
                    "switch" if consecutive and not same else "other")
            if consecutive and int(first["day"]) == 0:
                kind = "exploration_to_task"
            A, B = maps[a], maps[b]
            m = A["valid"] & B["valid"]
            rows.append(dict(rat=ra["rat"], s1=common.session_id(first), s2=common.session_id(second),
                             day1=int(first["day"]), day2=int(second["day"]), goal1=int(first["goal"]),
                             goal2=int(second["goal"]), dday=abs(int(rb["day"]) - int(ra["day"])),
                             same_goal=bool(same), pair=kind,
                             r_density=_corr(A["density"], B["density"], m),
                             r_coverage=_corr(A["coverage"], B["coverage"], m),
                             r_poprate=_corr(A["poprate"], B["poprate"], m),
                             r_occ=_corr(A["occ"], B["occ"], m),
                             r_density_partial_occ=_partial(A["density"], B["density"], [A["occ"], B["occ"]], m),
                             r_goal_profile=_corr(A["goal_prof"], B["goal_prof"],
                                                  np.ones(len(A["goal_prof"]), bool))))
    PR = pd.DataFrame(rows)
    tests = []
    cons = PR[PR["pair"].isin(["same_goal", "switch"])]
    for metric in ("r_density", "r_coverage", "r_poprate", "r_occ", "r_density_partial_occ", "r_goal_profile"):
        for grp, g in [("pooled", cons)] + list(cons.groupby("rat")):
            a = g.loc[g["pair"] == "same_goal", metric].dropna()
            b = g.loc[g["pair"] == "switch", metric].dropna()
            if len(a) >= 2 and len(b) >= 2:
                p = st.mannwhitneyu(a, b).pvalue
            else:
                p = np.nan
            tests.append(dict(group=grp, metric=metric, same_goal_mean=a.mean(), switch_mean=b.mean(),
                              n_same=len(a), n_switch=len(b), p_mannwhitney=p))
    # pairs at matched day lag: same goal vs different goal
    lagged = []
    for (rat, dd, sg), g in PR.groupby(["rat", "dday", "same_goal"]):
        lagged.append(dict(rat=rat, dday=dd, same_goal=sg, n=len(g),
                           r_density=g["r_density"].mean(), r_poprate=g["r_poprate"].mean(),
                           r_occ=g["r_occ"].mean()))
    return dict(pairs=PR, tests=pd.DataFrame(tests), by_lag=pd.DataFrame(lagged), split_half=pd.DataFrame(split))


def plot(res, cfg):
    PR, LG, SP = res["pairs"], res["by_lag"], res["split_half"]
    fig, ax = P.figure(180, 55, ncols=4)
    cons = PR[PR["pair"].isin(["same_goal", "switch", "exploration_to_task"])]
    order = ["same_goal", "switch", "exploration_to_task"]
    for j, metric in enumerate(("r_density", "r_poprate")):
        for i, k in enumerate(order):
            for rat, g in cons[cons["pair"] == k].groupby("rat"):
                x = i + (0.12 if rat == "Rat6" else -0.12)
                ax[j].scatter(np.full(len(g), x), g[metric], s=8, color=P.RAT_COLORS.get(rat))
                ax[j].scatter(np.full(len(g), x), g["r_occ"], s=8, marker="x", color=P.RAT_COLORS.get(rat), alpha=0.4)
        ax[j].set(xticks=range(3), xticklabels=["same goal", "switch", "expl→task"],
                  ylabel={"r_density": "field-density map r", "r_poprate": "population rate-map r"}[metric],
                  title="o maps, x occupancy")
    if len(LG):
        for (rat, sg), g in LG.groupby(["rat", "same_goal"]):
            ax[2].plot(g["dday"], g["r_poprate"], "o-" if sg else "s--", ms=3, color=P.RAT_COLORS.get(rat),
                       label=f"{rat} {'same' if sg else 'diff'} goal")
        ax[2].set(xlabel="days apart", ylabel="population rate-map r")
        ax[2].legend(frameon=False, fontsize=5)
    for rat, g in SP.groupby("rat"):
        ax[3].plot(g["day"], g["split_cell_r"], "o-", ms=3, color=P.RAT_COLORS.get(rat), label=rat)
    ax[3].set(xlabel="day", ylabel="within-day split-half map r (median cell)")
    ax[3].legend(frameon=False)
    return [("remapping", fig)]
