"""4. How stable is each cell's activity from visit to visit?

For every place cell and every maze element (node / edge) its fields cover:

* ``reliability``     fraction of visits on which the cell is active (> ``visits.active_hz``)
* ``rate_cv``         CV of the running rate across ACTIVE visits
* ``log_ratio``       median |log(r_i / r_j)| over pairs of active visits (0 = identical rates)
* ``profile_corr``    edges only: mean Pearson r of the within-edge spatial rate profile
                      between pairs of active visits, split by same / opposite running direction

Null for profile consistency: the same pairs with one profile circularly
shifted along the edge (keeps rate and smoothness, destroys position).
Drift: profile correlation vs time between the two visits.
Also per unit: split-visit (odd vs even visits) correlation of the element-level rate map.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

from .. import maze as mz, plotting as P, stats as S, visits as V
from . import common

NAME = "stability"
TITLE = "Visit-to-visit consistency of place-cell activity"
DEFAULT_QUERY = "repeat > 0"  # repeat 0 = habituation, not training
MAX_PAIRS = 60


def _corr(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3 or np.std(a[m]) == 0 or np.std(b[m]) == 0:
        return np.nan
    return float(np.corrcoef(a[m], b[m])[0, 1])


def run(sessions, cfg):
    lm = mz.maze_from_cfg(cfg)
    elem_of_bin = lm.element_of_bin()
    rng = np.random.default_rng(cfg.fields.seed)
    ue_rows, pair_rows, unit_rows = [], [], []
    for row, s, R in common.iter_loaded(sessions, cfg):
        sid = common.session_id(row)
        U = R["units"]
        pcs = list(U.loc[U["is_place"], "unit"])
        if not pcs:
            continue
        vis, vid = V.segment_visits(s, cfg, lm)
        vis = vis[vis["run_s"] >= cfg.visits.min_dur_s]
        keep = vis.index.to_numpy()
        counts = V.visit_counts(s, pcs, vis.reset_index(drop=True),
                                np.where(np.isin(vid, keep), np.searchsorted(keep, vid), -1), cfg)
        vis = vis.reset_index(drop=True)
        vid2 = np.where(np.isin(vid, keep), np.searchsorted(keep, vid), -1)
        occ, prof_cnt, nbv = V.edge_profiles(s, pcs, vis, vid2, lm, cfg)
        rates = counts / vis["run_s"].to_numpy()[None]
        F = R["fields"]
        uidx = {u: i for i, u in enumerate(R["uids"])}
        for k, u in enumerate(pcs):
            fb = np.concatenate([b for b in F.loc[F["unit"] == u, "bins"]])
            elems = sorted(set(elem_of_bin[fb]))
            r_u = rates[k]
            for e in elems:
                vi = np.where(vis["element"].to_numpy() == e)[0]
                if len(vi) < 3:
                    continue
                act = vi[r_u[vi] > cfg.visits.active_hz]
                ra = r_u[act]
                rec = dict(session_id=sid, rat=row["rat"], day=row["day"], unit=u, element=e,
                           kind="node" if e[0] == "N" else "edge", n_visits=len(vi), n_active=len(act),
                           reliability=len(act) / len(vi), rate_mean=float(r_u[vi].mean()),
                           rate_cv=float(ra.std() / ra.mean()) if len(ra) >= 2 else np.nan)
                if len(act) >= 2:
                    pairs = list(itertools.combinations(act, 2))
                    if len(pairs) > MAX_PAIRS:
                        pairs = [pairs[i] for i in rng.choice(len(pairs), MAX_PAIRS, replace=False)]
                    lr = [abs(np.log(r_u[i] / r_u[j])) for i, j in pairs]
                    rec["log_ratio"] = float(np.median(lr))
                    if e[0] == "E":
                        nb = nbv[act[0]]
                        same, opp, null = [], [], []
                        for i, j in pairs:
                            with np.errstate(invalid="ignore", divide="ignore"):
                                pi = np.where(occ[i, :nb] > 0, prof_cnt[u][i, :nb] / occ[i, :nb], np.nan)
                                pj = np.where(occ[j, :nb] > 0, prof_cnt[u][j, :nb] / occ[j, :nb], np.nan)
                            pi_s = gaussian_filter1d(np.nan_to_num(pi), 1.0)
                            pj_s = gaussian_filter1d(np.nan_to_num(pj), 1.0)
                            pi_s[~np.isfinite(pi)] = np.nan
                            pj_s[~np.isfinite(pj)] = np.nan
                            c = _corr(pi_s, pj_s)
                            if not np.isfinite(c):
                                continue
                            sd = vis.at[i, "direction"] == vis.at[j, "direction"]
                            (same if sd else opp).append(c)
                            null.append(_corr(np.roll(pi_s, rng.integers(1, max(nb, 2))), pj_s))
                            pair_rows.append(dict(session_id=sid, rat=row["rat"], unit=u, element=e,
                                                  same_dir=bool(sd), corr=c, null_corr=null[-1],
                                                  dt_s=abs(vis.at[j, "t_start"] - vis.at[i, "t_start"]),
                                                  dtrial=abs(vis.at[j, "trial"] - vis.at[i, "trial"])))
                        rec["profile_corr_same"] = float(np.nanmean(same)) if same else np.nan
                        rec["profile_corr_opp"] = float(np.nanmean(opp)) if opp else np.nan
                        rec["profile_corr_null"] = float(np.nanmean(null)) if null else np.nan
                ue_rows.append(rec)
            # split-visit element-map reliability (all elements with >= 2 visits)
            ev = vis.groupby("element").indices
            odd, even = [], []
            for e, ii in ev.items():
                if len(ii) >= 2:
                    odd.append(r_u[ii[0::2]].mean())
                    even.append(r_u[ii[1::2]].mean())
            unit_rows.append(dict(session_id=sid, rat=row["rat"], day=row["day"], unit=u,
                                  split_visit_r=_corr(np.array(odd), np.array(even)),
                                  si=float(U.loc[U["unit"] == u, "si"].iloc[0])))
    UE = pd.DataFrame(ue_rows)
    PR = pd.DataFrame(pair_rows)
    UN = pd.DataFrame(unit_rows)
    tests = []
    if len(PR):
        for grp, d in [("pooled", PR)] + list(PR.groupby("rat")):
            per_ses = d.groupby("session_id").agg(obs=("corr", "mean"), null=("null_corr", "mean"))
            from scipy.stats import wilcoxon
            w = wilcoxon(per_ses["obs"], per_ses["null"]) if len(per_ses) > 5 else None
            tests.append(dict(group=grp, test="profile corr vs shifted null (per-session means)",
                              obs=per_ses["obs"].mean(), null=per_ses["null"].mean(),
                              p=w.pvalue if w is not None else np.nan, n_sessions=len(per_ses)))
        PR["log_dt"] = np.log10(PR["dt_s"].clip(lower=1))
        PR["same_dir_i"] = PR["same_dir"].astype(int)
        drift = S.mixed("corr ~ log_dt + same_dir_i", PR.assign(rat=PR["rat"]))
        drift["model"] = "corr ~ log10(dt) + same_dir, (1|session)"
    else:
        drift = pd.DataFrame()
    return dict(unit_element=UE, pairs=PR, units=UN, tests=pd.DataFrame(tests), drift=drift)


def plot(res, cfg):
    UE, PR, UN = res["unit_element"], res["pairs"], res["units"]
    fig, ax = P.figure(180, 55, ncols=4)
    for k, g in UE.groupby("kind"):
        ax[0].hist(g["reliability"], bins=20, range=(0, 1), histtype="step", density=True,
                   color=P.KIND_COLORS[k], label=k)
    ax[0].set(xlabel="reliability (frac. active visits)", ylabel="density")
    ax[0].legend(frameon=False)
    if len(PR):
        bins = np.linspace(-1, 1, 41)
        ax[1].hist(PR.loc[PR["same_dir"], "corr"], bins, histtype="step", density=True, label="same dir")
        ax[1].hist(PR.loc[~PR["same_dir"], "corr"], bins, histtype="step", density=True, label="opposite dir")
        ax[1].hist(PR["null_corr"].dropna(), bins, histtype="step", density=True, color="0.5", label="shifted null")
        ax[1].set(xlabel="visit-pair profile r", ylabel="density")
        ax[1].legend(frameon=False)
        q = pd.qcut(PR["dt_s"], 10, duplicates="drop")
        for sd, g in PR.groupby("same_dir"):
            m = g.groupby(q.loc[g.index], observed=True).agg(x=("dt_s", "median"), y=("corr", "mean"))
            ax[2].plot(m["x"], m["y"], "o-", ms=3, label="same dir" if sd else "opposite")
        ax[2].set(xscale="log", xlabel="time between visits (s)", ylabel="mean profile r")
    ax[3].hist(UN["split_visit_r"].dropna(), bins=30, color="0.6")
    ax[3].set(xlabel="odd vs even visits: element-map r", ylabel="place cells")
    return [("stability", fig)]
