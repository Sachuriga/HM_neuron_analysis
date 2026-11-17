"""7. What drives the drop in spatial information when a new goal is introduced?

Day types: ``switch`` (first day with a new goal), ``recall`` (last day with the
old goal), ``stable`` (other task days).

A. The drop
   * session-level SI (shuffle z) of all selected pyramidal units by day type,
     mixed model ``si_z ~ day_type + rat`` with a random intercept per session
   * trial-resolved: bias-corrected SI (SI − mean shuffled SI) in sliding windows
     of ``goal.window_trials`` trials, plus sparsity and peak/mean selectivity

B. Mechanism, tested within each day (cells are tracked within a day)
   Templates = place-cell rate maps from the FIRST and LAST third of trials;
   held-out data = running time bins (``goal.decode_bin_s``) of the MIDDLE third.
   For each bin, Poisson log-likelihood of the population spike counts given the
   rat's actual positions under the early map, the late map and a flat
   (non-spatial, mean-rate) model.
   * coexistence / flicker of old & new map  -> the early-vs-late log-likelihood
     ratio per spike (LLR) is large in magnitude and BIMODAL across bins (bimodality
     coefficient > 0.555), and the per-trial mean LLR switches sign
   * poor selectivity                          -> both maps barely beat the flat model
     (low spatial gain = (LL_best_map − LL_flat) / spike), LLR near 0
   Every metric is compared between switch days and stable days (the control
   for normal within-day drift).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as st

from .. import fields as fl, maze as mz, plotting as P, stats as S
from . import common

NAME = "info_drop"
TITLE = "Spatial-information drop at goal switch: selectivity vs map coexistence"
DEFAULT_QUERY = "repeat > 0"  # repeat 0 = habituation, not training
EPS_HZ = 0.05


def day_type(row):
    if row["is_switch_day"]:
        return "switch"
    if row["is_recall_day"]:
        return "recall"
    return "stable"


def bimodality_coefficient(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 10:
        return np.nan
    g = st.skew(x)
    k = st.kurtosis(x)
    return float((g ** 2 + 1) / (k + 3 * (n - 1) ** 2 / ((n - 2) * (n - 3))))


def _window_si(s, R, cfg, lm, uids, rng):
    K = fl.smoothing_kernel(lm, cfg.fields.smooth_sigma_m)
    tr = s.trials["trial"].to_numpy()
    W = cfg.goal.window_trials
    rows = []
    sub = cfg.fields.__class__(**{**cfg.fields.__dict__, "n_shuffles": cfg.goal.n_shuffles_window})
    cfg_w = cfg.__class__(**{**cfg.__dict__, "fields": sub})
    for w0 in range(0, max(len(tr) - W + 1, 1)):
        trials = tr[w0:w0 + W]
        mask = fl.running_mask(s, cfg, np.isin(s.trial_id, trials))
        occ = fl.occupancy(s, mask, lm.n_bins)
        spk = fl.spike_counts(s, uids, mask, lm.n_bins)
        rate, valid, _ = fl.smooth_rate(spk, occ, K, cfg.fields.min_occ_s)
        si = fl.skaggs_info(rate, occ, valid)
        p = np.where(valid, occ, 0) / np.where(valid, occ, 0).sum()
        for k, u in enumerate(uids):
            if spk[k].sum() < 10:
                continue
            null = fl.shuffled_info(s, u, cfg_w, mask, lm, occ, valid, K, rng)
            r = np.where(valid, rate[k], 0)
            mean = (p * r).sum()
            sparsity = mean ** 2 / max((p * r ** 2).sum(), 1e-12)
            rows.append(dict(window=w0, first_trial=int(trials[0]), unit=u, n_spikes=int(spk[k].sum()),
                             si=float(si[k]), si_corr=float(si[k] - np.nanmean(null)),
                             sparsity=float(sparsity), peak_over_mean=float(np.nanmax(r) / max(mean, 1e-9)),
                             run_s=float(mask.sum() * s.dt)))
    return rows


def _llr(s, R, cfg, lm):
    pcs = list(R["units"].loc[R["units"]["is_place"], "unit"])
    groups = common.trial_groups(s, 3)
    if len(pcs) < 5 or len(groups) < 3:
        return None, None
    early = fl.rate_maps(s, pcs, cfg, mask=np.isin(s.trial_id, groups[0]), lm=lm)
    late = fl.rate_maps(s, pcs, cfg, mask=np.isin(s.trial_id, groups[-1]), lm=lm)
    ev = fl.running_mask(s, cfg, np.isin(s.trial_id, groups[1]))
    idx = np.where(ev)[0]
    if len(idx) < 50:
        return None, None
    tb = np.floor(s.t[idx] / cfg.goal.decode_bin_s).astype(int)
    ub, tb_id = np.unique(tb, return_inverse=True)
    dt = s.dt
    def lam(M):
        r = np.nan_to_num(M["rate"], nan=EPS_HZ)
        r = np.maximum(r, EPS_HZ)
        x = r[:, s.bin[idx]] * dt
        out = np.zeros((len(pcs), len(ub)))
        np.add.at(out.T, tb_id, x.T)
        return out
    le, ll_ = lam(early), lam(late)
    mean_hz = R["units"].set_index("unit").loc[pcs, "mean_hz"].to_numpy()
    occ_tb = np.bincount(tb_id, minlength=len(ub)) * dt
    lf = np.maximum(mean_hz, EPS_HZ)[:, None] * occ_tb[None]
    cnt = np.zeros((len(pcs), len(ub)))
    pos_in_ev = np.full(len(s.t), -1)
    pos_in_ev[idx] = tb_id
    for k, u in enumerate(pcs):
        ss = fl.spike_samples(s, s.spikes[u])
        ss = ss[ev[ss]]
        cnt[k] = np.bincount(pos_in_ev[ss], minlength=len(ub))
    LL = lambda l: (cnt * np.log(l) - l).sum(0)
    lle, lll, llf = LL(le), LL(ll_), LL(lf)
    nsp = cnt.sum(0)
    keep = (occ_tb >= 0.5 * cfg.goal.decode_bin_s) & (nsp > 0)
    tr_of_bin = np.zeros(len(ub), int)
    tr_of_bin[tb_id] = s.trial_id[idx]
    bins = pd.DataFrame(dict(t_bin=ub, trial=tr_of_bin, n_spikes=nsp, ll_early=lle, ll_late=lll, ll_flat=llf,
                             llr=lll - lle))[keep]
    bins["gain_per_spike"] = (np.maximum(bins["ll_early"], bins["ll_late"]) - bins["ll_flat"]) / bins["n_spikes"]
    # per-spike LLR: comparable across bins/sessions with different population rates
    bins["llr_per_spike"] = bins["llr"] / bins["n_spikes"]
    m = early["valid"] & late["valid"]
    cell_r = [np.corrcoef(early["rate"][k, m], late["rate"][k, m])[0, 1] for k in range(len(pcs))]
    per_trial = bins.groupby("trial")["llr_per_spike"].mean()
    summ = dict(n_place=len(pcs), n_bins=len(bins), early_late_map_r=float(np.nanmedian(cell_r)),
                llr_mean=float(bins["llr_per_spike"].mean()), llr_sd=float(bins["llr_per_spike"].std()),
                llr_abs_median=float(bins["llr_per_spike"].abs().median()),
                frac_late=float((bins["llr"] > 0).mean()),
                bimodality=bimodality_coefficient(bins["llr_per_spike"]),
                gain_per_spike=float(bins["gain_per_spike"].median()),
                trial_sign_changes=int((np.diff(np.sign(per_trial.to_numpy())) != 0).sum()),
                n_mid_trials=len(per_trial))
    return summ, bins


def run(sessions, cfg):
    lm = mz.maze_from_cfg(cfg)
    rng = np.random.default_rng(cfg.fields.seed)
    U, W, SM, B = [], [], [], []
    for row, s, R in common.iter_loaded(sessions, cfg):
        sid = common.session_id(row)
        dtp = day_type(row)
        meta = dict(session_id=sid, rat=row["rat"], day=row["day"], day_type=dtp, goal=row["goal"])
        U.append(R["units"].assign(**meta))
        uids = list(R["units"].loc[R["units"]["n_spikes_run"] >= cfg.units.min_spikes, "unit"])
        W += [{**meta, **r} for r in _window_si(s, R, cfg, lm, uids, rng)]
        summ, bins = _llr(s, R, cfg, lm)
        if summ:
            SM.append({**meta, **summ})
            B.append(bins.assign(**meta))
    U = pd.concat(U, ignore_index=True)
    W = pd.DataFrame(W)
    SM = pd.DataFrame(SM)
    B = pd.concat(B, ignore_index=True) if B else pd.DataFrame()
    models = []
    for y in ("si_z", "si"):
        d = U.dropna(subset=[y])
        if d["session_id"].nunique() >= 3 and d["day_type"].nunique() > 1:
            models.append(S.mixed(f"{y} ~ C(day_type, Treatment('stable')) + C(rat)", d).assign(y=y))
    if len(W) and W["session_id"].nunique() >= 3 and W["day_type"].nunique() > 1:
        wm = W.groupby(["session_id", "rat", "day_type", "window"])[["si_corr", "sparsity"]].mean().reset_index()
        early = wm[wm["window"] < 3]
        for y in ("si_corr", "sparsity"):
            models.append(S.mixed(f"{y} ~ C(day_type, Treatment('stable')) + C(rat)", early)
                          .assign(y=f"{y} (first 3 windows)"))
    comp = []
    if len(SM):
        for metric in ("early_late_map_r", "llr_sd", "llr_abs_median", "bimodality",
                       "gain_per_spike", "trial_sign_changes"):
            a = SM.loc[SM["day_type"] == "switch", metric].dropna()
            b = SM.loc[SM["day_type"] == "stable", metric].dropna()
            comp.append(dict(metric=metric, switch_mean=a.mean(), stable_mean=b.mean(), n_switch=len(a),
                             n_stable=len(b),
                             p_mannwhitney=st.mannwhitneyu(a, b).pvalue if len(a) >= 2 and len(b) >= 2 else np.nan))
    return dict(units=U, windows=W, session_mechanism=SM, llr_bins=B,
                models=pd.concat(models, ignore_index=True) if models else pd.DataFrame(),
                switch_vs_stable=pd.DataFrame(comp))


def plot(res, cfg):
    U, W, SM, B = res["units"], res["windows"], res["session_mechanism"], res["llr_bins"]
    colors = {"switch": "#d55e00", "recall": "#009e73", "stable": "0.4"}
    fig, ax = P.figure(180, 100, nrows=2, ncols=3)
    ax = ax.ravel()
    for i, (dt, g) in enumerate(U.groupby("day_type")):
        v = g.groupby("session_id")["si_z"].median()
        ax[0].scatter(np.full(len(v), i) + np.random.default_rng(1).normal(0, .05, len(v)), v, s=8, color=colors[dt])
    ax[0].set(xticks=range(U["day_type"].nunique()), xticklabels=sorted(U["day_type"].unique()),
              ylabel="session median SI (shuffle z)")
    if len(W):
        for dt, g in W.groupby("day_type"):
            m = g.groupby("window")["si_corr"].agg(["mean", "sem"])
            ax[1].plot(m.index, m["mean"], color=colors[dt], label=dt)
            ax[1].fill_between(m.index, m["mean"] - m["sem"], m["mean"] + m["sem"], color=colors[dt], alpha=0.2)
            m2 = g.groupby("window")["sparsity"].mean()
            ax[2].plot(m2.index, m2.values, color=colors[dt], label=dt)
        ax[1].set(xlabel=f"trial window ({cfg.goal.window_trials} trials)", ylabel="bias-corrected SI (bits/spike)")
        ax[1].legend(frameon=False)
        ax[2].set(xlabel="trial window", ylabel="sparsity (low = selective)")
    if len(B):
        bins = np.linspace(-1.5, 1.5, 61)
        for dt, g in B.groupby("day_type"):
            ax[3].hist(g["llr_per_spike"].clip(-1.5, 1.5), bins, histtype="step", density=True,
                       color=colors[dt], label=dt)
        ax[3].set(xlabel="LLR late vs early map per spike (middle third)", ylabel="density")
        ax[3].legend(frameon=False)
    if len(SM):
        for dt, g in SM.groupby("day_type"):
            ax[4].scatter(g["gain_per_spike"], g["bimodality"], s=10, color=colors[dt], label=dt)
            ax[5].scatter(g["day"], g["early_late_map_r"], s=10, color=colors[dt])
        ax[4].axhline(0.555, color="k", lw=0.6, ls=":")
        ax[4].set(xlabel="spatial gain (nats/spike, best map vs flat)", ylabel="LLR bimodality coefficient")
        ax[5].set(xlabel="day", ylabel="early vs late map r (median cell)")
    return [("info_drop", fig)]
