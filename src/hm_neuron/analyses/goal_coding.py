"""3. Temporal evolution of goal coding and how cells are recruited.

A. Across days and within sessions (early / middle / late thirds of trials):
   * goal enrichment = fraction of field peaks within ``goal.radius_m`` (track)
     of the goal / fraction of valid track inside that zone (1 = no enrichment)
   * the same for the PREVIOUS goal (does the old goal stay over-represented?)
   * median peak-to-goal distance relative to a uniform-placement null
   * goal cells: fraction of place cells with >= 1 field in the goal zone vs
     the expected 1-(1-q)^n_fields per cell (q = zone fraction); z-test on the
     sum of Bernoulli probabilities.

B. Recruitment within a session (cells are tracked within a day only):
   for each session-level field, the first trial on which the rat ran through
   it ("sampled") and the first sampled trial on which the cell fired in it
   (>= max(1 Hz, 0.3 * field peak)). ``lag`` = sampled trials before the first
   expression. Hypotheses:
     distance-based: lag correlates with goal distance (mixed model, Spearman)
     goal cells:     fields in the goal zone are expressed earlier than others
     random:         neither
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as st

from .. import fields as fl, maze as mz, plotting as P, stats as S
from . import common

NAME = "goal_coding"
TITLE = "Goal coding over days/trials; recruitment of fields"
DEFAULT_QUERY = "day > 0"


def _goal_cells(F, R, lm, goal, radius):
    d = lm.dist_to_node(goal)
    L = lm.bins["length"].to_numpy()
    v = R["valid"]
    q = L[v & (d <= radius)].sum() / L[v].sum()
    pc = R["units"][R["units"]["is_place"]]
    if pc.empty:
        return dict(frac_goal_cells=np.nan, exp_frac_goal_cells=np.nan, goal_cells_z=np.nan, goal_cells_p=np.nan)
    has = F[F["is_place"]].groupby("unit")["goal_dist_m"].min() <= radius
    obs = has.reindex(pc["unit"]).fillna(False).astype(float).to_numpy()
    pexp = 1 - (1 - q) ** pc["n_fields"].to_numpy()
    z = (obs.sum() - pexp.sum()) / np.sqrt((pexp * (1 - pexp)).sum() + 1e-12)
    return dict(frac_goal_cells=obs.mean(), exp_frac_goal_cells=pexp.mean(),
                goal_cells_z=z, goal_cells_p=2 * st.norm.sf(abs(z)))


def _recruitment(s, R, cfg, lm, sid):
    F = R["fields"]
    F = F[F["is_place"]] if len(F) else F
    if F.empty:
        return []
    run = fl.running_mask(s, cfg)
    trials = s.trials["trial"].to_numpy()
    tix = {t: i for i, t in enumerate(trials)}
    rows = []
    for u, Fu in F.groupby("unit"):
        ss = fl.spike_samples(s, s.spikes[u])
        ss = ss[run[ss]]
        for f in Fu.itertuples():
            inb = np.zeros(lm.n_bins, bool)
            inb[f.bins] = True
            in_field = run & (s.bin >= 0)
            in_field[in_field] = inb[s.bin[in_field]]
            occ_t = np.bincount(np.searchsorted(trials, s.trial_id[in_field]), minlength=len(trials)) * s.dt
            sp = ss[in_field[ss]]
            cnt_t = np.bincount(np.searchsorted(trials, s.trial_id[sp]), minlength=len(trials))
            sampled = occ_t >= 0.2
            if not sampled.any():
                continue
            with np.errstate(invalid="ignore", divide="ignore"):
                rate_t = cnt_t / occ_t
            thr = max(1.0, 0.3 * f.peak_hz)
            expressed = sampled & (rate_t >= thr)
            si = np.where(sampled)[0]
            first_sampled = si[0]
            lag = int(np.argmax(expressed[si])) if expressed.any() else np.nan
            rows.append(dict(session_id=sid, unit=u, field=f.field, goal_dist_m=f.goal_dist_m,
                             in_goal_zone=f.goal_dist_m <= cfg.goal.radius_m, size_m=f.size_m,
                             peak_hz=f.peak_hz, n_sampled=int(sampled.sum()),
                             first_sampled_trial=int(trials[first_sampled]), lag=lag,
                             expressed_frac=float(expressed[si].mean()),
                             first_trial_frac=first_sampled / max(len(trials) - 1, 1)))
    return rows


def run(sessions, cfg):
    lm = mz.maze_from_cfg(cfg)
    rad = cfg.goal.radius_m
    ses_rows, grp_rows, rec_rows = [], [], []
    for row, s, R in common.iter_loaded(sessions, cfg):
        sid = common.session_id(row)
        F = R["fields"]
        Fp = F[F["is_place"]] if len(F) else F
        meta = dict(session_id=sid, rat=row["rat"], day=row["day"], block=row["block"],
                    day_in_block=row["day_in_block"], goal=row["goal"], prev_goal=row["prev_goal"],
                    is_switch_day=row["is_switch_day"], is_recall_day=row["is_recall_day"])
        pk = Fp["peak_bin"].to_numpy() if len(Fp) else np.array([], int)
        e_new = common.goal_enrichment(pk, R["valid"], lm, row["goal"], rad)
        e_old = common.goal_enrichment(pk, R["valid"], lm, row["prev_goal"], rad)
        dn, w = common.goal_null_distances(R, lm, row["goal"])
        null_med = common.weighted_quantile(dn, w, 0.5)
        ses_rows.append(dict(**meta, n_fields=len(pk), n_place=int(R["units"]["is_place"].sum()),
                             enrich_goal=e_new["enrichment"], frac_goal=e_new["frac_in"],
                             exp_frac_goal=e_new["exp_frac"], enrich_prev_goal=e_old["enrichment"],
                             median_goal_dist=float(np.median(Fp["goal_dist_m"])) if len(Fp) else np.nan,
                             null_median_goal_dist=null_med,
                             binom_p_goal=(st.binomtest(e_new["n_in"], e_new["n"], e_new["exp_frac"]).pvalue
                                           if e_new["n"] else np.nan),
                             **_goal_cells(F, R, lm, row["goal"], rad)))
        for gi, trials in enumerate(common.trial_groups(s, cfg.goal.n_trial_groups)):
            M, Fg, _ = common.group_fields(s, R, cfg, lm, trials)
            pkg = Fg["peak_bin"].to_numpy() if len(Fg) else np.array([], int)
            a = common.goal_enrichment(pkg, M["valid"], lm, row["goal"], rad)
            b = common.goal_enrichment(pkg, M["valid"], lm, row["prev_goal"], rad)
            grp_rows.append(dict(**meta, trial_group=gi, first_trial=int(trials[0]), last_trial=int(trials[-1]),
                                 n_fields=len(pkg), enrich_goal=a["enrichment"], enrich_prev_goal=b["enrichment"],
                                 median_goal_dist=float(np.median(Fg["goal_dist_m"])) if len(Fg) else np.nan))
        for r in _recruitment(s, R, cfg, lm, sid):
            rec_rows.append({**meta, **r})
    SES, GRP, REC = pd.DataFrame(ses_rows), pd.DataFrame(grp_rows), pd.DataFrame(rec_rows)
    SES["q_goal"] = S.fdr_bh(SES["binom_p_goal"])
    tests = []
    if len(REC):
        d = REC.dropna(subset=["lag"]).copy()
        for grp, g in [("pooled", d)] + list(d.groupby("rat")):
            tests.append(dict(group=grp, test="lag vs goal distance (Spearman, fields)",
                              **S.spearman(g["goal_dist_m"].to_numpy(), g["lag"].to_numpy(float))))
            a, b = g.loc[g["in_goal_zone"], "lag"], g.loc[~g["in_goal_zone"], "lag"]
            if len(a) >= 3 and len(b) >= 3:
                mw = st.mannwhitneyu(a, b)
                tests.append(dict(group=grp, test="lag: goal-zone vs other fields (Mann-Whitney)",
                                  rho=a.mean() - b.mean(), p=mw.pvalue, n=len(a) + len(b)))
        # first sampling confound: control for when the field was first sampled
        mm = S.mixed("lag ~ goal_dist_m + first_trial_frac + C(rat)", d)
        mm["model"] = "lag ~ goal_dist + first_trial_frac + rat, (1|session)"
    else:
        mm = pd.DataFrame()
    ev = []
    for y in ("enrich_goal", "enrich_prev_goal", "median_goal_dist"):
        dd = SES.dropna(subset=[y])
        if dd["session_id"].nunique() >= 3:
            from statsmodels.formula.api import ols
            for grp, g in [("pooled", dd)] + list(dd.groupby("rat")):
                if len(g) < 4:
                    continue
                f = f"{y} ~ day_in_block" + (" + C(rat)" if grp == "pooled" and g["rat"].nunique() > 1 else "")
                r = ols(f, g).fit()
                ev.append(dict(group=grp, y=y, term="day_in_block", coef=r.params["day_in_block"],
                               p=r.pvalues["day_in_block"], n=len(g)))
    return dict(sessions=SES, trial_groups=GRP, recruitment=REC, recruitment_tests=pd.DataFrame(tests),
                recruitment_model=mm, evolution=pd.DataFrame(ev))


def plot(res, cfg):
    SES, GRP, REC = res["sessions"], res["trial_groups"], res["recruitment"]
    fig, ax = P.figure(180, 100, nrows=2, ncols=3)
    ax = ax.ravel()
    for rat, g in SES.groupby("rat"):
        c = P.RAT_COLORS.get(rat)
        ax[0].plot(g["day"], g["enrich_goal"], "o-", ms=3, color=c, label=f"{rat} current goal")
        ax[0].plot(g["day"], g["enrich_prev_goal"], "s--", ms=3, color=c, mfc="none", label=f"{rat} previous goal")
        for d in g.loc[g["is_switch_day"], "day"]:
            ax[0].axvline(d - 0.5, color="0.8", lw=0.6)
    ax[0].axhline(1, color="k", lw=0.6)
    ax[0].set(xlabel="day", ylabel="goal-zone field enrichment (obs/exp)")
    ax[0].legend(frameon=False, fontsize=5)
    for rat, g in SES.groupby("rat"):
        ax[1].plot(g["day"], g["frac_goal_cells"], "o-", ms=3, color=P.RAT_COLORS.get(rat), label=rat)
        ax[1].plot(g["day"], g["exp_frac_goal_cells"], ":", color=P.RAT_COLORS.get(rat))
    ax[1].set(xlabel="day", ylabel="fraction goal cells (obs, : expected)")
    for rat, g in SES.groupby("rat"):
        ax[2].plot(g["day"], g["median_goal_dist"] / g["null_median_goal_dist"], "o-", ms=3,
                   color=P.RAT_COLORS.get(rat))
    ax[2].axhline(1, color="k", lw=0.6)
    ax[2].set(xlabel="day", ylabel="median field–goal distance / null")
    if len(GRP):
        GRP = GRP.assign(x=GRP["day_in_block"] + GRP["trial_group"] / cfg.goal.n_trial_groups)
        for (rat, blk), g in GRP.groupby(["rat", "block"]):
            ax[3].plot(g["x"], g["enrich_goal"], "-", color=P.RAT_COLORS.get(rat), alpha=0.6, lw=0.8)
        ax[3].axhline(1, color="k", lw=0.6)
        ax[3].set(xlabel="day in goal block (+ trial third)", ylabel="goal enrichment")
    if len(REC):
        d = REC.dropna(subset=["lag"])
        ax[4].scatter(d["goal_dist_m"], d["lag"] + np.random.default_rng(0).uniform(-.2, .2, len(d)),
                      s=2, alpha=0.3, color="0.3")
        q = pd.cut(d["goal_dist_m"], np.arange(0, d["goal_dist_m"].max() + 1, 1))
        m = d.groupby(q, observed=True)["lag"].mean()
        ax[4].plot([i.mid for i in m.index], m.values, "o-", color="k", ms=3)
        ax[4].set(xlabel="field distance to goal (m)", ylabel="recruitment lag (sampled trials)")
        for z, g in d.groupby("in_goal_zone"):
            ax[5].hist(g["lag"], bins=np.arange(0, d["lag"].max() + 2) - 0.5, histtype="step", density=True,
                       label="goal zone" if z else "elsewhere")
        ax[5].set(xlabel="recruitment lag", ylabel="density")
        ax[5].legend(frameon=False)
    return [("goal_coding", fig)]
