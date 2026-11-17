"""1. Distribution of the number of place fields per cell.

Fields per place cell are counts >= 1, so the candidates are fitted as
zero-truncated DISCRETE models by maximum likelihood:

* Poisson (ZT)                – fields placed independently, same propensity for all cells
* negative binomial (ZT)      – gamma-Poisson: cells differ in propensity (Rich et al. 2014)
* geometric (ZT)
* discretised log-normal      – P(k) ∝ Φ((ln(k+½)−μ)/σ) − Φ((ln(k−½)−μ)/σ), k ≥ 1
* discretised Gaussian

Compared by AIC/BIC, pooled and per rat. Because field count scales with how
much track a cell sees and how much it fires, it is also reported per metre of
valid track and against mean rate.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import optimize, special, stats as st

from .. import fields as fl, maze as mz, plotting as P, stats as S
from . import common

NAME = "field_count"
TITLE = "Number of fields per cell: distribution fits"
DEFAULT_QUERY = "repeat > 0"  # repeat 0 = habituation, not training
KMAX = 400


def _ll_factory(name):
    k_all = np.arange(1, KMAX + 1)

    def norm_trunc(logp_all):
        # normalise over k >= 1 (zero-truncation / discretisation support)
        return logp_all - special.logsumexp(logp_all)

    if name == "poisson":
        def ll(th, k):
            lam = np.exp(th[0])
            lp = k * np.log(lam) - lam - special.gammaln(k + 1) - np.log1p(-np.exp(-lam))
            return lp.sum()
        return ll, [np.log(3.0)]
    if name == "negbin":
        def ll(th, k):
            r, p = np.exp(th[0]), special.expit(th[1])
            lp = st.nbinom.logpmf(k, r, p) - np.log1p(-st.nbinom.pmf(0, r, p))
            return lp.sum()
        return ll, [0.0, 0.0]
    if name == "geometric":
        def ll(th, k):
            p = special.expit(th[0])
            return (np.log(p) + (k - 1) * np.log1p(-p)).sum()
        return ll, [0.0]
    if name == "lognormal":
        def ll(th, k):
            mu, sg = th[0], np.exp(th[1])
            cdf = lambda v: st.norm.cdf((np.log(v) - mu) / sg)
            lp_all = np.log(np.clip(cdf(k_all + 0.5) - cdf(np.maximum(k_all - 0.5, 1e-9)), 1e-300, None))
            lp_all = norm_trunc(lp_all)
            return lp_all[k - 1].sum()
        return ll, [1.0, 0.0]
    if name == "gaussian":
        def ll(th, k):
            mu, sg = th[0], np.exp(th[1])
            lp_all = np.log(np.clip(st.norm.cdf(k_all + 0.5, mu, sg) - st.norm.cdf(k_all - 0.5, mu, sg), 1e-300, None))
            return norm_trunc(lp_all)[k - 1].sum()
        return ll, [5.0, 1.0]
    raise KeyError(name)


MODELS = ("poisson", "negbin", "geometric", "lognormal", "gaussian")


def fit(k: np.ndarray) -> pd.DataFrame:
    k = np.asarray(k, int)
    k = k[(k >= 1) & (k <= KMAX)]
    rows = []
    for m in MODELS:
        ll, x0 = _ll_factory(m)
        if m == "gaussian":
            x0 = [k.mean(), np.log(k.std() + 1e-3)]
        if m == "lognormal":
            x0 = [np.log(k).mean(), np.log(np.log(k).std() + 0.1)]
        r = optimize.minimize(lambda th: -ll(th, k), x0, method="Nelder-Mead",
                              options=dict(maxiter=4000, xatol=1e-6, fatol=1e-8))
        n, p = len(k), len(x0)
        rows.append(dict(model=m, loglik=-r.fun, n_params=p, n=n,
                         aic=2 * p + 2 * r.fun, bic=p * np.log(n) + 2 * r.fun, params=list(r.x),
                         **gof(m, r.x, k)))
    df = pd.DataFrame(rows)
    df["delta_aic"] = df["aic"] - df["aic"].min()
    df["delta_bic"] = df["bic"] - df["bic"].min()
    df["aic_weight"] = np.exp(-df["delta_aic"] / 2) / np.exp(-df["delta_aic"] / 2).sum()
    return df.sort_values("aic").reset_index(drop=True)


def gof(model, params, k, min_expected=5.0):
    """Pearson chi-square goodness of fit; neighbouring counts are pooled until expected >= 5."""
    k = np.asarray(k, int)
    ks = np.arange(1, KMAX + 1)
    obs = np.bincount(k, minlength=KMAX + 1)[1:].astype(float)
    exp = len(k) * pmf(model, params, ks)
    O, E, o, e = [], [], 0.0, 0.0
    for oi, ei in zip(obs, exp):
        o, e = o + oi, e + ei
        if e >= min_expected:
            O.append(o); E.append(e); o = e = 0.0
    if O and (o or e):
        O[-1] += o; E[-1] += e
    O, E = np.array(O), np.array(E)
    dof = len(O) - 1 - len(params)
    if dof < 1:
        return dict(gof_chi2=np.nan, gof_dof=dof, gof_p=np.nan)
    chi2 = float(((O - E) ** 2 / E).sum())
    return dict(gof_chi2=chi2, gof_dof=int(dof), gof_p=float(st.chi2.sf(chi2, dof)))


def pmf(model, params, ks):
    ks = np.asarray(ks)
    k_all = np.arange(1, KMAX + 1)
    th = params
    if model == "poisson":
        lam = np.exp(th[0]); p = st.poisson.pmf(k_all, lam)
    elif model == "negbin":
        p = st.nbinom.pmf(k_all, np.exp(th[0]), special.expit(th[1]))
    elif model == "geometric":
        q = special.expit(th[0]); p = q * (1 - q) ** (k_all - 1)
    elif model == "lognormal":
        c = lambda v: st.norm.cdf((np.log(v) - th[0]) / np.exp(th[1]))
        p = c(k_all + 0.5) - c(np.maximum(k_all - 0.5, 1e-9))
    else:
        p = st.norm.cdf(k_all + 0.5, th[0], np.exp(th[1])) - st.norm.cdf(k_all - 0.5, th[0], np.exp(th[1]))
    p = p / p.sum()
    return p[ks - 1]


def run(sessions, cfg):
    lm = mz.maze_from_cfg(cfg)
    U = []
    for row, s, R in common.iter_loaded(sessions, cfg):
        track = sum(common.valid_track(R, lm).values())
        N = fl.null_fields(s, R, cfg, lm)
        u = R["units"].assign(session_id=common.session_id(row), valid_track_m=track)
        u["n_fields_null"] = u["unit"].map(dict(zip(N["units"], N["n_fields"].mean(1))))
        u["n_fields_null_1"] = u["unit"].map(dict(zip(N["units"], N["n_fields"][:, 0])))
        U.append(u)
    U = pd.concat(U, ignore_index=True)
    pc = U[U["is_place"]].copy()
    pc["fields_per_10m"] = pc["n_fields"] / pc["valid_track_m"] * 10
    fits = [fit(pc["n_fields"]).assign(group="pooled")]
    for rat, g in pc.groupby("rat"):
        fits.append(fit(g["n_fields"]).assign(group=rat))
    fits = pd.concat(fits, ignore_index=True)
    fits["params"] = fits["params"].apply(lambda v: ";".join(f"{x:.4g}" for x in v))
    conf = pd.DataFrame([
        dict(test="n_fields ~ mean_hz (Spearman)", **S.spearman(pc["mean_hz"].to_numpy(), pc["n_fields"].to_numpy())),
        dict(test="n_fields ~ day (Spearman)", **S.spearman(pc["day"].to_numpy(float), pc["n_fields"].to_numpy(float))),
        dict(test="n_fields ~ surrogate n_fields (Spearman)",
             **S.spearman(pc["n_fields_null"].to_numpy(float), pc["n_fields"].to_numpy(float))),
        dict(test="surrogate n_fields ~ mean_hz (Spearman)",
             **S.spearman(pc["mean_hz"].to_numpy(), pc["n_fields_null"].to_numpy(float))),
    ])
    summary = pc.groupby(["rat"]).agg(n_cells=("unit", "size"), n_sessions=("session_id", "nunique"),
                                      mean=("n_fields", "mean"), median=("n_fields", "median"),
                                      var=("n_fields", "var"), null_mean=("n_fields_null", "mean"),
                                      null_median=("n_fields_null_1", "median")).reset_index()
    summary["fano"] = summary["var"] / summary["mean"]
    summary["frac_above_null"] = [float((g["n_fields"] > g["n_fields_null"]).mean())
                                  for _, g in pc.groupby("rat")]
    return dict(units=U, fits=fits, confounds=conf, summary=summary, per_session=_per_session(pc))


def _per_session(pc):
    """One row per session: distribution of field count over its place cells, and which model fits it best."""
    rows = []
    for (rat, sid), g in pc.groupby(["rat", "session_id"], sort=False):
        k = g["n_fields"].to_numpy()
        row = dict(rat=rat, session_id=sid, day=int(g["day"].iloc[0]), goal=int(g["goal"].iloc[0]),
                   n_place=len(k), mean=k.mean(), median=np.median(k), q25=np.percentile(k, 25),
                   q75=np.percentile(k, 75), max=int(k.max()),
                   surrogate_median=float(g["n_fields_null_1"].median()),
                   frac_above_null=float((g["n_fields"] > g["n_fields_null"]).mean()))
        if len(k) >= 10:
            f = fit(k).set_index("model")
            row.update(best_model=f["aic"].idxmin(), daic_lognormal=f.at["lognormal", "delta_aic"],
                       daic_gaussian=f.at["gaussian", "delta_aic"], daic_negbin=f.at["negbin", "delta_aic"],
                       lognormal_mu=f.at["lognormal", "params"][0],
                       lognormal_sigma=float(np.exp(f.at["lognormal", "params"][1])))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["rat", "day"]).reset_index(drop=True)


def _plot_sessions(pc, fits, ncols=7, bin_w=3):
    """Histogram of fields per place cell for every session (rows of panels per rat, one panel per day)."""
    rats = sorted(pc["rat"].unique())
    days = np.arange(int(pc["day"].min()), int(pc["day"].max()) + 1)
    nrow = int(np.ceil(len(days) / ncols))
    kmax = int(pc["n_fields"].max())
    edges = np.arange(0.5, kmax + bin_w + 0.5, bin_w)
    mid = (edges[:-1] + edges[1:]) / 2
    ks = np.arange(1, int(edges[-1]) + 1)
    fig, axs = P.figure(180, 24 * nrow * len(rats) + 8, nrows=nrow * len(rats), ncols=ncols,
                        sharex=True, sharey=True, squeeze=False)
    for ri, rat in enumerate(rats):
        g = pc[pc["rat"] == rat]
        fr = fits[(fits["group"] == rat) & (fits["model"] == "lognormal")]
        ref = None
        if len(fr):
            p = pmf("lognormal", [float(v) for v in fr["params"].iloc[0].split(";")], ks)
            ref = np.histogram(ks, edges, weights=p)[0]
        for j in range(nrow * ncols):
            ax = axs[ri * nrow + j // ncols, j % ncols]
            if j >= len(days):
                ax.axis("off")
                continue
            d = g[g["day"] == days[j]]
            if d.empty:
                ax.text(0.5, 0.5, f"day {days[j]}\nno usable session", transform=ax.transAxes,
                        ha="center", va="center", fontsize=5, color="0.4")
                continue
            k = d["n_fields"].to_numpy()
            ax.bar(mid, np.histogram(k, edges)[0] / len(k), width=bin_w, color=P.RAT_COLORS.get(rat),
                   alpha=0.55, label="observed")
            kn = d["n_fields_null_1"].dropna().to_numpy()
            ax.step(edges, np.r_[np.histogram(kn, edges)[0] / max(len(kn), 1), 0], where="post",
                    color="k", lw=0.6, label="surrogate")
            if ref is not None:
                ax.plot(mid, ref, color="0.35", lw=0.6, ls="--", label=f"log-normal fit, all {rat} sessions")
            ax.axvline(np.median(k), color=P.RAT_COLORS.get(rat), lw=0.8)
            ax.set_title(f"day {days[j]}: n = {len(k)}, median {np.median(k):.0f}", fontsize=5.5, pad=2)
            handles = ax.get_legend_handles_labels()
        # legend in the first unused panel of this rat's block (or the last panel if the grid is full)
        spare = len(days) if len(days) < nrow * ncols else nrow * ncols - 1
        axs[ri * nrow + spare // ncols, spare % ncols].legend(*handles, frameon=False, fontsize=5,
                                                              loc="center", handlelength=1.2, title=rat)
        for r in range(nrow):
            axs[ri * nrow + r, 0].set_ylabel(f"{rat}\nfraction of cells")
    for a in axs[-1]:
        a.set_xlabel("fields per cell")
    return fig


def _plot_days(pc):
    """Box plot of fields per place cell by training day, one panel per rat."""
    rats = sorted(pc["rat"].unique())
    fig, axs = P.figure(180, 55, ncols=len(rats), sharey=True, squeeze=False)
    for ax, rat in zip(axs[0], rats):
        g = pc[pc["rat"] == rat]
        days = sorted(g["day"].unique())
        c = P.RAT_COLORS.get(rat)
        ax.boxplot([g.loc[g["day"] == d, "n_fields"].to_numpy() for d in days], positions=days, widths=0.6,
                   patch_artist=True, showfliers=True,
                   boxprops=dict(facecolor=c, alpha=0.45, lw=0.5), medianprops=dict(color="k", lw=1.0),
                   whiskerprops=dict(lw=0.5), capprops=dict(lw=0.5),
                   flierprops=dict(marker=".", ms=1.5, mfc=c, mec="none", alpha=0.6))
        m = g.groupby("day")["n_fields_null_1"].median()
        ax.plot(m.index, m.values, "s--", ms=2.5, mfc="w", color="k", lw=0.7, label="surrogate median")
        for d in g.loc[g["is_switch_day"].astype(bool), "day"].unique():
            ax.axvline(d - 0.5, color="0.6", lw=0.5, ls=":")
        ax.set(yscale="log", xlabel="training day", title=rat)
        ax.set_xticks(days[::2]); ax.set_xticklabels(days[::2])
        ax.legend(frameon=False)
    axs[0, 0].set_ylabel("fields per place cell")
    return fig


MODEL_COLORS = {"lognormal": "#0072b2", "gaussian": "#d55e00", "negbin": "#009e73",
                "geometric": "#999999", "poisson": "#cc79a7"}


def plot(res, cfg):
    U = res["units"]
    pc = U[U["is_place"]]
    fits = res["fits"]
    fig, axs = P.figure(180, 110, nrows=2, ncols=2)
    ax = axs.ravel()
    k = pc["n_fields"].to_numpy()
    ks = np.arange(1, k.max() + 1)
    h = np.bincount(k, minlength=k.max() + 1)[1:] / len(k)
    pooled = fits[fits["group"] == "pooled"]
    for a in ax[:2]:
        a.bar(ks, h, color="0.8", width=1.0, label=f"data (n = {len(k)})")
        for r in pooled.itertuples():
            th = [float(v) for v in r.params.split(";")]
            a.plot(ks, pmf(r.model, th, ks), color=MODEL_COLORS[r.model],
                   label=f"{r.model} (ΔAIC {r.delta_aic:.0f})")
        kn = pc["n_fields_null_1"].dropna().to_numpy(int)
        hn = np.bincount(kn, minlength=k.max() + 2)[:k.max() + 2] / len(kn)
        a.step(np.arange(len(hn)), hn, where="mid", color="k", lw=0.7, label="spike-shift surrogate")
        a.set(xlabel="fields per place cell", ylabel="fraction of cells")
    ax[0].set(ylim=(0, h.max() * 1.25), title="linear scale")
    ax[0].legend(frameon=False)
    ax[1].set(yscale="log", ylim=(0.5 / len(k), 1), title="log scale")
    # survival: data per rat + pooled log-normal / Gaussian
    for rat, g in pc.groupby("rat"):
        kk = np.sort(g["n_fields"].to_numpy())
        ax[2].plot(kk, 1 - np.arange(len(kk)) / len(kk), drawstyle="steps-post",
                   color=P.RAT_COLORS.get(rat), label=rat)
    for r in pooled.itertuples():
        if r.model in ("lognormal", "gaussian"):
            th = [float(v) for v in r.params.split(";")]
            sf = 1 - np.concatenate([[0], np.cumsum(pmf(r.model, th, ks))[:-1]])
            ax[2].plot(ks, sf, "--", lw=0.8, color=MODEL_COLORS[r.model], label=f"{r.model} fit (pooled)")
    ax[2].set(xscale="log", yscale="log", xlabel="fields per cell, n", ylabel="P(N ≥ n)",
              ylim=(0.5 / len(k), 1.5))
    ax[2].legend(frameon=False)
    for rat, g in pc.groupby("rat"):
        ax[3].scatter(g["mean_hz"], g["n_fields"], s=3, alpha=0.4, color=P.RAT_COLORS.get(rat),
                      label=rat, linewidths=0)
    q = pd.qcut(pc["mean_hz"], 12, duplicates="drop")
    m = pc.groupby(q, observed=True).agg(x=("mean_hz", "median"), y=("n_fields_null", "mean"))
    ax[3].plot(m["x"], m["y"], "k-", lw=1.0, label="surrogate (mean)")
    ax[3].set(xscale="log", yscale="log", xlabel="mean running rate (Hz)", ylabel="fields per cell")
    ax[3].legend(frameon=False)
    return [("field_count", fig), ("field_count_by_session", _plot_sessions(pc, fits)),
            ("field_count_by_day", _plot_days(pc))]


# ----------------------------------------------------------------------------- LaTeX report
MODEL_LABELS = {"poisson": "Poisson (ZT)", "negbin": "negative binomial (ZT)", "geometric": "geometric",
                "lognormal": "log-normal (discretised)", "gaussian": "Gaussian (discretised)"}

CAPTIONS = {"field_count": (
    r"\textbf{Fields per place cell.} Top: distribution over all place cells (bars) with the "
    r"maximum-likelihood fit of each candidate model, on a linear (left) and logarithmic (right) "
    r"$y$-axis; $\Delta$AIC relative to the best model in the legend. Bottom left: survival function "
    r"$P(N\geq n)$ per rat on log--log axes with the pooled log-normal and Gaussian fits (dashed). "
    r"Bottom right: field count against mean running rate, one dot per place cell. "
    r"Black: the same cells after circularly shifting their spikes in time (spatial tuning destroyed, "
    r"rate and maze sampling preserved), i.e.\ the noise floor of the field detector."),
    "field_count_by_session": (
    r"\textbf{Fields per place cell in every session.} One panel per session (rat and training day in the "
    r"title, with the number of place cells and their median field count). Bars: fraction of the session's "
    r"place cells in bins of three fields; vertical line: median. Black steps: the same cells after a "
    r"circular spike shift (surrogate). Dashed: log-normal fitted to all sessions of that rat, for reference. "
    r"All panels share both axes."),
    "field_count_by_day": (
    r"\textbf{Fields per place cell across training.} Box plots (median, quartiles, 1.5 IQR whiskers, "
    r"outliers as dots) of the field count of the place cells recorded on each training day, per rat; "
    r"log $y$-axis. Open squares: median of the spike-shift surrogates. Dotted vertical lines: first day "
    r"with a new goal.")}


def latex(T):
    from .. import report as R
    fits, summ, conf = T["fits"], T["summary"], T["confounds"]
    U = T["units"]
    pc = U[U["is_place"]]
    out = [rf"{len(pc):,} of {len(U):,} good pyramidal units were place cells "
           rf"({pc['session_id'].nunique()} sessions, {pc['rat'].nunique()} rats); they had "
           rf"{pc['n_fields'].mean():.2f} fields on average (median {pc['n_fields'].median():.0f}, "
           rf"range {pc['n_fields'].min()}--{pc['n_fields'].max()}; Table~\ref{{tab:field_count:summary}})."
           .replace(",", "{,}", 2)]
    nm, om = pc["n_fields_null"].mean(), pc["n_fields"].mean()
    above = (pc["n_fields"] > pc["n_fields_null"]).mean()
    out[0] += (r" \textbf{Detector noise floor:} spike-shift surrogates of the same cells (spatial tuning "
               rf"destroyed, rate and sampling preserved) yield {nm:.2f} fields per cell, against {om:.2f} "
               rf"observed; {100 * above:.0f}\% of cells have more fields than their own surrogates.")
    if nm >= 0.5 * om:
        out[0] += (" Under the present detection criteria the field count is therefore dominated by firing "
                   "rate and sampling noise rather than by the number of genuine place fields, and the shape "
                   "of the distribution below should be read with that in mind.")
    for grp in ["pooled"] + sorted(g for g in fits["group"].unique() if g != "pooled"):
        f = fits[fits["group"] == grp].set_index("model")
        best = f["aic"].idxmin()
        others = ", ".join(rf"{MODEL_LABELS[m]} $\Delta$AIC $= {f.at[m, 'delta_aic']:.1f}$"
                           for m in ("lognormal", "gaussian", "negbin", "poisson") if m != best)
        out.append(rf"\textbf{{{R.esc(grp)}}}: best model by AIC is the {MODEL_LABELS[best]} "
                   rf"(Akaike weight {f.at[best, 'aic_weight']:.2f}); {others}.")
    pooled = fits[fits["group"] == "pooled"].sort_values("aic")
    mu = float(str(pooled.set_index("model").at["gaussian", "params"]).split(";")[0])
    if mu < 1:
        out.append(rf"Note on the Gaussian: its maximum-likelihood mean lies below the support "
                   rf"($\mu = {mu:.0f}$ fields), so only the upper tail of the bell is used and the fit "
                   r"is close to a geometric decay rather than a symmetric distribution around a typical count.")
    b = pooled.iloc[0]
    verdict = ("is not rejected" if b["gof_p"] >= 0.05 else
               "is itself rejected, so it is the best of the candidates rather than a complete description")
    p_txt = R.fmt(float(b["gof_p"])).strip("$")
    out.append(rf"Goodness of fit: the best pooled model ({MODEL_LABELS[b['model']]}) {verdict} "
               rf"(Pearson $\chi^2_{{{int(b['gof_dof'])}}} = {b['gof_chi2']:.1f}$, $p = {p_txt}$; "
               r"Table~\ref{tab:field_count:fits}).")
    text = "\n\n".join(out[:1]) + "\n\n" + r"\begin{itemize}" + "\n" + "\n".join(
        r"\item " + o for o in out[1:]) + "\n" + r"\end{itemize}" + "\n\n"
    text += R.table(summ, "Fields per place cell, by rat. Fano = variance/mean ($>1$: over-dispersed "
                    "relative to Poisson). Surrogate = same cells with circularly shifted spikes.", "tab:field_count:summary",
                    headers=dict(n_cells="place cells", n_sessions="sessions", var="variance", fano="Fano",
                                 null_mean="surrogate mean", null_median="surrogate median",
                                 frac_above_null=r"frac.\ $>$ surrogate"), size=r"\footnotesize")
    ps = T.get("per_session")
    if ps is not None and len(ps):
        fitted = ps.dropna(subset=["best_model"])
        n_ln = int((fitted["best_model"] == "lognormal").sum())
        others = fitted.loc[fitted["best_model"] != "lognormal", "best_model"].value_counts()
        rest = ("; the others are best fitted by " + ", ".join(
            f"{MODEL_LABELS[m]} ({n})" for m, n in others.items())) if len(others) else ""
        text += (rf"\paragraph{{Per session.}} Session medians range from {ps['median'].min():.0f} to "
                 rf"{ps['median'].max():.0f} fields per cell (Fig.~\ref{{fig:field_count_by_session}}, "
                 rf"Fig.~\ref{{fig:field_count_by_day}}, Table~\ref{{tab:field_count:sessions}}). Fitted to "
                 rf"single sessions, the log-normal has the lowest AIC in {n_ln} of {len(fitted)} sessions{rest}. "
                 r"With 30--85 cells per session these single-session comparisons have little power "
                 rf"(median $\Delta$AIC of the Gaussian over the log-normal: "
                 rf"{(fitted['daic_gaussian'] - fitted['daic_lognormal']).median():.1f})." + "\n\n")
        t = ps.assign(best_model=ps["best_model"].map(
            dict(lognormal="log-normal", gaussian="Gaussian", negbin="neg. binomial",
                 geometric="geometric", poisson="Poisson")).fillna("--"))
        text += R.table(t, "Fields per place cell in each session. IQR bounds are the 25th and 75th "
                        r"percentiles; $\Delta$AIC columns are relative to the best model of that session.",
                        "tab:field_count:sessions",
                        cols=["rat", "day", "goal", "n_place", "median", "q25", "q75", "max",
                              "surrogate_median", "best_model", "daic_lognormal", "daic_gaussian"],
                        headers=dict(n_place="place cells", q25="25\\%", q75="75\\%",
                                     surrogate_median="surrogate median", best_model="best model",
                                     daic_lognormal=r"$\Delta$AIC log-normal",
                                     daic_gaussian=r"$\Delta$AIC Gaussian"), size=r"\scriptsize")
    f = fits.assign(model=fits["model"].map(MODEL_LABELS))
    text += R.table(f, r"Maximum-likelihood fits to the number of fields per place cell ($k\geq1$). "
                    r"Models are sorted by AIC within each group. "
                    r"$\chi^2$: Pearson goodness of fit with counts pooled to an expected frequency $\geq5$.",
                    "tab:field_count:fits",
                    cols=["group", "model", "n", "n_params", "loglik", "delta_aic", "delta_bic", "aic_weight",
                          "gof_chi2", "gof_dof", "gof_p"],
                    headers=dict(n="$n$", n_params="params", loglik=r"$\ln L$",
                                 delta_aic=r"$\Delta$AIC", delta_bic=r"$\Delta$BIC", aic_weight="weight",
                                 gof_chi2=r"$\chi^2$", gof_dof="dof", gof_p=r"$p_{\mathrm{GOF}}$"),
                    digits=4, size=r"\footnotesize")
    text += R.table(conf, "Dependence of field count on firing rate and training day (Spearman, place cells).",
                    "tab:field_count:confounds")
    return text
