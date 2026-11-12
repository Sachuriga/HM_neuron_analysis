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

from .. import maze as mz, plotting as P, stats as S
from . import common

NAME = "field_count"
TITLE = "Number of fields per cell: distribution fits"
DEFAULT_QUERY = "day > 0"
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
                         aic=2 * p + 2 * r.fun, bic=p * np.log(n) + 2 * r.fun, params=list(r.x)))
    df = pd.DataFrame(rows)
    df["delta_aic"] = df["aic"] - df["aic"].min()
    df["delta_bic"] = df["bic"] - df["bic"].min()
    df["aic_weight"] = np.exp(-df["delta_aic"] / 2) / np.exp(-df["delta_aic"] / 2).sum()
    return df.sort_values("aic").reset_index(drop=True)


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
        U.append(R["units"].assign(session_id=common.session_id(row), valid_track_m=track))
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
    ])
    summary = pc.groupby(["rat"]).agg(n_cells=("unit", "size"), n_sessions=("session_id", "nunique"),
                                      mean=("n_fields", "mean"), median=("n_fields", "median"),
                                      var=("n_fields", "var")).reset_index()
    summary["fano"] = summary["var"] / summary["mean"]
    return dict(units=U, fits=fits, confounds=conf, summary=summary)


def plot(res, cfg):
    U = res["units"]
    pc = U[U["is_place"]]
    fits = res["fits"]
    fig, ax = P.figure(180, 60, ncols=3)
    k = pc["n_fields"].to_numpy()
    ks = np.arange(1, k.max() + 1)
    h = np.bincount(k, minlength=k.max() + 1)[1:] / len(k)
    ax[0].bar(ks, h, color="0.75", width=1.0, label="data")
    pooled = fits[fits["group"] == "pooled"]
    for r in pooled.itertuples():
        th = [float(v) for v in r.params.split(";")]
        ax[0].plot(ks, pmf(r.model, th, ks), label=f"{r.model} (ΔAIC {r.delta_aic:.0f})")
    ax[0].set(xlabel="fields per place cell", ylabel="fraction of cells", yscale="log",
              ylim=(0.5 / len(k), None))
    ax[0].legend(frameon=False)
    for rat, g in pc.groupby("rat"):
        kk = np.sort(g["n_fields"].to_numpy())
        ax[1].plot(kk, 1 - np.arange(len(kk)) / len(kk), drawstyle="steps-post",
                   color=P.RAT_COLORS.get(rat), label=rat)
    ax[1].set(xscale="log", yscale="log", xlabel="fields per cell", ylabel="P(N ≥ n)")
    ax[1].legend(frameon=False)
    for rat, g in pc.groupby("rat"):
        ax[2].scatter(g["mean_hz"], g["n_fields"], s=3, alpha=0.5, color=P.RAT_COLORS.get(rat))
    ax[2].set(xscale="log", xlabel="mean running rate (Hz)", ylabel="fields per cell")
    return [("field_count", fig)]
