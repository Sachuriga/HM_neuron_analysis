"""Statistics helpers.

With only two animals a random effect of rat cannot be estimated, so the
defaults are: report every effect per rat, and pool with a mixed model whose
random intercept is the SESSION (rat as a fixed effect). Units within a session
are therefore not treated as independent samples.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy import stats as st


def fdr_bh(p):
    p = np.asarray(p, float)
    out = np.full(p.shape, np.nan)
    ok = np.isfinite(p)
    if not ok.any():
        return out
    pv = p[ok]
    n = len(pv)
    order = np.argsort(pv)
    q = pv[order] * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    res = np.empty(n)
    res[order] = np.clip(q, 0, 1)
    out[ok] = res
    return out


def mixed(formula, df, group="session_id", re_formula=None):
    """Linear mixed model, random intercept per `group`. Returns a tidy coef table."""
    import statsmodels.formula.api as smf
    d = df.dropna(subset=[c for c in df.columns if c in formula]).copy()
    # a fixed factor with a single level (e.g. one rat selected) cannot be estimated
    for term in ("C(rat)",):
        if term in formula and d["rat"].nunique() < 2:
            formula = formula.replace(f" + {term}", "").replace(f"{term} + ", "")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if d[group].nunique() < 3:
            m = smf.ols(formula, d).fit()
        else:
            m = smf.mixedlm(formula, d, groups=d[group], re_formula=re_formula).fit(reml=True)
    t = pd.DataFrame({"coef": m.params, "se": m.bse, "z": m.tvalues, "p": m.pvalues})
    t.index.name = "term"
    t.attrs["n_obs"] = int(m.nobs)
    t.attrs["n_groups"] = int(d[group].nunique())
    t.attrs["formula"] = formula
    return t.reset_index()


def glm_poisson(formula, df, offset=None, cluster=None):
    """Poisson GLM, cluster-robust SEs by `cluster` column if given."""
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    d = df.copy()
    kw = {}
    if offset is not None:
        kw["offset"] = d[offset]
    m = smf.glm(formula, d, family=sm.families.Poisson(), **kw)
    use_cluster = cluster is not None and d[cluster].nunique() >= 3
    r = (m.fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(d[cluster])[0]})
         if use_cluster else m.fit())
    t = pd.DataFrame({"coef": r.params, "se": r.bse, "z": r.tvalues, "p": r.pvalues})
    t.index.name = "term"
    return t.reset_index()


def per_group(df, by, fn):
    """Apply `fn(sub_df) -> dict` per group and return a DataFrame."""
    rows = []
    for k, g in df.groupby(by):
        r = fn(g)
        if r is None:
            continue
        k = k if isinstance(k, tuple) else (k,)
        rows.append({**dict(zip([by] if isinstance(by, str) else by, k)), **r})
    return pd.DataFrame(rows)


def spearman(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 4:
        return dict(rho=np.nan, p=np.nan, n=int(m.sum()))
    r, p = st.spearmanr(x[m], y[m])
    return dict(rho=float(r), p=float(p), n=int(m.sum()))


def hier_bootstrap(df, value, levels=("rat", "session_id"), n=2000, stat=np.mean, seed=0):
    """Hierarchical bootstrap CI of `stat(value)`: resample sessions within rat, then rows."""
    rng = np.random.default_rng(seed)
    groups = {k: g[value].dropna().to_numpy() for k, g in df.groupby(list(levels))}
    by_rat = {}
    for (rat, ses), v in groups.items():
        by_rat.setdefault(rat, []).append(v)
    boots = np.empty(n)
    for i in range(n):
        vals = []
        for rat, sess in by_rat.items():
            for j in rng.integers(0, len(sess), len(sess)):
                v = sess[j]
                if len(v):
                    vals.append(v[rng.integers(0, len(v), len(v))])
        boots[i] = stat(np.concatenate(vals)) if vals else np.nan
    return dict(est=float(stat(df[value].dropna())), lo=float(np.nanpercentile(boots, 2.5)),
                hi=float(np.nanpercentile(boots, 97.5)))


# ----------------------------------------------------------------------------- circular
def circ_mean(ph):
    return float(np.angle(np.mean(np.exp(1j * ph))))


def mrl(ph):
    return float(np.abs(np.mean(np.exp(1j * ph)))) if len(ph) else np.nan


def rayleigh(ph):
    """Rayleigh test (Zar). Returns (z, p)."""
    n = len(ph)
    if n < 2:
        return np.nan, np.nan
    R = n * mrl(ph)
    z = R ** 2 / n
    p = np.exp(np.sqrt(1 + 4 * n + 4 * (n ** 2 - R ** 2)) - (1 + 2 * n))
    return float(z), float(min(max(p, 0.0), 1.0))
