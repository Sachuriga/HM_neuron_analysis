"""2. Field density vs how often a node / edge is used.

Per session and maze element: number of place-field peaks, element length,
number of visits (traversals), and running occupancy. Field density
(peaks/m) is related to usage with a Poisson GLM, ``n_peaks ~ log(visits) +
kind + session`` with offset log(length), cluster-robust SEs by session.
Elements with no valid bins are excluded (a field cannot be detected where the
animal never ran).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .. import fields as fl, maze as mz, plotting as P, stats as S, visits as V
from . import common

NAME = "field_density_usage"
TITLE = "Field density vs node/edge usage"
DEFAULT_QUERY = "repeat > 0"  # repeat 0 = habituation, not training


def run(sessions, cfg):
    lm = mz.maze_from_cfg(cfg)
    elem = lm.element_of_bin()
    L = lm.bins["length"].to_numpy()
    rows = []
    for row, s, R in common.iter_loaded(sessions, cfg):
        vis, _ = V.segment_visits(s, cfg, lm)
        nvis = vis.groupby("element").size() if len(vis) else pd.Series(dtype=int)
        f = R["fields"]
        f = f[f["is_place"]] if len(f) else f
        npk = f.groupby("peak_element").size() if len(f) else pd.Series(dtype=int)
        N = fl.null_fields(s, R, cfg, lm)
        df = pd.DataFrame(dict(element=elem, length=L, occ=R["occ"], valid=R["valid"], null=N["peaks"]))
        g = df.groupby("element").agg(length=("length", "sum"), occ_s=("occ", "sum"),
                                      valid_frac=("valid", "mean"), n_peaks_null=("null", "sum")).reset_index()
        g["n_null"] = N["n_null"]
        g["visits"] = g["element"].map(nvis).fillna(0).astype(int)
        g["n_peaks"] = g["element"].map(npk).fillna(0).astype(int)
        g["kind"] = np.where(g["element"].str.startswith("N"), "node", "edge")
        g["n_place_cells"] = int(R["units"]["is_place"].sum())
        g = g.assign(session_id=common.session_id(row), rat=row["rat"], day=row["day"],
                     session=common.gl_label(row["block"], row["day_in_block"]))
        rows.append(g)
    E = pd.concat(rows, ignore_index=True)
    E = E[(E["valid_frac"] > 0.5) & (E["visits"] > 0)].copy()
    E["density_per_m"] = E["n_peaks"] / E["length"]
    # normalise by number of place cells so sessions with more cells are comparable
    E["density_per_m_per_cell"] = E["density_per_m"] / E["n_place_cells"].clip(lower=1)
    E["log_visits"] = np.log(E["visits"])
    E["log_occ"] = np.log(E["occ_s"].clip(lower=1e-3))
    E["log_len"] = np.log(E["length"])
    # surrogate: peaks summed over n_null spike-shift maps per cell -> same units as the observed density
    E["null_density_per_m_per_cell"] = E["n_peaks_null"] / E["n_null"] / E["length"] / E["n_place_cells"].clip(lower=1)
    E["log_len_null"] = E["log_len"] + np.log(E["n_null"])
    E["log_null"] = np.log((E["n_peaks_null"] + 0.5) / E["n_null"] / E["length"])
    glm = []
    for usage in ("log_visits", "log_occ"):
        for grp, d in [("pooled", E)] + list(E.groupby("rat")):
            for data, y, off, extra in (("observed", "n_peaks", "log_len", ""),
                                        ("surrogate", "n_peaks_null", "log_len_null", ""),
                                        ("observed | surrogate", "n_peaks", "log_len", " + log_null")):
                t = S.glm_poisson(f"{y} ~ {usage}{extra} + C(kind) + C(session_id)", d,
                                  offset=off, cluster="session_id")
                glm.append(t[t["term"].isin([usage, "C(kind)[T.node]", "log_null"])]
                           .assign(usage=usage, group=grp, data=data))
    # robustness: restrict to well-sampled elements
    rob = []
    for lo in (1, 3, 5, 10):
        d = E[E["visits"] >= lo]
        for data, y, off in (("observed", "n_peaks", "log_len"), ("surrogate", "n_peaks_null", "log_len_null")):
            t = S.glm_poisson(f"{y} ~ log_visits + C(kind) + C(session_id)", d, offset=off, cluster="session_id")
            rob.append(t[t["term"] == "log_visits"].assign(min_visits=lo, n_obs=len(d), data=data))

    def rho(g):
        a = S.spearman(g["visits"].to_numpy(float), g["density_per_m"].to_numpy())
        b = S.spearman(g["visits"].to_numpy(float), g["null_density_per_m_per_cell"].to_numpy())
        return dict(**a, rho_null=b["rho"])
    per_ses = S.per_group(E, ["rat", "session_id", "day", "session"], rho)
    return dict(elements=E, glm=pd.concat(glm, ignore_index=True),
                robustness=pd.concat(rob, ignore_index=True), per_session=per_ses)


def _binned(g, x, y, n=8):
    q = pd.qcut(g[x], n, duplicates="drop")
    return g.groupby(q, observed=True).agg(x=(x, "median"), y=(y, "mean"), se=(y, "sem"))


def plot(res, cfg):
    E = res["elements"]
    fig, ax = P.figure(180, 58, ncols=3)
    for j, (x, lab) in enumerate((("visits", "visits per session"), ("occ_s", "running occupancy (s)"))):
        for k, g in E.groupby("kind"):
            c = P.KIND_COLORS[k]
            if j == 0:
                ax[j].scatter(g[x], g["density_per_m_per_cell"], s=1.5, alpha=0.12, color=c, linewidths=0,
                              rasterized=True)
            m = _binned(g, x, "density_per_m_per_cell")
            ax[j].errorbar(m["x"], m["y"], m["se"], fmt="o-", ms=2.5, color=c, label=k)
        ax[j].set(xscale="log", xlabel=lab, ylabel="field peaks / m / place cell")
    ax[0].set(ylim=(0, E["density_per_m_per_cell"].quantile(0.99)))
    ax[1].legend(frameon=False)
    ps = res["per_session"]
    for rat, g in ps.groupby("rat"):
        c = P.RAT_COLORS.get(rat)
        ax[2].plot(g["day"], g["rho"], "o-", ms=3, color=c, label=rat)
    ax[2].axhline(0, color="k", lw=0.6)
    ax[2].set(ylabel="Spearman ρ (density vs visits)")
    lab = ps.drop_duplicates("day").sort_values("day")
    common.gl_xticks(ax[2], lab["day"].astype(int), lab["session"])
    ax[2].legend(frameon=False)
    return [("field_density_usage", fig)]


# ----------------------------------------------------------------------------- LaTeX report
CAPTIONS = {"field_density_usage": (
    r"\textbf{Field density vs usage.} Left: field peaks per metre per place cell against the "
    r"number of visits; dots are single maze elements in single sessions, markers are means $\pm$ s.e.m.\ "
    r"in visit octiles. Middle: the same against running "
    r"occupancy. Right: within-session Spearman correlation between field density and visits, per session. " + common.GL_NOTE)}


def latex(T):
    from .. import report as R
    E, glm, ps = T["elements"], T["glm"], T["per_session"]
    k = E.groupby("kind").agg(n=("element", "size"), visits=("visits", "median"), occ_s=("occ_s", "median"),
                              n_peaks=("n_peaks", "sum"), length_m=("length", "sum")).reset_index()
    k["peaks_per_m"] = k["n_peaks"] / k["length_m"]
    text = (rf"{len(E):,} element$\times$session observations ({E['session_id'].nunique()} sessions; "
            rf"elements with $>50\%$ valid bins and at least one visit). "
            rf"The within-session Spearman correlation between field density and visits was "
            rf"{ps['rho'].median():.2f} (median; positive in {int((ps['rho'] > 0).sum())} of "
            rf"{int(ps['rho'].notna().sum())} sessions); for spike-shift surrogates of the same cells it was "
            rf"{ps['rho_null'].median():.2f}." + "\n\n").replace(",", "{,}", 1)
    gp = glm[(glm["group"] == "pooled") & (glm["usage"] == "log_visits") & (glm["term"] == "log_visits")]
    gp = gp.set_index("data")
    b = lambda d: rf"${gp.at[d, 'coef']:.2f} \pm {gp.at[d, 'se']:.2f}$"
    text += (rf"In the Poisson GLM the slope of field count on $\ln$(visits) is {b('observed')} for the "
             rf"observed fields and {b('surrogate')} for the surrogates "
             rf"(Table~\ref{{tab:field_density_usage:glm}}). ")
    if gp.at["surrogate", "coef"] < gp.at["observed", "coef"] < 0:
        text += (r"The surrogate slope is the steeper of the two: rarely visited elements have noisier rate "
                 r"estimates and therefore more spurious peaks, so the negative relation is expected from the "
                 r"detector alone. With the element's surrogate density as a covariate the "
                 rf"observed slope is {b('observed | surrogate')}. Relative to that noise floor, fields are "
                 r"thus not depleted in heavily used elements.")
    text += "\n\n"
    text += R.table(k, "Maze elements entering the analysis (summed over sessions).",
                    "tab:field_density_usage:elements",
                    headers=dict(n="observations", visits="median visits", occ_s="median occupancy (s)",
                                 n_peaks="field peaks", length_m="track (m)", peaks_per_m="peaks / m"))
    g = glm.assign(usage=glm["usage"].map(dict(log_visits="ln visits", log_occ="ln occupancy")),
                   term=glm["term"].map(lambda t: "node vs edge" if t.startswith("C(kind)") else
                                        "ln surrogate density" if t == "log_null" else "usage"))
    g = g[g["term"] != "node vs edge"]
    text += R.table(g, cols=["usage", "group", "data", "term", "coef", "se", "z", "p"],
                    caption=r"Poisson GLM, \texttt{n\_peaks $\sim$ usage + kind + session}, offset $\ln$(length), "
                    r"session-clustered standard errors. A usage coefficient of 1 means field count is "
                    r"proportional to usage; 0 means no dependence. \emph{surrogate}: the same model on "
                    r"spike-shift surrogate fields (detector bias only). \emph{observed | surrogate}: observed "
                    r"peaks with the element's surrogate density as an additional covariate.",
                    label="tab:field_density_usage:glm", size=r"\footnotesize")
    text += R.table(T["robustness"], cols=["min_visits", "n_obs", "data", "coef", "se", "z", "p"],
                    caption=r"Slope on $\ln$(visits) when only elements with at least \texttt{min\_visits} "
                    r"visits are kept.", label="tab:field_density_usage:robust",
                    headers=dict(min_visits=r"min.\ visits", n_obs="observations"), size=r"\footnotesize")
    return text
