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

from .. import maze as mz, plotting as P, stats as S, visits as V
from . import common

NAME = "field_density_usage"
TITLE = "Field density vs node/edge usage"
DEFAULT_QUERY = "day > 0"


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
        df = pd.DataFrame(dict(element=elem, length=L, occ=R["occ"], valid=R["valid"]))
        g = df.groupby("element").agg(length=("length", "sum"), occ_s=("occ", "sum"),
                                      valid_frac=("valid", "mean")).reset_index()
        g["visits"] = g["element"].map(nvis).fillna(0).astype(int)
        g["n_peaks"] = g["element"].map(npk).fillna(0).astype(int)
        g["kind"] = np.where(g["element"].str.startswith("N"), "node", "edge")
        g["n_place_cells"] = int(R["units"]["is_place"].sum())
        g = g.assign(session_id=common.session_id(row), rat=row["rat"], day=row["day"])
        rows.append(g)
    E = pd.concat(rows, ignore_index=True)
    E = E[(E["valid_frac"] > 0.5) & (E["visits"] > 0)].copy()
    E["density_per_m"] = E["n_peaks"] / E["length"]
    # normalise by number of place cells so sessions with more cells are comparable
    E["density_per_m_per_cell"] = E["density_per_m"] / E["n_place_cells"].clip(lower=1)
    E["log_visits"] = np.log(E["visits"])
    E["log_occ"] = np.log(E["occ_s"].clip(lower=1e-3))
    E["log_len"] = np.log(E["length"])
    glm = []
    for usage in ("log_visits", "log_occ"):
        for grp, d in [("pooled", E)] + list(E.groupby("rat")):
            t = S.glm_poisson(f"n_peaks ~ {usage} + C(kind) + C(session_id)", d,
                              offset="log_len", cluster="session_id")
            glm.append(t[t["term"].isin([usage, "C(kind)[T.node]"])].assign(usage=usage, group=grp))
    per_ses = S.per_group(E, ["rat", "session_id", "day"],
                          lambda g: S.spearman(g["visits"].to_numpy(float), g["density_per_m"].to_numpy()))
    return dict(elements=E, glm=pd.concat(glm, ignore_index=True), per_session=per_ses)


def plot(res, cfg):
    E = res["elements"]
    fig, ax = P.figure(150, 55, ncols=3)
    for k, g in E.groupby("kind"):
        ax[0].scatter(g["visits"], g["density_per_m_per_cell"], s=2, alpha=0.3, color=P.KIND_COLORS[k], label=k)
        q = pd.qcut(g["visits"], 8, duplicates="drop")
        m = g.groupby(q, observed=True).agg(x=("visits", "median"), y=("density_per_m_per_cell", "mean"))
        ax[0].plot(m["x"], m["y"], "-", color=P.KIND_COLORS[k], lw=1.5)
    ax[0].set(xscale="log", xlabel="visits per session", ylabel="field peaks / m / place cell")
    ax[0].legend(frameon=False)
    for k, g in E.groupby("kind"):
        q = pd.qcut(g["occ_s"], 8, duplicates="drop")
        m = g.groupby(q, observed=True).agg(x=("occ_s", "median"), y=("density_per_m_per_cell", "mean"))
        ax[1].plot(m["x"], m["y"], "o-", ms=3, color=P.KIND_COLORS[k], label=k)
    ax[1].set(xscale="log", xlabel="running occupancy (s)", ylabel="field peaks / m / place cell")
    ps = res["per_session"]
    for rat, g in ps.groupby("rat"):
        ax[2].plot(g["day"], g["rho"], "o-", ms=3, color=P.RAT_COLORS.get(rat), label=rat)
    ax[2].axhline(0, color="k", lw=0.6)
    ax[2].set(xlabel="day", ylabel="Spearman ρ (density vs visits)")
    ax[2].legend(frameon=False)
    return [("field_density_usage", fig)]
