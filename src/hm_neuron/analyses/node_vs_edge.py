"""9. Are there more place fields on nodes (intersections) or on edges (hedges)?

A field is assigned to the kind of its PEAK bin. Raw counts are compared with
two null expectations per session: proportional to the valid TRACK LENGTH of
nodes vs edges, and proportional to running OCCUPANCY. Observed/expected
ratios are tested per session (binomial) and pooled with a session-clustered
Poisson GLM.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as st

from .. import maze as mz, plotting as P, stats as S
from . import common

NAME = "node_vs_edge"
TITLE = "Node vs edge fields, normalised by track length and occupancy"
DEFAULT_QUERY = "repeat > 0"  # repeat 0 = habituation, not training


def run(sessions, cfg):
    lm = mz.maze_from_cfg(cfg)
    isn = lm.is_node()
    L = lm.bins["length"].to_numpy()
    rows, frows = [], []
    for row, s, R in common.iter_loaded(sessions, cfg):
        f = R["fields"]
        f = f[f["is_place"]] if len(f) else f
        v = R["valid"]
        n_node = int((f["peak_kind"] == "node").sum()) if len(f) else 0
        n_edge = int((f["peak_kind"] == "edge").sum()) if len(f) else 0
        n = n_node + n_edge
        len_node, len_edge = L[v & isn].sum(), L[v & ~isn].sum()
        occ_node, occ_edge = R["occ"][v & isn].sum(), R["occ"][v & ~isn].sum()
        q_len = len_node / (len_node + len_edge)
        q_occ = occ_node / (occ_node + occ_edge)
        rows.append(dict(session_id=common.session_id(row), rat=row["rat"], day=row["day"],
                         n_node=n_node, n_edge=n_edge, n=n, frac_node=n_node / max(n, 1),
                         exp_frac_len=q_len, exp_frac_occ=q_occ,
                         oe_len=(n_node / max(n, 1)) / q_len, oe_occ=(n_node / max(n, 1)) / q_occ,
                         p_len=st.binomtest(n_node, n, q_len).pvalue if n else np.nan,
                         p_occ=st.binomtest(n_node, n, q_occ).pvalue if n else np.nan,
                         density_node_per_m=n_node / len_node, density_edge_per_m=n_edge / len_edge))
        if len(f):
            frows.append(f[["unit", "peak_kind", "node_frac", "size_m"]].assign(
                session_id=common.session_id(row), rat=row["rat"]))
    ses = pd.DataFrame(rows)
    ses["q_len"] = S.fdr_bh(ses["p_len"])
    ses["q_occ"] = S.fdr_bh(ses["p_occ"])
    # pooled: counts per (session, kind) ~ kind with offset log(length) or log(occupancy)
    long = []
    for r in ses.itertuples():
        long += [dict(session_id=r.session_id, rat=r.rat, kind="node", n=r.n_node,
                      log_len=np.log(r.exp_frac_len), log_occ=np.log(r.exp_frac_occ)),
                 dict(session_id=r.session_id, rat=r.rat, kind="edge", n=r.n_edge,
                      log_len=np.log(1 - r.exp_frac_len), log_occ=np.log(1 - r.exp_frac_occ))]
    long = pd.DataFrame(long)
    long = long[long["session_id"].isin(ses.loc[ses["n"] > 0, "session_id"])]
    glm = []
    for off in ("log_len", "log_occ"):
        for grp, d in [("pooled", long)] + list(long.groupby("rat")):
            t = S.glm_poisson("n ~ C(kind, Treatment('edge')) + C(session_id)", d, offset=off,
                              cluster="session_id")
            t = t[t["term"].str.startswith("C(kind")].assign(offset=off, group=grp)
            t["rate_ratio_node_vs_edge"] = np.exp(t["coef"])
            glm.append(t)
    return dict(sessions=ses, glm=pd.concat(glm, ignore_index=True),
                fields=pd.concat(frows, ignore_index=True) if frows else pd.DataFrame())


def plot(res, cfg):
    ses = res["sessions"]
    fig, ax = P.figure(150, 55, ncols=3)
    for rat, g in ses.groupby("rat"):
        c = P.RAT_COLORS.get(rat)
        ax[0].plot(g["day"], g["frac_node"], "o-", color=c, ms=3, label=rat)
        ax[0].plot(g["day"], g["exp_frac_len"], ":", color=c)
        ax[0].plot(g["day"], g["exp_frac_occ"], "--", color=c)
    ax[0].set(xlabel="day", ylabel="fraction of fields on nodes",
              title="data (o), length null (:), occupancy null (--)")
    ax[0].legend(frameon=False)
    for i, col in enumerate(("oe_len", "oe_occ")):
        for rat, g in ses.groupby("rat"):
            x = i + (0.15 if rat == "Rat6" else -0.15) + np.random.default_rng(0).normal(0, 0.03, len(g))
            ax[1].scatter(x, g[col], s=6, color=P.RAT_COLORS.get(rat))
    ax[1].axhline(1, color="k", lw=0.6)
    ax[1].set(xticks=[0, 1], xticklabels=["vs length", "vs occupancy"], ylabel="node fields: obs / exp")
    f = res["fields"]
    if len(f):
        ax[2].hist(f["node_frac"], bins=20, color="0.6")
        ax[2].set(xlabel="fraction of field length on nodes", ylabel="fields")
    return [("node_vs_edge", fig)]
