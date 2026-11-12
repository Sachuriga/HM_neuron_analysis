"""Helpers shared by analyses."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .. import fields as fl
from .. import io
from .. import maze as mz


def session_id(row) -> str:
    return f"{row['rat']}_{row['date']}"


def iter_loaded(sessions: pd.DataFrame, cfg, with_fields=True, verbose=True):
    """Yield (catalog_row, Session, field_result_or_None)."""
    for _, row in sessions.iterrows():
        if verbose:
            print(f"  {row['rat']} {row['date']} (day {row['day']}, goal {row['goal']})", flush=True)
        s = io.load_session(cfg, row["rat"], str(row["date"]))
        F = fl.session_fields(s, cfg) if with_fields else None
        yield row, s, F


def gather(sessions, cfg, verbose=True):
    """Concatenated unit and field tables across sessions (adds session_id)."""
    U, F = [], []
    for row, s, R in iter_loaded(sessions, cfg, verbose=verbose):
        sid = session_id(row)
        U.append(R["units"].assign(session_id=sid))
        if len(R["fields"]):
            F.append(R["fields"].drop(columns=["bins"]).assign(session_id=sid))
    return (pd.concat(U, ignore_index=True) if U else pd.DataFrame(),
            pd.concat(F, ignore_index=True) if F else pd.DataFrame())


def valid_track(R, lm: mz.LinearMaze):
    """Track length (m) of valid bins split by kind."""
    L = lm.bins["length"].to_numpy()
    isn = lm.is_node()
    v = R["valid"]
    return dict(node=float(L[v & isn].sum()), edge=float(L[v & ~isn].sum()))


def goal_null_distances(R, lm, goal, weight="length"):
    """Distances to goal of valid bins with their weights (track length or occupancy)."""
    v = R["valid"]
    d = lm.dist_to_node(goal)[v]
    w = (lm.bins["length"].to_numpy() if weight == "length" else R["occ"])[v]
    return d, w / w.sum()


def weighted_quantile(x, w, q):
    o = np.argsort(x)
    cw = np.cumsum(w[o])
    return np.interp(q, cw / cw[-1], x[o])
