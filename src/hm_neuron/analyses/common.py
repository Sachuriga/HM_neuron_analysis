"""Helpers shared by analyses."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .. import fields as fl
from .. import io
from .. import maze as mz


def session_id(row) -> str:
    return f"{row['rat']}_{row['date']}"


MAX_ALIGN_ERROR_M = 0.25


def session_ok(s):
    """Usable session: trial times aligned to the tracking and some in-trial data."""
    err = s.steps.attrs.get("align_error_m", np.nan)
    if not np.isfinite(err) or err > MAX_ALIGN_ERROR_M:
        return False, f"trial log does not align with tracking (step-end error {err})"
    if s.in_trial.sum() < 100:
        return False, "no in-trial samples"
    return True, ""


def iter_loaded(sessions: pd.DataFrame, cfg, with_fields=True, verbose=True):
    """Yield (catalog_row, Session, field_result_or_None)."""
    for _, row in sessions.iterrows():
        if verbose:
            print(f"  {row['rat']} {row['date']} (day {row['day']}, goal {row['goal']})", flush=True)
        s = io.load_session(cfg, row["rat"], str(row["date"]))
        ok, why = session_ok(s)
        if not ok:
            print(f"    skipped: {why}", flush=True)
            continue
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
    if not v.any():
        return np.array([np.nan]), np.array([1.0])
    d = lm.dist_to_node(goal)[v]
    w = (lm.bins["length"].to_numpy() if weight == "length" else R["occ"])[v]
    return d, w / w.sum()


def weighted_quantile(x, w, q):
    if len(x) == 0 or not np.isfinite(x).any():
        return np.nan
    o = np.argsort(x)
    cw = np.cumsum(w[o])
    return np.interp(q, cw / cw[-1], x[o])


def trial_groups(s, n_groups):
    """Split the session's trials into `n_groups` consecutive groups -> list of trial-number arrays."""
    tr = s.trials["trial"].to_numpy()
    return [g for g in np.array_split(tr, n_groups) if len(g)]


def group_fields(s, R, cfg, lm, trials):
    """Rate maps + fields of the session's place cells using only `trials`."""
    pcs = list(R["units"].loc[R["units"]["is_place"], "unit"])
    m = np.isin(s.trial_id, trials)
    M = fl.rate_maps(s, pcs, cfg, mask=m, lm=lm)
    rows = []
    for k, u in enumerate(pcs):
        for j, fd in enumerate(fl.detect_fields(M["rate"][k], M["valid"], lm, cfg, goal=s.goal)):
            fd.pop("bins")
            rows.append(dict(unit=u, field=j, **fd))
    return M, pd.DataFrame(rows), pcs


def goal_enrichment(peak_bins, valid, lm, node, radius):
    """Observed / expected fraction of field peaks within `radius` (track) of `node`.

    Expected = fraction of valid track length inside the zone (uniform placement).
    """
    if node is None or (isinstance(node, float) and np.isnan(node)) or pd.isna(node):
        return dict(n=len(peak_bins), n_in=np.nan, frac_in=np.nan, exp_frac=np.nan, enrichment=np.nan)
    d = lm.dist_to_node(int(node))
    L = lm.bins["length"].to_numpy()
    zone = d <= radius
    exp = L[valid & zone].sum() / L[valid].sum()
    n_in = int(zone[np.asarray(peak_bins, int)].sum()) if len(peak_bins) else 0
    n = len(peak_bins)
    return dict(n=n, n_in=n_in, frac_in=n_in / n if n else np.nan, exp_frac=exp,
                enrichment=(n_in / n) / exp if n and exp > 0 else np.nan)
