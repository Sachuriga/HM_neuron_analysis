"""Segment the trajectory into visits to maze elements (nodes and edges).

A visit is a maximal run of consecutive in-trial samples on the same element
(``N101`` or ``E101-102``). For edges the traversal direction is recorded
(``+1`` = from the lower to the higher node id). Rates use running samples only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import maze as mz
from .fields import running_mask, spike_samples


def element_sequence(s, lm: mz.LinearMaze):
    elem = lm.element_of_bin()
    lab = np.full(len(s.t), "", dtype=object)
    ok = (s.bin >= 0) & s.in_trial
    lab[ok] = elem[s.bin[ok]]
    return lab


def segment_visits(s, cfg, lm=None):
    """Returns (visits DataFrame, visit_id per sample (-1 = none))."""
    lm = lm or mz.maze_from_cfg(cfg)
    lab = element_sequence(s, lm)
    run = running_mask(s, cfg)
    change = np.r_[True, lab[1:] != lab[:-1]] | np.r_[True, np.diff(s.trial_id) != 0]
    starts = np.where(change)[0]
    stops = np.r_[starts[1:], len(lab)]
    s_along = lm.bins["s"].to_numpy()
    rows, vid = [], np.full(len(lab), -1, int)
    for a, b in zip(starts, stops):
        e = lab[a]
        if not e:
            continue
        dur = s.t[min(b, len(s.t) - 1)] - s.t[a]
        if dur < cfg.visits.min_dur_s:
            continue
        direction = 0
        if e.startswith("E"):
            bb = s.bin[a:b]
            d = np.diff(s_along[bb])
            direction = int(np.sign(d.sum())) if len(d) else 0
        k = len(rows)
        vid[a:b] = k
        rows.append(dict(visit=k, element=e, kind="node" if e[0] == "N" else "edge",
                         direction=direction, i0=a, i1=b, t_start=s.t[a], t_end=s.t[b - 1],
                         dur_s=dur, run_s=run[a:b].sum() * s.dt, trial=int(s.trial_id[a]),
                         speed=float(np.nanmean(s.speed[a:b]))))
    return pd.DataFrame(rows), vid


def visit_counts(s, uids, visits: pd.DataFrame, vid: np.ndarray, cfg):
    """Running spike counts per (unit, visit). Returns (n_units, n_visits) array."""
    run = running_mask(s, cfg)
    out = np.zeros((len(uids), len(visits)))
    for k, u in enumerate(uids):
        si = spike_samples(s, s.spikes[u])
        si = si[run[si] & (vid[si] >= 0)]
        out[k] = np.bincount(vid[si], minlength=len(visits))
    return out


def visit_profiles(s, uid, visits, lm, cfg, element):
    """Per-visit spatial rate profile across the bins of one edge (running samples).

    Returns (profiles (n_visits, n_bins_edge) with NaN where unvisited, visit index array).
    """
    a, b = map(int, element[1:].split("-"))
    eb = lm.edge_bins[(a, b)]
    pos = {bb: i for i, bb in enumerate(eb)}
    run = running_mask(s, cfg)
    si = spike_samples(s, s.spikes[uid])
    sel = visits.index[visits["element"] == element]
    prof = np.full((len(sel), len(eb)), np.nan)
    for r, v in enumerate(sel):
        i0, i1 = visits.at[v, "i0"], visits.at[v, "i1"]
        m = run[i0:i1]
        bins = s.bin[i0:i1][m]
        occ = np.zeros(len(eb))
        for bb in bins:
            if bb in pos:
                occ[pos[bb]] += s.dt
        spk_i = si[(si >= i0) & (si < i1)]
        spk_i = spk_i[run[spk_i]]
        cnt = np.zeros(len(eb))
        for bb in s.bin[spk_i]:
            if bb in pos:
                cnt[pos[bb]] += 1
        with np.errstate(invalid="ignore", divide="ignore"):
            prof[r] = np.where(occ > 0, cnt / occ, np.nan)
    return prof, np.asarray(sel)
