"""Rate maps, spatial information and place fields on the linearised maze.

All maps live on the bins of :class:`hm_neuron.maze.LinearMaze`. Spike counts
and occupancy are smoothed separately with a Gaussian kernel over TRACK
distance (so firing never leaks through walls between corridors), then divided.
Only running samples (speed >= ``cfg.fields.speed_min_m_s``) inside trial
windows are used.

Significance of spatial information uses circular shifts of each spike train
within the concatenated running time (shift >= ``shuffle_min_shift_s``).
"""

from __future__ import annotations

import pickle
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from . import maze as mz


# ----------------------------------------------------------------------------- kernels
@lru_cache(maxsize=8)
def _kernel_cached(maze_id, sigma):
    lm = _MAZES[maze_id]
    D = lm.dist
    K = np.exp(-0.5 * (D / sigma) ** 2)
    K[D > 3 * sigma] = 0.0
    return csr_matrix(K)


_MAZES: dict = {}


def smoothing_kernel(lm: mz.LinearMaze, sigma: float):
    _MAZES[id(lm)] = lm
    return _kernel_cached(id(lm), float(sigma))


# ----------------------------------------------------------------------------- basics
def running_mask(s, cfg, extra=None) -> np.ndarray:
    m = s.in_trial & (s.bin >= 0) & (np.nan_to_num(s.speed) >= cfg.fields.speed_min_m_s)
    if extra is not None:
        m &= extra
    return m


def spike_samples(s, spikes) -> np.ndarray:
    """Index of the position sample nearest to each spike."""
    i = np.searchsorted(s.t, spikes)
    i = np.clip(i, 1, len(s.t) - 1)
    left = s.t[i - 1]
    i = i - ((spikes - left) < (s.t[i] - spikes))
    return i


def occupancy(s, mask, n_bins) -> np.ndarray:
    return np.bincount(s.bin[mask], minlength=n_bins) * s.dt


def spike_counts(s, uids, mask, n_bins) -> np.ndarray:
    out = np.zeros((len(uids), n_bins))
    for k, u in enumerate(uids):
        si = spike_samples(s, s.spikes[u])
        si = si[mask[si]]
        out[k] = np.bincount(s.bin[si], minlength=n_bins)
    return out


def smooth_rate(spk, occ, K, min_occ_s):
    """spk: (n, bins) counts; occ: (bins,) s. Returns rate (n, bins), valid (bins,), occ_s."""
    occ_s = K @ occ
    spk_s = (K @ np.atleast_2d(spk).T).T
    valid = occ_s >= min_occ_s
    with np.errstate(invalid="ignore", divide="ignore"):
        rate = np.where(valid, spk_s / occ_s, np.nan)
    return rate, valid, occ_s


def skaggs_info(rate, occ, valid):
    """Skaggs spatial information (bits/spike) per row of `rate`."""
    rate = np.atleast_2d(rate)
    p = np.where(valid, occ, 0.0)
    p = p / p.sum()
    r = np.where(valid, rate, 0.0)
    mean = (r * p).sum(1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = r / mean
        term = np.where(ratio > 0, p * ratio * np.log2(ratio), 0.0)
    si = term.sum(1)
    si[mean[:, 0] <= 0] = np.nan
    return si


def rate_maps(s, uids, cfg, mask=None, lm=None):
    """Rate maps for `uids` over the running samples selected by `mask`.

    Returns dict(rate, valid, occ, occ_s, counts).
    """
    lm = lm or mz.maze_from_cfg(cfg)
    m = running_mask(s, cfg, mask)
    occ = occupancy(s, m, lm.n_bins)
    spk = spike_counts(s, uids, m, lm.n_bins)
    K = smoothing_kernel(lm, cfg.fields.smooth_sigma_m)
    rate, valid, occ_s = smooth_rate(spk, occ, K, cfg.fields.min_occ_s)
    return dict(rate=rate, valid=valid, occ=occ, occ_s=occ_s, counts=spk)


def shuffled_info(s, uid, cfg, mask, lm, occ, valid, K, rng):
    """Null distribution of SI from circular shifts within running time."""
    ms = np.where(mask)[0]
    n = len(ms)
    si = spike_samples(s, s.spikes[uid])
    si = si[mask[si]]
    if len(si) == 0 or n < 10:
        return np.full(cfg.fields.n_shuffles, np.nan)
    rank = np.searchsorted(ms, si)
    min_shift = int(cfg.fields.shuffle_min_shift_s / s.dt)
    lo, hi = min(min_shift, n // 2), max(n - min_shift, n // 2 + 1)
    shifts = rng.integers(lo, hi, cfg.fields.n_shuffles)
    nb = lm.n_bins
    batch = max(1, int(2e6 // max(len(rank), 1)))
    out = []
    for c in range(0, len(shifts), batch):
        sh = shifts[c:c + batch]
        b = s.bin[ms[(rank[None] + sh[:, None]) % n]]          # (batch, n_spk)
        cnt = np.bincount((np.arange(len(sh))[:, None] * nb + b).ravel(),
                          minlength=len(sh) * nb).reshape(len(sh), nb)
        rate, _, _ = smooth_rate(cnt.astype(float), occ, K, cfg.fields.min_occ_s)
        out.append(skaggs_info(rate, occ, valid))
    return np.concatenate(out)


# ----------------------------------------------------------------------------- fields
def detect_fields(rate, valid, lm: mz.LinearMaze, cfg, goal=None) -> list[dict]:
    """Place fields of one rate map: connected sets of valid bins >= frac*peak."""
    f = cfg.fields
    r = np.where(valid, rate, np.nan)
    if not np.isfinite(r).any():
        return []
    peak = np.nanmax(r)
    if peak < f.min_peak_hz:
        return []
    above = np.nan_to_num(r) >= f.field_frac * peak
    idx = np.where(above)[0]
    if len(idx) == 0:
        return []
    sub = lm.adj[idx][:, idx]
    ncomp, lab = connected_components(sub, directed=False)
    lengths = lm.bins["length"].to_numpy()
    is_node = lm.is_node()
    elem = lm.element_of_bin()
    dgoal = lm.dist_to_node(goal) if goal is not None else None
    out = []
    for c in range(ncomp):
        b = idx[lab == c]
        size = lengths[b].sum()
        pk = b[np.nanargmax(r[b])]
        if r[pk] < f.min_peak_hz or size < f.min_field_m:
            continue
        w = r[b] * lengths[b]
        d = dict(n_bins=len(b), size_m=float(size), peak_bin=int(pk), peak_hz=float(r[pk]),
                 mean_in_hz=float(np.average(r[b], weights=lengths[b])),
                 peak_kind="node" if is_node[pk] else "edge",
                 peak_element=elem[pk],
                 node_frac=float(lengths[b][is_node[b]].sum() / size),
                 n_elements=len(set(elem[b])),
                 x=float(np.average(lm.bins["x"].to_numpy()[b], weights=w)),
                 y=float(np.average(lm.bins["y"].to_numpy()[b], weights=w)),
                 bins=b)
        if dgoal is not None:
            d["goal_dist_m"] = float(dgoal[pk])
            d["goal_dist_min_m"] = float(dgoal[b].min())
        out.append(d)
    return out


def session_fields(s, cfg, refresh=False, lm=None):
    """Per-session unit and field tables (cached).

    Returns dict with
      ``units``  one row per selected unit: si, si_p, si_z, mean_hz, peak_hz, n_fields, is_place
      ``fields`` one row per field (unit, field, size_m, peak_kind, goal_dist_m, ...)
      ``rate``   (n_units, n_bins) rate maps, ``valid``, ``occ``, ``occ_s``, ``uids``
    """
    lm = lm or mz.maze_from_cfg(cfg)
    key = cfg.key("fields+lin+units")
    cache = Path(cfg.cache_dir) / "fields" / f"{s.label}_{key}.pkl"
    if cache.exists() and not refresh:
        with open(cache, "rb") as fh:
            return pickle.load(fh)
    uids = s.select_units(cfg)
    mask = running_mask(s, cfg)
    occ = occupancy(s, mask, lm.n_bins)
    K = smoothing_kernel(lm, cfg.fields.smooth_sigma_m)
    spk = spike_counts(s, uids, mask, lm.n_bins)
    rate, valid, occ_s = smooth_rate(spk, occ, K, cfg.fields.min_occ_s)
    si = skaggs_info(rate, occ, valid)
    rng = np.random.default_rng(cfg.fields.seed)
    urows, frows = [], []
    run_time = mask.sum() * s.dt
    for k, u in enumerate(uids):
        nspk = spk[k].sum()
        if nspk >= cfg.units.min_spikes:
            null = shuffled_info(s, u, cfg, mask, lm, occ, valid, K, rng)
            p = (1 + np.sum(null >= si[k])) / (1 + np.isfinite(null).sum())
            z = (si[k] - np.nanmean(null)) / (np.nanstd(null) + 1e-12)
        else:
            p, z = np.nan, np.nan
        flds = detect_fields(rate[k], valid, lm, cfg, goal=s.goal) if nspk >= cfg.units.min_spikes else []
        is_place = bool(np.isfinite(p) and p < cfg.fields.alpha and len(flds) > 0)
        urows.append(dict(unit=u, n_spikes_run=int(nspk), mean_hz=nspk / max(run_time, 1e-9),
                          peak_hz=float(np.nanmax(rate[k])) if np.isfinite(rate[k]).any() else np.nan,
                          si=float(si[k]), si_p=float(p), si_z=float(z),
                          n_fields=len(flds), is_place=is_place,
                          cell_type=s.units.loc[u, "cell_type"]))
        for j, fd in enumerate(flds):
            frows.append(dict(unit=u, field=j, is_place=is_place, **fd))
    meta = {k: s.meta[k] for k in ("rat", "date", "day", "repeat", "session", "goal", "block",
                                  "day_in_block", "prev_goal", "is_switch_day", "is_recall_day")
            if k in s.meta}
    units = pd.DataFrame(urows).assign(**meta)
    fields = pd.DataFrame(frows).assign(**meta) if frows else pd.DataFrame()
    out = dict(units=units, fields=fields, rate=rate, valid=valid, occ=occ, occ_s=occ_s,
               uids=np.array(uids), run_time_s=run_time)
    cache.parent.mkdir(parents=True, exist_ok=True)
    with open(cache, "wb") as fh:
        pickle.dump(out, fh)
    return out
