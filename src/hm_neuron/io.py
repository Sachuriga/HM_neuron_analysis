"""Session discovery, loading and local caching.

The data volume is read-only and slow, so every session is read from its NWB
once and cached as a pickle under ``cfg.cache_dir`` (default ``~/hm_cache``).
Spikes, position, LFP and trials in the NWB share one clock (ephys seconds).
"""

from __future__ import annotations

import dataclasses as dc
import pickle
import re
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

from . import maze as mz

_DESC = re.compile(r"(Rat\d+).*?(\d{8}).*?Repeat (\d+), Day (\d+), Session (\d+), Goal_Node (\d+)")
CACHE_VERSION = 1


# ----------------------------------------------------------------------------- catalog
def find_task_nwbs(root) -> list[Path]:
    root = Path(root) / "task"
    return sorted(p for p in root.glob("*/2026*/*.nwb")
                  if not p.name.startswith("._") and not p.name.endswith(".tmp.nwb"))


def catalog(cfg, refresh=False) -> pd.DataFrame:
    """One row per task session: rat, date, repeat, day, session, goal, path, and
    goal-block bookkeeping (``block``, ``day_in_block``, ``prev_goal``,
    ``is_switch_day`` = first day with a new goal, ``is_recall_day`` = last day
    with the old goal i.e. Session 1 of a Repeat)."""
    cache = Path(cfg.cache_dir) / "catalog.csv"
    if cache.exists() and not refresh:
        return pd.read_csv(cache, dtype={"date": str})
    rows = []
    for p in find_task_nwbs(cfg.data_root):
        with h5py.File(p, "r") as f:
            d = f["session_description"][()]
            d = d.decode() if isinstance(d, bytes) else str(d)
        m = _DESC.search(d)
        if not m:
            continue
        rat, date, rep, day, ses, goal = m.groups()
        rows.append(dict(rat=rat, date=date, repeat=int(rep), day=int(day),
                         session=int(ses), goal=int(goal), path=str(p)))
    df = pd.DataFrame(rows).sort_values(["rat", "day"]).reset_index(drop=True)
    out = []
    for rat, g in df.groupby("rat", sort=False):
        g = g.copy()
        block = (g["goal"] != g["goal"].shift()).cumsum() - 1
        g["block"] = block.values
        g["day_in_block"] = g.groupby("block").cumcount()
        prev = g.groupby("block")["goal"].first().shift()
        g["prev_goal"] = g["block"].map(prev).astype("Int64")
        g["is_switch_day"] = (g["day_in_block"] == 0) & (g["block"] > 0)
        nxt = g["block"].shift(-1)
        g["is_recall_day"] = ((nxt != g["block"]) & nxt.notna() & (g["session"] == 1)
                              & (g["day"] > 0))
        g["is_exploration"] = g["day"] == 0   # Day 0: single free-exploration trial
        out.append(g)
    df = pd.concat(out, ignore_index=True)
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache, index=False)
    return df


# ----------------------------------------------------------------------------- session
@dc.dataclass
class Session:
    rat: str
    date: str
    meta: dict                     # catalog row
    t: np.ndarray                  # position timestamps (s)
    x: np.ndarray                  # metres
    y: np.ndarray
    speed: np.ndarray              # m/s, smoothed
    bin: np.ndarray                # linearised maze bin per sample (-1 = off maze)
    in_trial: np.ndarray           # bool, sample inside a trial window
    trial_id: np.ndarray           # trial number per sample (-1 outside)
    steps: pd.DataFrame            # node-to-node traversals (Trials_Data)
    trials: pd.DataFrame           # one row per trial: trial, t_start, t_end, start_node, n_steps
    units: pd.DataFrame            # unit table (index = unit id)
    spikes: dict                   # unit id -> spike times (s)
    path: str = ""

    @property
    def goal(self) -> int:
        return int(self.meta["goal"])

    @property
    def dt(self) -> float:
        return float(np.median(np.diff(self.t)))

    @property
    def label(self) -> str:
        return f"{self.rat}_{self.date}"

    def select_units(self, cfg, cell_type=True) -> list:
        u = self.units
        m = u["quality_label"].isin(cfg.units.quality)
        if cell_type and cfg.units.cell_type:
            m &= u["cell_type"].isin(cfg.units.cell_type)
        return list(u.index[m])

    def lfp_theta_channel(self, cfg):
        from .lfp import load_reference_lfp
        return load_reference_lfp(self, cfg)


def _dec(a):
    return np.array([x.decode() if isinstance(x, bytes) else x for x in a])


def _read_nwb(path: str) -> dict:
    with h5py.File(path, "r") as f:
        pos = f["processing/Behavior/Position/Rat"]
        xy = pos["data"][:].astype(float)
        t = pos["timestamps"][:].astype(float)
        T = f["processing/Behavior/Trials_Data"]
        steps = pd.DataFrame({c: (_dec(T[c][:]) if T[c].dtype == object else T[c][:])
                              for c in T if isinstance(T[c], h5py.Dataset) and T[c].ndim == 1})
        U = f["units"]
        cols = [c for c in U.attrs["colnames"] if c not in ("spike_times", "waveform_mean")]
        ids = U["id"][:]
        units = pd.DataFrame({c: (_dec(U[c][:]) if U[c].dtype == object else U[c][:])
                              for c in cols if c in U and U[c].shape == ids.shape}, index=ids)
        units.index.name = "unit"
        idx = U["spike_times_index"][:]
        st = U["spike_times"]
        keep = units["quality_label"].to_numpy() != "noise"
        spikes, start = {}, 0
        for uid, stop, k in zip(ids, idx, keep):
            if k:
                spikes[int(uid)] = st[start:stop].astype(float)
            start = stop
        units = units[keep]
    return dict(xy=xy, t=t, steps=steps, units=units, spikes=spikes)


def _clean_position(t, xy_px, max_gap_s=0.5, smooth_s=0.1):
    order = np.argsort(t)
    t, xy = t[order], xy_px[order]
    x = xy[:, 0] / mz.SCALE_X
    y = xy[:, 1] / mz.SCALE_Y
    dt = np.median(np.diff(t))
    bad = ~(np.isfinite(x) & np.isfinite(y))
    if bad.any() and (~bad).sum() > 2:
        # run-length of each NaN gap; fill only gaps shorter than max_gap_s
        edges = np.diff(np.r_[0, bad.astype(int), 0])
        starts, stops = np.where(edges == 1)[0], np.where(edges == -1)[0]
        short = (stops - starts) * dt <= max_gap_s
        fill = np.zeros_like(bad)
        for s0, s1 in zip(starts[short], stops[short]):
            fill[s0:s1] = True
        for v in (x, y):
            v[fill] = np.interp(t[fill], t[~bad], v[~bad])
        x[bad & ~fill] = np.nan
        y[bad & ~fill] = np.nan
    return t, x, y


def _speed(t, x, y, smooth_s):
    dt = np.median(np.diff(t))
    xs = gaussian_filter1d(np.nan_to_num(x, nan=np.nanmean(x)), smooth_s / dt)
    ys = gaussian_filter1d(np.nan_to_num(y, nan=np.nanmean(y)), smooth_s / dt)
    sp = np.hypot(np.gradient(xs, t), np.gradient(ys, t))
    sp[~np.isfinite(x)] = np.nan
    return sp


def _trial_table(steps: pd.DataFrame) -> pd.DataFrame:
    g = steps.groupby("Trial_Num")
    tr = pd.DataFrame({
        "t_start": g["Trial_start_s"].first(),
        "t_end": g["Trial_end_s"].first(),
        "start_node": g["Start_node"].first().astype(int),
        "goal": g["Goal_node"].first().astype(int),
        "n_steps": g.size(),
        "path_m": g["Length_m"].sum(),
    })
    tr.index.name = "trial"
    return tr.reset_index()


def load_session(cfg, rat: str, date: str, refresh=False) -> Session:
    cat = catalog(cfg)
    row = cat[(cat["rat"] == rat) & (cat["date"].astype(str) == str(date))]
    if row.empty:
        raise KeyError(f"no session {rat} {date}")
    meta = row.iloc[0].to_dict()
    cache = Path(cfg.cache_dir) / "sessions" / f"{rat}_{date}_v{CACHE_VERSION}_{cfg.key('lin')}.pkl"
    if cache.exists() and not refresh:
        with open(cache, "rb") as fh:
            d = pickle.load(fh)
        d["meta"] = meta
        return Session(**d)

    raw = _read_nwb(meta["path"])
    t, x, y = _clean_position(raw["t"], raw["xy"])
    speed = _speed(t, x, y, cfg.fields.speed_smooth_s)
    lm = mz.maze_from_cfg(cfg)
    b = lm.project(x, y)
    steps = raw["steps"]
    for c in ("Start_node", "Next_node", "Goal_node"):
        steps[c] = pd.to_numeric(steps[c], errors="coerce").astype("Int64")
    trials = _trial_table(steps)
    trial_id = np.full(t.shape, -1, int)
    for r in trials.itertuples():
        trial_id[(t >= r.t_start) & (t <= r.t_end)] = r.trial
    d = dict(rat=rat, date=str(date), meta=meta, t=t, x=x, y=y, speed=speed, bin=b,
             in_trial=trial_id >= 0, trial_id=trial_id, steps=steps, trials=trials,
             units=raw["units"], spikes=raw["spikes"], path=meta["path"])
    cache.parent.mkdir(parents=True, exist_ok=True)
    with open(cache, "wb") as fh:
        pickle.dump({k: v for k, v in d.items() if k != "meta"}, fh)
    return Session(**d)


def iter_sessions(cfg, rats=None, dates=None, query=None):
    """Yield sessions; `query` is a pandas query on the catalog, e.g. ``"day > 0"``."""
    cat = catalog(cfg)
    if rats:
        cat = cat[cat["rat"].isin(rats)]
    if dates:
        cat = cat[cat["date"].astype(str).isin([str(d) for d in dates])]
    if query:
        cat = cat.query(query)
    for r in cat.itertuples():
        yield load_session(cfg, r.rat, str(r.date))
