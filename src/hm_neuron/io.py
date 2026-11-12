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
CACHE_VERSION = 3


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
        with open_nwb(p) as f:
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


class open_nwb:
    """``h5py.File`` context that ignores the SMB volume's spurious
    'Bad file descriptor' error on close (data are already read by then)."""

    def __init__(self, path):
        self.f = h5py.File(path, "r")

    def __enter__(self):
        return self.f

    def __exit__(self, *exc):
        try:
            self.f.close()
        except RuntimeError:
            pass
        return False


def _read_nwb(path: str) -> dict:
    with open_nwb(path) as f:
        return _read_open(f)


def _read_open(f) -> dict:
    pos = f["processing/Behavior/Position/Rat"]
    xy = pos["data"][:].astype(float)
    t = pos["timestamps"][:].astype(float)
    ref = f["timestamps_reference_time"][()]
    ref = pd.Timestamp(ref.decode() if isinstance(ref, bytes) else str(ref)).timestamp()
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
    return dict(xy=xy, t=t, steps=steps, units=units, spikes=spikes, ref_epoch=ref)


def _read_nwb_retry(path, tries=3):
    """The SMB volume occasionally fails on h5py close ('Bad file descriptor'); retry."""
    import time
    for k in range(tries):
        try:
            return _read_nwb(path)
        except (OSError, RuntimeError):
            if k == tries - 1:
                raise
            time.sleep(2)


TIME_COLS = ("Time_start", "Time_end", "Trial_start_time", "Trial_end_time", "Trial_start_s", "Trial_end_s")


def _step_alignment_error(steps, t, x, y, offset):
    """Median distance (m) between tracked position at each step end and its Next_node."""
    pos = mz.build_graph().nodes
    nn = pd.to_numeric(steps["Next_node"], errors="coerce")
    ok = nn.isin(list(pos)).to_numpy()
    if ok.sum() < 5:
        return np.nan
    te = steps["Time_end"].to_numpy(float)[ok] + offset
    px = np.interp(te, t, x, left=np.nan, right=np.nan)
    py = np.interp(te, t, y, left=np.nan, right=np.nan)
    nxy = np.array([pos[int(n)]["pos"] for n in nn[ok]])
    d = np.hypot(px - nxy[:, 0], py - nxy[:, 1])
    return float(np.nanmedian(d)) if np.isfinite(d).sum() >= 5 else np.nan


def _fix_step_times(steps, ref_epoch, t, x, y, label=""):
    """Put all step/trial times on the session (ephys) clock and validate them.

    Some NWBs store trial times as Unix epoch seconds; these are converted by
    subtracting ``timestamps_reference_time``. The alignment is then checked
    against the data: the tracked position at each step's end should sit on its
    ``Next_node``. If the median error is > 0.25 m, the offset minimising it is
    searched (coarse 1 s grid over +-4 h, then 0.1 s) and applied only if the
    error falls below 0.15 m. ``steps.attrs`` records offset and error.
    """
    import warnings
    steps = steps.copy()
    epoch = bool(np.nanmedian(steps["Time_start"]) > 1e8)
    if epoch:
        for c in TIME_COLS:
            if c in steps and np.nanmedian(steps[c]) > 1e8:
                steps[c] = steps[c] - ref_epoch
    err0 = _step_alignment_error(steps, t, x, y, 0.0)
    shift, err = 0.0, err0
    if not (np.isfinite(err0) and err0 <= 0.25):
        grid = np.arange(-4 * 3600, 4 * 3600 + 1, 1.0)
        errs = np.array([_step_alignment_error(steps, t, x, y, o) for o in grid])
        if np.isfinite(errs).any():
            o = grid[np.nanargmin(errs)]
            fine = np.arange(o - 1, o + 1.01, 0.1)
            fe = np.array([_step_alignment_error(steps, t, x, y, v) for v in fine])
            best, berr = fine[np.nanargmin(fe)], np.nanmin(fe)
            if berr < 0.15:
                shift, err = float(best), float(berr)
        if shift:
            warnings.warn(f"{label}: trial times shifted by {shift:+.1f} s "
                          f"(step-end error {err0:.2f} -> {err:.2f} m)", stacklevel=2)
            for c in TIME_COLS:
                if c in steps:
                    steps[c] = steps[c] + shift
        else:
            warnings.warn(f"{label}: trial times could not be aligned (step-end error {err0:.2f} m)",
                          stacklevel=2)
    steps.attrs.update(time_shift_s=shift, align_error_m=err, was_epoch=epoch)
    return steps


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
        # Trial_start_s/_end_s exist in only half the NWBs; *_time is in all of them
        "t_start": g["Trial_start_time"].first(),
        "t_end": g["Trial_end_time"].first(),
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

    raw = _read_nwb_retry(meta["path"])
    t, x, y = _clean_position(raw["t"], raw["xy"])
    speed = _speed(t, x, y, cfg.fields.speed_smooth_s)
    lm = mz.maze_from_cfg(cfg)
    b = lm.project(x, y)
    steps = _fix_step_times(raw["steps"], raw["ref_epoch"], t, x, y, f"{rat} {date}")
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
