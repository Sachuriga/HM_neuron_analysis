"""10. How many goal, start, bridge and outside-maze cells are there in each session?

Every good unit (pyramidal cells and interneurons, reported separately) is tested for
a firing preference for each of these behavioural states. Classes are NOT exclusive.

Spatial classes (in-trial running samples only, so nothing here is a
running-vs-immobility contrast):

  goal          within ``goal.radius_m`` (track distance) of the session's goal node
  start         within ``goal.radius_m`` of the start node of the CURRENT trial
                (the start node changes from trial to trial, so this is not a fixed place)
  bridge_long   on an inter-island corridor longer than ``LONG_BRIDGE_M``
                (121-302 and 223-404, 1.9-2.0 m)
  bridge_short  on one of the other inter-island corridors
                (124-201, 220-305, 324-401, 0.7 m)
  bridge        on any of the five

Temporal classes (reference = all in-trial time on the maze):

  outside       the rat is off the maze: carried from the goal to the next start,
                and before the first / after the last trial
  reward        the rat is still at the goal after arriving, until it is picked up

The rat's own track is frozen between trials, so the two cannot be told apart
from it. The pick-up time comes from the Researcher track in the NWB: the last
moment in the gap at which the researcher was within ``NEAR_M`` of the place
where the rat ended the trial. Without that split, "between trials" would be
mostly reward consumption at the goal (the rat stays there for most of each gap).

Test. A unit is in a class when its rate in the state is >= ``visits.active_hz``
and exceeds the reference rate with p < ``fields.alpha`` (one-sided) against the
complete set of circular shifts (>= ``fields.shuffle_min_shift_s``) of its spike
train relative to behaviour. All shifts are evaluated at once with an FFT
cross-correlation, so the p-value is exact for this null and needs no seed.

Location controls. A spatially tuned population gives "X cells" for any location,
so the same test is repeated for control locations with similar running occupancy
(0.5-2x): zones around other nodes (goal), and random sets of within-island
corridors of the same total length (bridges). Start cells have no such control
because the start zone is defined per trial; their chance level is alpha.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import fft as sfft
from scipy import stats as st

from .. import fields as fl, io, maze as mz, plotting as P, stats as S
from . import common

NAME = "cell_classes"
TITLE = "Goal, start, bridge and outside-maze cells per session"
DEFAULT_QUERY = "repeat > 0"  # repeat 0 = habituation, not training

LONG_BRIDGE_M = 1.0        # inter-island corridors longer than this are "long"
NEAR_M = 1.5               # researcher this close to the rat's last position = at the rat
AMBIGUOUS_M = 2.0          # next start this close to the trial's end point: pick-up cannot be timed
AT_GOAL_M = 0.3            # trial ended at the goal when the last position is this close to it
PICKUP_SEARCH_S = 180.0    # pick-up is searched in this window after the trial end
MAX_SPEED_M_S = 3.0        # faster = tracking jump (rat put down at the start), not running
MIN_STATE_S = 30.0         # a temporal state needs this much time to be tested
N_CTRL_SETS = 50           # random corridor sets per bridge class
MIN_CTRL = 5               # fewer matched control locations than this: no control value
OCC_MATCH = (0.5, 2.0)     # control occupancy / zone occupancy

SPATIAL = ("goal", "start", "bridge_long", "bridge_short", "bridge")
TEMPORAL = ("outside", "reward")
CLASSES = SPATIAL + TEMPORAL
CONTROLLED = ("goal", "bridge_long", "bridge_short", "bridge")
LABELS = {"goal": "goal", "start": "start", "bridge_long": "long bridge", "bridge_short": "short bridge",
          "bridge": "any bridge", "outside": "outside maze", "reward": "at goal after arrival"}
CELL_TYPES = ("pyramidal", "interneuron")
META = ("rat", "date", "day", "repeat", "session", "goal", "block", "day_in_block",
        "is_switch_day", "is_recall_day")


# ----------------------------------------------------------------------------- zones
def bridge_edges(lm) -> dict:
    """{'bridge_long': [(a, b), ...], 'bridge_short': [...]}: edges joining two islands."""
    out = {"bridge_long": [], "bridge_short": []}
    for a, b in lm.edges:
        if mz.island_of(a) != mz.island_of(b):
            long = lm.G.edges[a, b]["length"] > LONG_BRIDGE_M
            out["bridge_long" if long else "bridge_short"].append((a, b))
    return out


def bridge_bins(lm) -> dict:
    """Bin masks of the long, the short and all inter-island corridors (node zones excluded)."""
    out = {}
    for k, edges in bridge_edges(lm).items():
        m = np.zeros(lm.n_bins, bool)
        for e in edges:
            m[lm.edge_bins[e]] = True
        out[k] = m
    out["bridge"] = out["bridge_long"] | out["bridge_short"]
    return out


def researcher_track(s, cfg):
    """(t, x, y) of the tracked researcher (finite samples, metres), or None. Cached."""
    cache = Path(cfg.cache_dir) / "researcher" / f"{s.label}.npz"
    if cache.exists():
        d = np.load(cache)
        return d["t"], d["x"], d["y"]
    if not s.path or not Path(s.path).exists():
        return None
    try:
        with io.open_nwb(s.path) as f:
            g = f["processing/Behavior/Position"]
            if "Researcher" not in g:
                return None
            xy = g["Researcher/data"][:].astype(float)
            t = g["Researcher/timestamps"][:].astype(float)
    except (OSError, KeyError):
        print(f"    {s.label}: researcher track could not be read; outside/reward not classified", flush=True)
        return None
    ok = np.isfinite(xy).all(1) & np.isfinite(t)
    o = np.argsort(t[ok])
    t, x, y = t[ok][o], xy[ok, 0][o] / mz.SCALE_X, xy[ok, 1][o] / mz.SCALE_Y
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, t=t, x=x, y=y)
    return t, x, y


def pickups(s, res, lm) -> pd.DataFrame:
    """Per trial: when the rat was picked up after it (``t_pick``, NaN if it cannot be timed).

    ``t_pick`` is the last researcher sample within ``NEAR_M`` of the rat's final
    tracked position, searched from the trial end to the next trial start (at most
    ``PICKUP_SEARCH_S``). Gaps whose next start node lies within ``AMBIGUOUS_M``
    of that position are left untimed: the researcher never has to walk away.
    """
    rt, rx, ry = res
    tr = s.trials.sort_values("t_start").reset_index(drop=True)
    fin = np.where(np.isfinite(s.x) & np.isfinite(s.y))[0]
    tf = s.t[fin]
    nodes = lm.G.nodes
    gxy = nodes[s.goal]["pos"] if s.goal in nodes else (np.nan, np.nan)
    rows = []
    for i, q in enumerate(tr.itertuples()):
        last = i == len(tr) - 1
        t_next = np.inf if last else float(tr["t_start"].iloc[i + 1])
        k = np.searchsorted(tf, q.t_end, side="right") - 1
        if k < 0:
            continue
        ex, ey = s.x[fin[k]], s.y[fin[k]]
        at_goal = bool(np.hypot(ex - gxy[0], ey - gxy[1]) <= AT_GOAL_M)
        t_pick = np.nan
        nxt = None if last else int(tr["start_node"].iloc[i + 1])
        ambiguous = (nxt in nodes and
                     np.hypot(ex - nodes[nxt]["pos"][0], ey - nodes[nxt]["pos"][1]) < AMBIGUOUS_M)
        if not ambiguous:
            a, b = np.searchsorted(rt, [q.t_end, min(t_next, q.t_end + PICKUP_SEARCH_S)])
            near = np.where(np.hypot(rx[a:b] - ex, ry[a:b] - ey) <= NEAR_M)[0]
            if len(near):
                t_pick = float(rt[a + near[-1]])
        rows.append(dict(trial=q.trial, t_end=float(q.t_end), t_next=t_next, t_pick=t_pick,
                         ended_at_goal=at_goal))
    return pd.DataFrame(rows)


def label_states(s, cfg, lm, res=None) -> dict:
    """Boolean sample masks: ``run`` (reference of the spatial classes), ``maze`` (reference
    of the temporal classes), one per class, and ``picks`` (the pick-up table or None).

    Everything is restricted to the span of the spike record: some recordings stop
    before the last trial, and those samples would count as time without spikes.
    """
    t = s.t
    sp = [v for v in s.spikes.values() if len(v)]
    t0, t1 = min(v.min() for v in sp), max(v.max() for v in sp)
    rec = (t >= t0) & (t <= t1)
    run = fl.running_mask(s, cfg) & rec & (np.nan_to_num(s.speed) < MAX_SPEED_M_S)
    b = np.clip(s.bin, 0, None)
    r = cfg.goal.radius_m
    Z = dict(run=run, maze=s.in_trial & (s.bin >= 0) & rec)
    Z["goal"] = run & (lm.dist_to_node(s.goal)[b] <= r) if s.goal in lm.node_bin else np.zeros_like(run)
    dstart = np.full(len(t), np.inf)
    for q in s.trials.itertuples():
        if q.start_node in lm.node_bin:
            m = run & (s.trial_id == q.trial)
            dstart[m] = lm.dist_to_node(q.start_node)[s.bin[m]]
    Z["start"] = run & (dstart <= r)
    for k, m in bridge_bins(lm).items():
        Z[k] = run & m[b]
    outside, reward, picks = np.zeros_like(run), np.zeros_like(run), None
    if res is not None and len(s.trials):
        picks = pickups(s, res, lm)
        outside |= t < s.trials["t_start"].min()
        for q in picks.itertuples():
            if not np.isfinite(q.t_pick):
                continue
            outside |= (t > q.t_pick) & (t < q.t_next)
            if q.ended_at_goal:
                reward |= (t > q.t_end) & (t <= q.t_pick)
    Z["outside"] = outside & rec & ~s.in_trial
    Z["reward"] = reward & rec & ~s.in_trial
    Z["picks"] = picks
    return Z


def control_zones(s, lm, Z, cfg):
    """Control locations for the goal and bridge classes.

    Returns (mask (n_bins, n_ctrl) bool, labels (n_ctrl,) = class each column controls for).
    """
    run = Z["run"]
    occ = np.bincount(s.bin[run], minlength=lm.n_bins).astype(float)
    L = lm.bins["length"].to_numpy()
    cols, lab = [], []

    def matched(mask, target):
        return target > 0 and OCC_MATCH[0] <= occ[mask].sum() / target <= OCC_MATCH[1]

    if s.goal in lm.node_bin:
        r = cfg.goal.radius_m
        dg = lm.dist_to_node(s.goal)
        target = occ[dg <= r].sum()
        for n, nb in lm.node_bin.items():
            m = lm.dist[:, nb] <= r
            if dg[nb] > 2 * r and matched(m, target):     # zone does not touch the goal zone
                cols.append(m)
                lab.append("goal")
    bb = bridge_bins(lm)
    inner = [e for e in lm.edges if mz.island_of(e[0]) == mz.island_of(e[1])]
    rng = np.random.default_rng(cfg.fields.seed)
    for k in ("bridge_long", "bridge_short", "bridge"):
        target_len, target_occ, n_ok = L[bb[k]].sum(), occ[bb[k]].sum(), 0
        for _ in range(20 * N_CTRL_SETS):
            if n_ok == N_CTRL_SETS:
                break
            m, tot = np.zeros(lm.n_bins, bool), 0.0
            for j in rng.permutation(len(inner)):
                idx = lm.edge_bins[inner[j]]
                m[idx] = True
                tot += L[idx].sum()
                if tot >= target_len:
                    break
            if matched(m, target_occ):
                cols.append(m)
                lab.append(k)
                n_ok += 1
    M = np.array(cols).T if cols else np.zeros((lm.n_bins, 0), bool)
    return M, np.array(lab, dtype=object)


# ----------------------------------------------------------------------------- shift test
def shift_null(counts, FL, n, min_shift):
    """sum_i counts[i] * labels[k, (i + j) % n] for every circular shift j with
    min_shift <= j <= n - min_shift. ``FL`` = rfft(labels, axis=1). Returns (K, n_shifts)."""
    x = sfft.irfft(FL * np.conj(sfft.rfft(counts))[None], n=n, axis=1, workers=-1)
    lo = min(min_shift, n // 2)
    return x[:, lo:max(n - min_shift + 1, lo + 1)]


def _sample_counts(s, spikes, pos, n):
    """Spike count per reference sample (``pos`` maps position sample -> reference index or -1)."""
    spk = spikes[(spikes >= s.t[0]) & (spikes <= s.t[-1])]
    k = pos[fl.spike_samples(s, spk)] if len(spk) else np.array([], int)
    return np.bincount(k[k >= 0], minlength=n).astype(float)


def _fast_len(n):
    """Largest FFT-friendly length <= n (drops well under 1 % of the reference samples)."""
    f = getattr(sfft, "prev_fast_len", None)
    return int(f(n, real=True)) if f and n > 0 else n


def _reference(mask):
    idx = np.where(mask)[0]
    idx = idx[:_fast_len(len(idx))]
    pos = np.full(len(mask), -1)
    pos[idx] = np.arange(len(idx))
    return idx, pos


def spatial_tests(s, cfg, uids, Z, ctrl=None):
    """In-zone vs out-of-zone running rate for the SPATIAL classes (+ control zones).

    Returns dict(rate_in, rate_out, p: (n_units, K), n_spikes: (n_units,), T: (K,)).
    With the labels fixed, the in-zone spike count is a sufficient statistic.
    """
    idx, pos = _reference(Z["run"])
    n = len(idx)
    L = np.array([Z[c][idx] for c in SPATIAL], float)
    if ctrl is not None and ctrl.shape[1]:
        L = np.vstack([L, ctrl[s.bin[idx]].T.astype(float)])
    K = len(L)
    T_in = L.sum(1) * s.dt
    T_out = n * s.dt - T_in
    FL = sfft.rfft(L, axis=1)
    m = int(cfg.fields.shuffle_min_shift_s / s.dt)
    out = dict(rate_in=np.full((len(uids), K), np.nan), rate_out=np.full((len(uids), K), np.nan),
               p=np.ones((len(uids), K)), n_spikes=np.zeros(len(uids)), T=T_in)
    for j, u in enumerate(uids):
        c = _sample_counts(s, s.spikes[u], pos, n)
        N = c.sum()
        out["n_spikes"][j] = N
        if N == 0 or n < 4:
            continue
        obs = L @ c
        null = shift_null(c, FL, n, m)
        out["p"][j] = (1 + (null >= obs[:, None] - 0.5).sum(1)) / (1 + null.shape[1])
        with np.errstate(invalid="ignore", divide="ignore"):
            out["rate_in"][j] = obs / T_in
            out["rate_out"][j] = (N - obs) / T_out
    return out


def temporal_tests(s, cfg, uids, Z):
    """Rate in each TEMPORAL state vs the in-trial rate on the maze.

    Returns dict(rate (n_units, n_temporal), rate_maze (n_units,), p, n_spikes, T (n_temporal,)).
    States shorter than ``MIN_STATE_S`` are not tested (NaN).
    """
    ref = Z["maze"] | Z["outside"] | Z["reward"]
    idx, pos = _reference(ref)
    n = len(idx)
    L = np.array([Z[c][idx] for c in ("maze",) + TEMPORAL], float)
    T = L.sum(1) * s.dt
    FL = sfft.rfft(L, axis=1)
    m = int(cfg.fields.shuffle_min_shift_s / s.dt)
    nu, nt = len(uids), len(TEMPORAL)
    out = dict(rate=np.full((nu, nt), np.nan), rate_maze=np.full(nu, np.nan), p=np.full((nu, nt), np.nan),
               n_spikes=np.zeros(nu), T=T[1:])
    ok = (T[1:] >= MIN_STATE_S) & (T[0] >= MIN_STATE_S)
    for j, u in enumerate(uids):
        c = _sample_counts(s, s.spikes[u], pos, n)
        out["n_spikes"][j] = c.sum()
        if T[0] <= 0:
            continue
        obs = L @ c
        out["rate_maze"][j] = obs[0] / T[0]
        if not ok.any():
            continue
        null = shift_null(c, FL, n, m)
        for k in np.where(ok)[0]:
            out["rate"][j, k] = obs[k + 1] / T[k + 1]
            d_obs = obs[k + 1] / T[k + 1] - obs[0] / T[0]
            d_null = null[k + 1] / T[k + 1] - null[0] / T[0]
            out["p"][j, k] = (1 + (d_null >= d_obs - 1e-9).sum()) / (1 + len(d_null))
    return out


def classify_session(s, cfg, lm, res=None):
    """Unit table of one session (one row per good unit with enough spikes) plus the
    control-zone membership matrix. Returns (units DataFrame, ctrl_is (n_units, n_ctrl),
    ctrl_labels, states)."""
    Z = label_states(s, cfg, lm, res)
    uids = s.select_units(cfg, cell_type=False)
    ctrl, ctrl_lab = control_zones(s, lm, Z, cfg)
    sp = spatial_tests(s, cfg, uids, Z, ctrl)
    tp = temporal_tests(s, cfg, uids, Z)
    alpha, min_rate = cfg.fields.alpha, cfg.visits.active_hz
    keep = tp["n_spikes"] >= cfg.units.min_spikes
    is_sp = (sp["p"] < alpha) & (sp["rate_in"] >= min_rate) & (sp["rate_in"] > sp["rate_out"])
    is_tp = (tp["p"] < alpha) & (tp["rate"] >= min_rate) & (tp["rate"] > tp["rate_maze"][:, None])
    T_run = Z["run"].sum() * s.dt
    U = pd.DataFrame(dict(unit=uids, cell_type=s.units.loc[uids, "cell_type"].to_numpy(),
                          n_spikes=tp["n_spikes"].astype(int), n_spikes_run=sp["n_spikes"].astype(int),
                          rate_run=sp["n_spikes"] / max(T_run, 1e-9), rate_maze=tp["rate_maze"]))
    for k, c in enumerate(SPATIAL):
        U[f"rate_{c}"], U[f"p_{c}"], U[f"is_{c}"] = sp["rate_in"][:, k], sp["p"][:, k], is_sp[:, k]
    for k, c in enumerate(TEMPORAL):
        U[f"rate_{c}"], U[f"p_{c}"], U[f"is_{c}"] = tp["rate"][:, k], tp["p"][:, k], is_tp[:, k]
    U["n_classes"] = U[[f"is_{c}" for c in CLASSES if c != "bridge"]].sum(1)
    Z["T"] = dict(run=T_run, maze=Z["maze"].sum() * s.dt,
                  **{c: Z[c].sum() * s.dt for c in CLASSES})
    Z["speed"] = {c: float(np.nanmedian(s.speed[Z[c]])) if Z[c].any() else np.nan
                  for c in ("run",) + SPATIAL}
    return U[keep].reset_index(drop=True), is_sp[keep][:, len(SPATIAL):], ctrl_lab, Z


# ----------------------------------------------------------------------------- run
def _session_rows(meta, U, ctrl_is, ctrl_lab, Z, alpha):
    picks = Z["picks"]
    timed = picks[np.isfinite(picks["t_pick"])] if picks is not None and len(picks) else None
    gaps = timed[np.isfinite(timed["t_next"])] if timed is not None else None
    base = dict(**meta, **{f"T_{k}_s": round(float(v), 1) for k, v in Z["T"].items()},
                **{f"speed_{k}_m_s": v for k, v in Z["speed"].items()},
                n_gaps=0 if picks is None else int(np.isfinite(picks["t_next"]).sum()),
                n_gaps_timed=0 if gaps is None else len(gaps),
                stay_med_s=float((gaps["t_pick"] - gaps["t_end"]).median()) if gaps is not None and len(gaps) else np.nan,
                carry_med_s=float((gaps["t_next"] - gaps["t_pick"]).median()) if gaps is not None and len(gaps) else np.nan)
    rows = []
    for ct in CELL_TYPES:
        sel = (U["cell_type"] == ct).to_numpy()
        n = int(sel.sum())
        r = dict(**base, cell_type=ct, n_units=n)
        for c in CLASSES:
            tested = Z["T"][c] > 0 and not (c in TEMPORAL and U.loc[sel, f"p_{c}"].isna().all())
            k = int(U.loc[sel, f"is_{c}"].sum()) if tested else np.nan
            r[f"n_{c}"] = k
            r[f"frac_{c}"] = k / n if n and tested else np.nan
            r[f"p_chance_{c}"] = float(st.binomtest(k, n, alpha, alternative="greater").pvalue) if n and tested else np.nan
            if c in CONTROLLED:
                cc = ctrl_is[sel][:, ctrl_lab == c].sum(0)
                r[f"n_ctrl_{c}"] = len(cc)
                enough = len(cc) >= MIN_CTRL and tested
                r[f"ctrl_{c}"] = float(np.median(cc)) if enough else np.nan
                r[f"ctrl_p_{c}"] = (1 + (cc >= k).sum()) / (1 + len(cc)) if enough else np.nan
        rows.append(r)
    return rows


def _summary(SES):
    rows = []
    for ct in CELL_TYPES:
        d0 = SES[SES["cell_type"] == ct]
        for grp, g in [("pooled", d0)] + list(d0.groupby("rat")):
            for c in CLASSES:
                gg = g.dropna(subset=[f"n_{c}"])
                gg = gg[gg["n_units"] > 0]
                if gg.empty:
                    continue
                r = dict(group=grp, cell_type=ct, cls=c, n_sessions=len(gg), n_units=int(gg["n_units"].sum()),
                         n_cells=int(gg[f"n_{c}"].sum()), frac=gg[f"n_{c}"].sum() / gg["n_units"].sum(),
                         frac_session_mean=gg[f"frac_{c}"].mean(), frac_session_sd=gg[f"frac_{c}"].std(),
                         sessions_above_chance=int((gg[f"p_chance_{c}"] < 0.05).sum()),
                         frac_ctrl=np.nan, p_vs_ctrl=np.nan)
                if c in CONTROLLED:
                    h = gg.dropna(subset=[f"ctrl_{c}"])
                    if len(h):
                        r["frac_ctrl"] = h[f"ctrl_{c}"].sum() / h["n_units"].sum()
                        d = (h[f"n_{c}"] - h[f"ctrl_{c}"]).to_numpy(float)
                        if len(d) >= 6 and np.any(d != 0):
                            r["p_vs_ctrl"] = float(st.wilcoxon(d).pvalue)
                rows.append(r)
    return pd.DataFrame(rows)


def _trends(SES):
    rows = []
    for ct in CELL_TYPES:
        d0 = SES[(SES["cell_type"] == ct) & (SES["n_units"] > 0)]
        for grp, g in [("pooled", d0)] + list(d0.groupby("rat")):
            for c in CLASSES:
                for x in ("day", "day_in_block"):
                    rows.append(dict(group=grp, cell_type=ct, cls=c, x=x,
                                     **S.spearman(g[x].to_numpy(float), g[f"frac_{c}"].to_numpy(float))))
    return pd.DataFrame(rows)


def _overlap(U):
    cl = [c for c in CLASSES if c != "bridge"]
    rows = []
    for ct in CELL_TYPES:
        d = U[U["cell_type"] == ct]
        for a in cl:
            r = dict(cell_type=ct, cls=a, n=int(d[f"is_{a}"].sum()))
            for b in cl:
                r[b] = int((d[f"is_{a}"] & d[f"is_{b}"]).sum())
            rows.append(r)
    return pd.DataFrame(rows)


def run(sessions, cfg):
    lm = mz.maze_from_cfg(cfg)
    U_all, ses_rows = [], []
    for row, s, _ in common.iter_loaded(sessions, cfg, with_fields=False):
        res = researcher_track(s, cfg)
        U, ctrl_is, ctrl_lab, Z = classify_session(s, cfg, lm, res)
        meta = dict(session_id=common.session_id(row), **{k: row[k] for k in META if k in row})
        U_all.append(U.assign(**meta))
        ses_rows += _session_rows(meta, U, ctrl_is, ctrl_lab, Z, cfg.fields.alpha)
    if not U_all:
        raise ValueError("no usable sessions")
    U, SES = pd.concat(U_all, ignore_index=True), pd.DataFrame(ses_rows)
    for c in CLASSES:
        U[f"is_{c}"] = U[f"is_{c}"].astype(bool)
    zones = pd.DataFrame([dict(cls=k, edge=f"{a}-{b}", length_m=lm.G.edges[a, b]["length"],
                               track_m=lm.bins["length"].to_numpy()[lm.edge_bins[(a, b)]].sum())
                          for k, edges in bridge_edges(lm).items() for a, b in edges])
    return dict(units=U, sessions=SES, summary=_summary(SES), trends=_trends(SES),
                overlap=_overlap(U), zones=zones)


# ----------------------------------------------------------------------------- plots
PANELS = ("goal", "start", "outside", "bridge_long", "bridge_short", "reward")


def plot(res, cfg):
    SES = res["sessions"]
    d = SES[SES["cell_type"] == "pyramidal"]
    fig, ax = P.figure(180, 105, nrows=2, ncols=3, sharex=True)
    for a, c in zip(ax.ravel(), PANELS):
        for rat, g in d.groupby("rat"):
            g = g.sort_values("day")
            col = P.RAT_COLORS.get(rat)
            a.plot(g["day"], 100 * g[f"frac_{c}"], "o-", ms=3, color=col, label=rat)
            if c in CONTROLLED:
                a.plot(g["day"], 100 * g[f"ctrl_{c}"] / g["n_units"].where(g["n_units"] > 0), ":", color=col,
                       label=f"{rat}, control locations")
        for day in sorted(d.loc[d["is_switch_day"].astype(bool), "day"].unique()):
            a.axvline(day - 0.5, color="0.85", lw=0.6, zorder=0)
        a.axhline(100 * cfg.fields.alpha, color="k", lw=0.6, ls="--")
        a.set_ylim(bottom=0)
        a.set_title(f"{LABELS[c]} cells")
    for a in ax[1]:
        a.set_xlabel("day")
    for a in ax[:, 0]:
        a.set_ylabel("% of pyramidal units")
    h, lab = ax[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, frameon=False, ncol=len(lab), loc="outside lower center")

    lm = mz.maze_from_cfg(cfg)
    bb = bridge_bins(lm)
    v = np.full(lm.n_bins, np.nan)
    v[bb["bridge_long"]], v[bb["bridge_short"]] = 1.0, 0.0
    fig2, ax2 = P.figure(120, 65)
    P.draw_maze(ax2, lm, values=v, cmap="coolwarm", vmin=-0.15, vmax=1.15, node_size=6)
    from matplotlib.lines import Line2D
    cm = P.plt.get_cmap("coolwarm")
    ax2.legend([Line2D([], [], color=cm(1.0 / 1.3 + 0.15 / 1.3), lw=2.5),
                Line2D([], [], color=cm(0.15 / 1.3), lw=2.5)],
               ["long bridge", "short bridge"], frameon=False, loc="lower left")
    return [("cell_classes", fig), ("cell_classes_zones", fig2)]


# ----------------------------------------------------------------------------- LaTeX report
CAPTIONS = {
    "cell_classes": (
        r"\textbf{Cell classes across training days} (good pyramidal units). One line per rat; "
        r"dotted lines: the same test at control locations with similar running occupancy (other "
        r"nodes for the goal, random within-island corridors of the same length for the bridges). "
        r"Dashed black line: the 5\,\% expected from the test alone. Grey vertical lines: first day "
        r"with a new goal. Classes are not exclusive."),
    "cell_classes_zones": (
        r"\textbf{Long and short bridges.} The five corridors that join two islands, split at "
        r"1\,m: two long ones (red) and three short ones (blue). Node zones are not part of a bridge."),
}


def _pct(k, n):
    if pd.isna(k) or not n:
        return "--"
    return f"{int(k)} ({100 * k / n:.0f}%)"


def latex(T):
    from .. import report as R
    SES, SUM, TR, ZN = T["sessions"], T["summary"], T["trends"], T["zones"]
    pyr = SES[SES["cell_type"] == "pyramidal"]
    edges = {k: ", ".join(R.esc(e.replace("-", "–")) for e in g["edge"]) for k, g in ZN.groupby("cls")}
    rng_m = {}
    for k, g in ZN.groupby("cls"):
        lo, hi = f"{g['length_m'].min():.1f}", f"{g['length_m'].max():.1f}"
        rng_m[k] = lo if lo == hi else f"{lo}--{hi}"
    n_ses = pyr["session_id"].nunique()
    spd = {k: pyr[f"speed_{k}_m_s"].median() for k in ("run", "bridge_long", "bridge_short")
           if f"speed_{k}_m_s" in pyr}
    stay, carry = pyr["stay_med_s"].median(), pyr["carry_med_s"].median()
    timed = pyr["n_gaps_timed"].sum() / max(pyr["n_gaps"].sum(), 1)
    text = (
        rf"Every good unit with at least 100 spikes ({n_ses} sessions with usable trial timing) was "
        r"tested for a firing preference for six behavioural states. A unit belongs to a class when its rate in that state is at least "
        r"0.5\,Hz and exceeds its reference rate with $p<0.05$ (one-sided) against all circular "
        r"shifts ($\geq 20$\,s) of its spike train relative to behaviour. Classes are not exclusive, "
        r"and the test alone admits at most 5\,\% of units." "\n"
        r"\begin{description}\setlength{\itemsep}{0pt}" "\n"
        r"\item[Goal] within 0.5\,m track distance of the goal node, against the rest of the maze "
        r"(running, in trial)." "\n"
        r"\item[Start] within 0.5\,m of the start node of the current trial. The start node changes "
        r"every trial, so this is a position relative to the trial and not a fixed place." "\n"
        rf"\item[Long bridge] on the inter-island corridors {edges.get('bridge_long', '--')} "
        rf"({rng_m.get('bridge_long', '--')}\,m)." "\n"
        rf"\item[Short bridge] on the inter-island corridors {edges.get('bridge_short', '--')} "
        rf"({rng_m.get('bridge_short', '--')}\,m)." "\n"
        r"\item[Outside maze] while the rat is off the maze (carried from the goal to the next start, "
        r"and before the first and after the last trial), against all in-trial time." "\n"
        r"\item[At goal after arrival] from goal arrival until the rat is picked up, against all "
        r"in-trial time." "\n"
        r"\end{description}" "\n"
        r"The rat's own track is frozen between trials. The pick-up was therefore timed from the "
        r"researcher track, as the last moment at which the researcher was within 1.5\,m of where "
        rf"the rat ended the trial ({100 * timed:.0f}\,\% of inter-trial gaps could be timed). "
        rf"The rat stayed at the goal for a median of {stay:.0f}\,s and was then carried for "
        rf"{carry:.0f}\,s, so ``between trials'' is mostly time at the goal; it is listed as its own "
        r"class and is not part of the outside-maze class. All states are restricted to the span of "
        r"the spike record." "\n\n")

    P_ = SUM[(SUM["cell_type"] == "pyramidal") & (SUM["group"] == "pooled")].set_index("cls")
    items = []
    for c in ("goal", "start", "bridge_long", "bridge_short", "outside", "reward"):
        if c not in P_.index:
            continue
        r = P_.loc[c]
        line = (rf"\textbf{{{LABELS[c].capitalize()}}}: {R.fmt(int(r['n_cells']))} of "
                rf"{R.fmt(int(r['n_units']))} pyramidal units ({100 * r['frac']:.1f}\,\%); above the "
                rf"5\,\% chance level in {int(r['sessions_above_chance'])} of {int(r['n_sessions'])} sessions")
        if pd.notna(r["frac_ctrl"]):
            line += rf"; control locations give {100 * r['frac_ctrl']:.1f}\,\%"
            if pd.notna(r["p_vs_ctrl"]):
                line += rf" (Wilcoxon across sessions, $p = {R.fmt(float(r['p_vs_ctrl'])).strip('$')}$)"
        items.append(line + ".")
    if items:
        text += (r"\begin{itemize}\setlength{\itemsep}{0pt}" "\n" + "\n".join(r"\item " + i for i in items)
                 + "\n" r"\end{itemize}" "\n\n")
    text += (r"A spatially tuned population yields cells of this kind for any location, which is why goal "
             r"and bridge counts are compared with control locations of similar running occupancy (zones "
             r"around other nodes; random sets of within-island corridors of the same total length). ")
    if len(spd) == 3 and all(pd.notna(v) for v in spd.values()):
        text += (r"The controls are not matched for running speed: the median was "
                 rf"{spd['bridge_long']:.2f}\,m/s on long bridges and {spd['bridge_short']:.2f}\,m/s on short "
                 rf"bridges, against {spd['run']:.2f}\,m/s over the whole maze. ")
    text += (r"Start cells have no location control because the start zone moves with the trial. "
             r"The outside-maze and at-goal classes contrast behavioural states that also differ in "
             r"movement (carried or eating against running and pausing on the maze), so they describe "
             r"when a unit fires and not what it encodes." "\n\n")

    t = TR[(TR["cell_type"] == "pyramidal") & (TR["group"] != "pooled") & TR["cls"].isin(PANELS)]
    sig = t[t["p"] < 0.05]
    if len(sig):
        bits = [rf"{LABELS[r.cls]} cells in {R.esc(r.group)} against "
                rf"{'training day' if r.x == 'day' else 'day within the goal block'} "
                rf"($\rho = {r.rho:.2f}$, $p = {R.fmt(float(r.p)).strip('$')}$)" for r in sig.itertuples()]
        text += (rf"Of the {len(t)} per-rat rank correlations between the fraction of cells and training "
                 rf"(Table~\ref{{tab:cell_classes:trends}}), {len(sig)} reach $p<0.05$ uncorrected: "
                 + "; ".join(bits) + ". ")
        both = sig.groupby(["cls", "x"])["rho"].agg(lambda v: len(v) > 1 and np.sign(v).nunique() == 1).any()
        text += (rf"With {len(t)} tests, {0.05 * len(t):.1f} would be expected by chance alone"
                 + (". " if both else ", and no class changes in the same direction in both rats. ") + "\n\n")
    elif len(t):
        text += (rf"None of the {len(t)} per-rat rank correlations between the fraction of cells and "
                 r"training reaches $p<0.05$ (Table~\ref{tab:cell_classes:trends}): the counts do not "
                 "change systematically with learning.\n\n")

    for rat, g in pyr.groupby("rat"):
        g = g.sort_values("day")
        tab = pd.DataFrame({"day": g["day"].astype(int), "goal node": g["goal"].astype(int).astype(str),
                            "units": g["n_units"].astype(int)})
        for c in ("goal", "start", "bridge_long", "bridge_short", "bridge", "outside", "reward"):
            tab[LABELS[c]] = [_pct(k, n) for k, n in zip(g[f"n_{c}"], g["n_units"])]
        text += R.table(tab, rf"{R.esc(rat)}: number of good pyramidal units in each class per session "
                        r"(percentage of the units tested). Classes are not exclusive; ``any bridge'' "
                        r"pools the five corridors and is not the sum of long and short.",
                        f"tab:cell_classes:{rat}", headers={"at goal after arrival": "at goal"},
                        size=r"\scriptsize")

    s = SUM[SUM["cls"] != "bridge"].copy()
    s["cls"] = s["cls"].map(LABELS)
    for c in ("frac", "frac_session_mean", "frac_session_sd", "frac_ctrl"):
        s[c] = 100 * s[c]
    text += R.table(s, r"Cell classes pooled over sessions. \%: of all units tested; session mean and SD "
                    r"of the per-session percentage; above chance: sessions whose count exceeds the 5\,\% "
                    r"expected from the test (binomial, $p<0.05$); control: same test at matched control "
                    r"locations; $p$: Wilcoxon signed-rank across sessions, observed against control count.",
                    "tab:cell_classes:summary",
                    cols=["cell_type", "group", "cls", "n_sessions", "n_units", "n_cells", "frac",
                          "frac_session_mean", "frac_session_sd", "sessions_above_chance", "frac_ctrl", "p_vs_ctrl"],
                    headers=dict(cell_type="cell type", group="rat", cls="class", n_sessions="sessions",
                                 n_units="units", n_cells="cells", frac=r"\%", frac_session_mean=r"mean \%",
                                 frac_session_sd=r"SD \%", sessions_above_chance="above chance",
                                 frac_ctrl=r"control \%", p_vs_ctrl="$p$"), size=r"\scriptsize")
    t = TR[(TR["cell_type"] == "pyramidal") & TR["cls"].isin(PANELS)].copy()
    t["cls"] = t["cls"].map(LABELS)
    t["x"] = t["x"].map({"day": "training day", "day_in_block": "day in goal block"})
    text += R.table(t, r"Change with learning: Spearman correlation across sessions between the fraction "
                    r"of pyramidal units in a class and training day, or day since the last goal switch. "
                    r"Uncorrected $p$-values.", "tab:cell_classes:trends",
                    cols=["group", "cls", "x", "rho", "p", "n"],
                    headers=dict(group="rat", cls="class", x="against", rho=r"$\rho$", p="$p$", n="sessions"),
                    size=r"\scriptsize")
    return text
