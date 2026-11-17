"""cell_classes: zones, exact circular-shift null, classification, LaTeX section."""

import numpy as np
import pandas as pd
from scipy import fft as sfft

from hm_neuron import maze as mz
from hm_neuron.analyses import cell_classes as cc

from conftest import make_session


def test_bridges_split_long_short(lm):
    e = cc.bridge_edges(lm)
    assert sorted(e["bridge_long"]) == [(121, 302), (223, 404)]
    assert sorted(e["bridge_short"]) == [(124, 201), (220, 305), (324, 401)]
    b = cc.bridge_bins(lm)
    assert not (b["bridge_long"] & b["bridge_short"]).any()
    assert (b["bridge"] == (b["bridge_long"] | b["bridge_short"])).all()
    assert not b["bridge"][lm.is_node()].any()           # node zones are not bridge


def test_shift_null_matches_brute_force():
    rng = np.random.default_rng(0)
    n, m = 500, 40
    c = rng.poisson(0.3, n).astype(float)
    L = (rng.random((2, n)) < 0.2).astype(float)
    got = cc.shift_null(c, sfft.rfft(L, axis=1), n, m)
    want = np.array([[np.sum(c * np.roll(L[k], -j)) for j in range(m, n - m + 1)] for k in range(2)])
    assert got.shape == want.shape and np.allclose(got, want)


def test_classifies_goal_and_bridge_cells(cfg, lm):
    goal = 107
    dg = lm.dist_to_node(goal)
    bb = cc.bridge_bins(lm)
    fns = {1: lambda b: np.where(dg[b] <= 0.3, 15.0, 0.2),          # goal cell
           2: lambda b: np.where(bb["bridge_long"][b], 15.0, 0.2),  # long-bridge cell
           3: lambda b: np.full(len(b), 3.0)}                       # untuned
    s = make_session(lm, fns, n_steps=4000, goal=goal, seed=3)
    U, ctrl_is, ctrl_lab, Z = cc.classify_session(s, cfg, lm, res=None)
    U = U.set_index("unit")
    assert U.at[1, "is_goal"] and not U.at[1, "is_bridge_long"]
    assert U.at[2, "is_bridge_long"] and U.at[2, "is_bridge"] and not U.at[2, "is_bridge_short"]
    assert not U.loc[3, [f"is_{c}" for c in cc.SPATIAL]].any()
    assert U["p_outside"].isna().all() and not U["is_outside"].any()   # no researcher track
    assert set(ctrl_lab) <= set(cc.CONTROLLED) and ctrl_is.shape == (len(U), len(ctrl_lab))


def test_pickup_splits_gap_into_reward_and_outside(cfg, lm):
    s = make_session(lm, {1: lambda b: np.full(len(b), 5.0)}, n_steps=600, goal=107, n_trials=3)
    # carve 20 s gaps after trials 1 and 2; the rat "ends" each trial where it stands
    for k in (1, 2):
        s.trials.loc[s.trials["trial"] == k, "t_end"] -= 20.0
    for q in s.trials.itertuples():
        gap = (s.t > q.t_end) & (s.trial_id == q.trial)
        s.trial_id[gap] = -1
    s.in_trial = s.trial_id >= 0
    gaps = s.trials.iloc[:2]
    ends = np.array([[s.x[i], s.y[i]] for i in np.searchsorted(s.t, gaps["t_end"], side="right") - 1])
    nodes = list(lm.G.nodes)
    xy = np.array([lm.G.nodes[n]["pos"] for n in nodes])
    far = np.linalg.norm(xy[:, None] - ends[None], axis=-1).min(1)
    assert far.max() > cc.AMBIGUOUS_M
    s.trials["start_node"] = nodes[int(far.argmax())]     # next start far from both end points
    rt, rx, ry = [], [], []
    for q in gaps.itertuples():
        i = np.searchsorted(s.t, q.t_end, side="right") - 1
        tt = np.arange(q.t_end + 0.5, q.t_end + 19.5, 0.5)
        near = tt <= q.t_end + 12.0              # researcher at the rat for 12 s, then walks away
        rt.append(tt)
        rx.append(np.where(near, s.x[i] + 0.5, s.x[i] + 6.0))
        ry.append(np.full(len(tt), s.y[i]))
    res = tuple(np.concatenate(v) for v in (rt, rx, ry))
    picks = cc.pickups(s, res, lm)
    timed = picks[np.isfinite(picks["t_next"])]
    assert np.allclose(timed["t_pick"] - timed["t_end"], 12.0)
    Z = cc.label_states(s, cfg, lm, res)
    assert abs(Z["outside"].sum() * s.dt - 2 * 8.0) < 1.0
    assert not (Z["outside"] & (Z["reward"] | s.in_trial)).any()


def test_latex_section_from_tables():
    rng = np.random.default_rng(0)
    rows = []
    for rat in ("Rat5", "Rat6"):
        for day in range(1, 9):
            for ct in cc.CELL_TYPES:
                r = dict(session_id=f"{rat}_{day}", rat=rat, date=20260600 + day, day=day, goal=107,
                         day_in_block=(day - 1) % 4, is_switch_day=day % 4 == 1, is_recall_day=False,
                         cell_type=ct, n_units=40, n_gaps=20, n_gaps_timed=19, stay_med_s=45.0, carry_med_s=17.0,
                         speed_run_m_s=0.2, speed_bridge_long_m_s=0.4, speed_bridge_short_m_s=0.3)
                for c in cc.CLASSES:
                    k = int(rng.integers(0, 12))
                    r.update({f"n_{c}": k, f"frac_{c}": k / 40, f"p_chance_{c}": 0.2})
                    if c in cc.CONTROLLED:
                        r.update({f"n_ctrl_{c}": 30, f"ctrl_{c}": 4.0, f"ctrl_p_{c}": 0.3})
                rows.append(r)
    SES = pd.DataFrame(rows)
    zones = pd.DataFrame(dict(cls=["bridge_long", "bridge_short"], edge=["121-302", "124-201"],
                              length_m=[1.99, 0.68], track_m=[1.8, 0.5]))
    T = dict(sessions=SES, summary=cc._summary(SES), trends=cc._trends(SES), zones=zones)
    tex = cc.latex(T)
    assert r"\begin{description}" in tex and tex.count(r"\begin{table}") == 4
    assert "121--302" in tex and "long bridge" in tex
    stripped = tex.replace(r"\_", "")
    for lab in ("tab:cell_classes:Rat5", "tab:cell_classes:Rat6", "tab:cell_classes:summary",
                "tab:cell_classes:trends"):
        stripped = stripped.replace(lab, "")
    assert "_" not in stripped                    # no unescaped underscores
    assert mz.island_of(121) != mz.island_of(302)
