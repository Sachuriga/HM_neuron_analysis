"""Synthetic sessions: a rat random-walks the real maze graph at constant speed."""

import numpy as np
import pandas as pd
import pytest

from hm_neuron import maze as mz
from hm_neuron.config import Config
from hm_neuron.io import Session


@pytest.fixture(scope="session")
def cfg(tmp_path_factory):
    c = Config()
    c.cache_dir = tmp_path_factory.mktemp("cache")
    c.fields.n_shuffles = 100
    return c


@pytest.fixture(scope="session")
def lm(cfg):
    return mz.maze_from_cfg(cfg)


def make_session(lm, rate_fns, n_steps=900, speed=0.3, fs=30.0, seed=0, n_trials=6, goal=None):
    """rate_fns: {uid: f(bin_index_array) -> Hz}. Returns Session."""
    rng = np.random.default_rng(seed)
    G = lm.G
    node = list(G.nodes)[0]
    xs, ys = [], []
    for _ in range(n_steps):
        nxt = rng.choice(list(G.neighbors(node)))
        a, b = np.array(G.nodes[node]["pos"]), np.array(G.nodes[nxt]["pos"])
        n = max(2, int(np.linalg.norm(b - a) / speed * fs))
        f = np.linspace(0, 1, n, endpoint=False)
        xs.append(a[0] + f * (b[0] - a[0]))
        ys.append(a[1] + f * (b[1] - a[1]))
        node = nxt
    x, y = np.concatenate(xs), np.concatenate(ys)
    t = np.arange(len(x)) / fs
    b = lm.project(x, y)
    edges_t = np.linspace(t[0], t[-1] + 1e-6, n_trials + 1)
    trial_id = np.searchsorted(edges_t, t, side="right")
    trials = pd.DataFrame(dict(trial=np.arange(1, n_trials + 1), t_start=edges_t[:-1], t_end=edges_t[1:],
                               start_node=0, goal=goal or 0, n_steps=1, path_m=1.0))
    spikes = {}
    for u, fn in rate_fns.items():
        lam = fn(b) / fs
        k = rng.poisson(lam)
        spikes[u] = np.sort(np.repeat(t, k) + rng.uniform(0, 1 / fs, k.sum()))
    units = pd.DataFrame(dict(quality_label="good", cell_type="pyramidal"), index=list(rate_fns))
    goal = goal or int(list(G.nodes)[10])
    meta = dict(rat="RatX", date="20990101", day=1, repeat=1, session=1, goal=goal, block=1,
                day_in_block=0, prev_goal=pd.NA, is_switch_day=True, is_recall_day=False)
    return Session(rat="RatX", date="20990101", meta=meta, t=t, x=x, y=y,
                   speed=np.full(len(t), speed), bin=b, in_trial=np.ones(len(t), bool), trial_id=trial_id,
                   steps=pd.DataFrame(), trials=trials, units=units, spikes=spikes)
