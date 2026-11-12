import numpy as np
import pytest

from hm_neuron import fields as fl, maze as mz, stats as S
from hm_neuron.analyses import field_count, info_drop
from hm_neuron.config import Config

from conftest import make_session


# ----------------------------------------------------------------------------- maze
def test_graph_basic():
    G = mz.build_graph()
    assert G.number_of_nodes() == 96
    assert all(d["length"] > 0.2 for *_, d in G.edges(data=True))


def test_projection_nodes_and_edges(lm):
    for n in list(lm.G.nodes)[:10]:
        x, y = lm.G.nodes[n]["pos"]
        assert lm.project([x], [y])[0] == lm.node_bin[n]
    a, b = lm.edges[0]
    pa, pb = np.array(lm.G.nodes[a]["pos"]), np.array(lm.G.nodes[b]["pos"])
    mid = (pa + pb) / 2
    bi = lm.project([mid[0]], [mid[1]])[0]
    assert bi in lm.edge_bins[(a, b)]
    assert lm.project([100.0], [100.0])[0] == -1          # far off the maze


def test_track_distance(lm):
    a, b = lm.edges[0]
    L = lm.G.edges[a, b]["length"]
    assert lm.dist[lm.node_bin[a], lm.node_bin[b]] == pytest.approx(L, rel=1e-6)
    assert np.allclose(lm.dist, lm.dist.T)


# ----------------------------------------------------------------------------- fields
@pytest.fixture(scope="module")
def synth(lm):
    centre = lm.edge_bins[lm.edges[5]][len(lm.edge_bins[lm.edges[5]]) // 2]
    d = lm.dist[centre]
    rate_fns = {
        1: lambda b: np.where(b >= 0, 0.2 + 15 * np.exp(-0.5 * (d[b] / 0.12) ** 2), 0),   # one field
        2: lambda b: np.full(b.shape, 3.0),                                              # flat
    }
    return make_session(lm, rate_fns, n_steps=1500), centre


def test_field_detection(synth, cfg, lm):
    s, centre = synth
    R = fl.session_fields(s, cfg, refresh=True, lm=lm)
    u = R["units"].set_index("unit")
    assert u.loc[1, "is_place"] and not u.loc[2, "is_place"]
    f = R["fields"][R["fields"]["unit"] == 1]
    assert len(f) == 1
    assert lm.dist[f["peak_bin"].iloc[0], centre] < 0.1
    assert 0.1 < f["size_m"].iloc[0] < 0.8
    assert u.loc[1, "si"] > u.loc[2, "si"]


def test_skaggs_uniform_is_zero():
    occ = np.ones(10)
    assert fl.skaggs_info(np.full(10, 2.0), occ, np.ones(10, bool))[0] == pytest.approx(0, abs=1e-12)


# ----------------------------------------------------------------------------- stats
def test_fdr_bh():
    q = S.fdr_bh([0.01, 0.04, 0.03, 0.5])
    # sorted .01 .03 .04 .5 -> .04 .06 .0533 .5 -> monotone: .04 .0533 .0533 .5
    assert np.allclose(q, [0.04, 0.053333, 0.053333, 0.5], atol=1e-4)


def test_rayleigh():
    rng = np.random.default_rng(0)
    assert S.rayleigh(rng.uniform(-np.pi, np.pi, 500))[1] > 0.01
    assert S.rayleigh(rng.vonmises(0, 2, 500))[1] < 1e-6


def test_field_count_model_selection():
    rng = np.random.default_rng(0)
    k = rng.negative_binomial(1.5, 0.2, 3000)
    k = k[k > 0]
    assert field_count.fit(k).iloc[0]["model"] in ("negbin", "lognormal")
    k = rng.poisson(4, 3000)
    k = k[k > 0]
    best = field_count.fit(k)
    assert best.set_index("model").loc["poisson", "delta_aic"] < 4


def test_bimodality():
    rng = np.random.default_rng(0)
    assert info_drop.bimodality_coefficient(rng.normal(size=2000)) < 0.555
    assert info_drop.bimodality_coefficient(np.r_[rng.normal(-3, .5, 1000), rng.normal(3, .5, 1000)]) > 0.555


def test_config_yaml(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("fields:\n  speed_min_m_s: 0.1\ncache_dir: ~/x\n")
    c = Config.from_yaml(p)
    assert c.fields.speed_min_m_s == 0.1 and str(c.cache_dir).endswith("x")
    with pytest.raises(KeyError):
        p.write_text("nope: 1\n")
        Config.from_yaml(p)
