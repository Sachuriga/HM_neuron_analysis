"""HexMaze geometry and a linearised (graph) representation of the maze.

The node table and connectivity rules are ported from ``HM_Rat_Analysis.maze``
(pixel coordinates from the tracker, adjacency < 65 px plus six manual edges,
corner nodes 501/502 removed). Everything here is in METRES.

Linearisation: every maze position is projected onto its nearest edge. If it
lies within ``node_radius_m`` of an end it belongs to that NODE zone, otherwise
to one of the equal-length bins along the EDGE ("hedge"). Bins are joined in a
bin graph whose shortest-path distances give track distance between any two
places, e.g. distance of a place field to the goal node.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path

SCALE_X = 2352 / 2 / 9          # px per metre, as in the tracker / HM_Rat_Analysis
SCALE_Y = 1424 / 2 / 5
MAZE_EXTENT = (0.0, 9.0, 0.0, 5.0)
NODE_CSV = Path(__file__).resolve().parent / "data" / "node_list_new.csv"
ADJACENCY_THRESHOLD_PX = 65
EXCLUDED_NODES = (501, 502)
MANUAL_EDGES = ((121, 302), (324, 401), (305, 220), (404, 223), (201, 124), (224, 218))


def island_of(node: int) -> int:
    return int(node) // 100


@lru_cache(maxsize=1)
def node_table() -> pd.DataFrame:
    df = pd.read_csv(NODE_CSV, header=None, names=["node", "x_px", "y_px"])
    df["node"] = df["node"].astype(int)
    df["x"] = df["x_px"] / SCALE_X
    df["y"] = df["y_px"] / SCALE_Y
    return df[~df["node"].isin(EXCLUDED_NODES)].reset_index(drop=True)


@lru_cache(maxsize=1)
def build_graph() -> nx.Graph:
    """Maze graph, int node ids, ``pos`` in metres, edge ``length`` in metres."""
    nt = node_table()
    G = nx.Graph()
    for r in nt.itertuples():
        G.add_node(r.node, pos=(r.x, r.y))
    px = nt[["x_px", "y_px"]].to_numpy(float)
    ids = nt["node"].to_numpy()
    d = np.linalg.norm(px[:, None] - px[None], axis=-1)
    pairs = [(ids[i], ids[j]) for i, j in zip(*np.where(np.triu(d < ADJACENCY_THRESHOLD_PX, 1)))]
    for a, b in list(pairs) + list(MANUAL_EDGES):
        if a in G and b in G:
            pa, pb = np.array(G.nodes[a]["pos"]), np.array(G.nodes[b]["pos"])
            G.add_edge(int(a), int(b), length=float(np.linalg.norm(pa - pb)))
    return G


def edge_key(a, b) -> tuple:
    a, b = int(a), int(b)
    return (a, b) if a < b else (b, a)


class LinearMaze:
    """Bins of the linearised maze plus projection of xy positions onto them.

    Attributes
    ----------
    bins : DataFrame, one row per bin with columns
        ``kind`` ('node'|'edge'), ``node`` (node id or -1), ``edge`` (a, b) or None,
        ``s`` (distance along edge from a), ``length`` (m), ``x``, ``y`` (centre).
    dist : (n_bins, n_bins) track distance between bin centres, metres.
    """

    def __init__(self, node_radius_m=0.08, edge_bin_m=0.04, max_dist_m=0.15, G=None):
        self.G = build_graph() if G is None else G
        self.r = float(node_radius_m)
        self.max_dist = float(max_dist_m)
        rows, links = [], []
        self.node_bin = {}
        for n, d in self.G.nodes(data=True):
            self.node_bin[n] = len(rows)
            rows.append(dict(kind="node", node=n, edge=None, s=0.0,
                             length=2 * self.r, x=d["pos"][0], y=d["pos"][1]))
        self.edges = sorted(edge_key(a, b) for a, b in self.G.edges())
        self.edge_bins = {}
        for a, b in self.edges:
            L = self.G.edges[a, b]["length"]
            pa, pb = np.array(self.G.nodes[a]["pos"]), np.array(self.G.nodes[b]["pos"])
            span = max(L - 2 * self.r, 1e-6)
            k = max(1, int(round(span / edge_bin_m)))
            w = span / k
            idx = []
            for i in range(k):
                s = self.r + (i + 0.5) * w
                p = pa + (pb - pa) * s / L
                idx.append(len(rows))
                rows.append(dict(kind="edge", node=-1, edge=(a, b), s=s, length=w, x=p[0], y=p[1]))
            self.edge_bins[(a, b)] = np.array(idx)
            chain = [self.node_bin[a]] + idx + [self.node_bin[b]]
            cs = [0.0] + [self.r + (i + 0.5) * w for i in range(k)] + [L]
            for i in range(len(chain) - 1):
                links.append((chain[i], chain[i + 1], cs[i + 1] - cs[i]))
        self.bins = pd.DataFrame(rows)
        n = len(rows)
        i, j, w = zip(*links)
        A = csr_matrix((w + w, (i + j, j + i)), shape=(n, n))
        self.adj = A
        self.dist = shortest_path(A, directed=False)
        self._seg_a = np.array([self.G.nodes[a]["pos"] for a, _ in self.edges])
        self._seg_b = np.array([self.G.nodes[b]["pos"] for _, b in self.edges])

    @property
    def n_bins(self):
        return len(self.bins)

    def is_node(self) -> np.ndarray:
        return (self.bins["kind"] == "node").to_numpy()

    def element_of_bin(self) -> np.ndarray:
        """Label per bin: 'N101' for a node, 'E101-102' for an edge."""
        return np.array([f"N{r.node}" if r.kind == "node" else f"E{r.edge[0]}-{r.edge[1]}"
                         for r in self.bins.itertuples()])

    def dist_to_node(self, node: int) -> np.ndarray:
        return self.dist[:, self.node_bin[int(node)]]

    def project(self, x, y, chunk=20000):
        """Bin index per position (-1 when NaN or further than max_dist from the maze)."""
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        out = np.full(x.shape, -1, int)
        a, b = self._seg_a, self._seg_b
        ab = b - a
        L2 = (ab ** 2).sum(1)
        L = np.sqrt(L2)
        ok_idx = np.where(np.isfinite(x) & np.isfinite(y))[0]
        for c0 in range(0, len(ok_idx), chunk):
            ii = ok_idx[c0:c0 + chunk]
            p = np.stack([x[ii], y[ii]], 1)
            t = np.clip(((p[:, None] - a[None]) * ab[None]).sum(-1) / L2[None], 0, 1)
            proj = a[None] + t[..., None] * ab[None]
            d = np.linalg.norm(p[:, None] - proj, axis=-1)
            e = d.argmin(1)
            dmin = d[np.arange(len(ii)), e]
            s = t[np.arange(len(ii)), e] * L[e]
            for k in np.unique(e):
                m = e == k
                ea, eb = self.edges[k]
                Lk = L[k]
                sk = s[m]
                res = np.empty(m.sum(), int)
                near_a, near_b = sk < self.r, sk > Lk - self.r
                res[near_a] = self.node_bin[ea]
                res[near_b] = self.node_bin[eb]
                mid = ~(near_a | near_b)
                eb_idx = self.edge_bins[(ea, eb)]
                w = (Lk - 2 * self.r) / len(eb_idx)
                j = np.clip(((sk[mid] - self.r) / w).astype(int), 0, len(eb_idx) - 1)
                res[mid] = eb_idx[j]
                out[ii[m]] = res
            out[ii[dmin > self.max_dist]] = -1
        return out


@lru_cache(maxsize=8)
def get_maze(node_radius_m=0.08, edge_bin_m=0.04, max_dist_m=0.15) -> LinearMaze:
    return LinearMaze(node_radius_m, edge_bin_m, max_dist_m)


def maze_from_cfg(cfg) -> LinearMaze:
    return get_maze(cfg.lin.node_radius_m, cfg.lin.edge_bin_m, cfg.lin.max_dist_m)
