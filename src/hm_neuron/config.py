"""All paths and analysis parameters in one place.

Defaults live here; override any of them with a YAML file
(``hm --config my.yaml ...`` or ``Config.from_yaml``). Only the keys you set
are changed, e.g.::

    data_root: /Volumes/genzel/Rat/HM/Rat_HM_Neuron
    fields:
      speed_min_m_s: 0.05
"""

from __future__ import annotations

import dataclasses as dc
from pathlib import Path

import yaml


@dc.dataclass
class UnitSel:
    quality: tuple = ("good",)          # quality_label values to keep
    cell_type: tuple = ("pyramidal",)   # cell_type values for field analyses
    min_spikes: int = 100               # spikes during running


@dc.dataclass
class Linearise:
    node_radius_m: float = 0.08         # positions within this of a node = node zone
    edge_bin_m: float = 0.04            # bin length along edges
    max_dist_m: float = 0.15            # positions further than this from the maze are dropped


@dc.dataclass
class Fields:
    speed_min_m_s: float = 0.05         # running threshold (old pipeline default)
    speed_smooth_s: float = 0.25
    smooth_sigma_m: float = 0.05        # Gaussian smoothing over graph distance
    min_occ_s: float = 0.30             # minimum occupancy for a valid bin
    field_frac: float = 0.30            # field = connected bins >= frac * peak
    min_peak_hz: float = 1.0
    min_field_m: float = 0.10           # minimum field extent (sum of bin lengths)
    n_shuffles: int = 500               # circular-shift shuffles for SI significance
    shuffle_min_shift_s: float = 20.0
    alpha: float = 0.05
    seed: int = 0


@dc.dataclass
class Theta:
    band_hz: tuple = (6.0, 10.0)
    delta_hz: tuple = (2.0, 4.0)
    lfp_down_hz: float = 250.0          # cached LFP sample rate
    select_window_s: float = 600.0      # LFP window used to choose the theta channel
    min_spikes: int = 50


@dc.dataclass
class Visits:
    min_dur_s: float = 0.2              # shorter visits ignored
    active_hz: float = 0.5              # unit is "active" on a visit above this rate


@dc.dataclass
class Goal:
    radius_m: float = 0.5               # track distance defining the goal zone
    n_trial_groups: int = 3             # early / middle / late thirds of a session
    window_trials: int = 3              # sliding window for trial-resolved SI
    decode_bin_s: float = 1.0           # time bin for early-vs-late map likelihood ratio
    n_shuffles_window: int = 20         # shuffles for bias-corrected window SI


@dc.dataclass
class Config:
    data_root: Path = Path("/Volumes/genzel/Rat/HM/Rat_HM_Neuron")
    cache_dir: Path = Path.home() / "hm_cache"
    out_dir: Path = Path("results")
    rats: tuple = ("Rat5", "Rat6")
    units: UnitSel = dc.field(default_factory=UnitSel)
    lin: Linearise = dc.field(default_factory=Linearise)
    fields: Fields = dc.field(default_factory=Fields)
    theta: Theta = dc.field(default_factory=Theta)
    visits: Visits = dc.field(default_factory=Visits)
    goal: Goal = dc.field(default_factory=Goal)
    fig_format: tuple = ("pdf", "png")
    dpi: int = 300

    @classmethod
    def from_yaml(cls, path=None) -> "Config":
        cfg = cls()
        if path is None:
            return cfg
        with open(path) as fh:
            over = yaml.safe_load(fh) or {}
        _update(cfg, over)
        return cfg

    def to_dict(self) -> dict:
        def conv(v):
            if isinstance(v, Path):
                return str(v)
            if isinstance(v, tuple):
                return list(v)
            return v
        return {k: (({kk: conv(vv) for kk, vv in dc.asdict(v).items()})
                    if dc.is_dataclass(v) else conv(v))
                for k, v in ((f.name, getattr(self, f.name)) for f in dc.fields(self))}

    def key(self, section: str) -> str:
        """Short hash of a config section, used to version cache files."""
        import hashlib, json
        d = self.to_dict()
        sub = {s: d[s] for s in section.split("+")}
        return hashlib.md5(json.dumps(sub, sort_keys=True, default=str).encode()).hexdigest()[:8]


def _update(obj, over: dict):
    for k, v in over.items():
        if not hasattr(obj, k):
            raise KeyError(f"unknown config key: {k}")
        cur = getattr(obj, k)
        if dc.is_dataclass(cur) and isinstance(v, dict):
            _update(cur, v)
        elif isinstance(cur, Path):
            setattr(obj, k, Path(v).expanduser())
        elif isinstance(cur, tuple):
            setattr(obj, k, tuple(v))
        else:
            setattr(obj, k, type(cur)(v) if cur is not None else v)
