"""Analysis registry.

Every analysis module defines

* ``NAME``, ``TITLE`` – CLI name and one-line description
* ``run(sessions, cfg) -> dict[str, DataFrame]`` – ``sessions`` is a catalog
  subset (``hm_neuron.io.catalog``); returned tables are saved as CSV
* ``plot(results, cfg) -> list[(name, Figure)]``
* optionally ``DEFAULT_QUERY`` – catalog query applied when none is given
"""

from importlib import import_module

MODULES = (
    "field_count",
    "field_density_usage",
    "goal_coding",
    "stability",
    "theta",
    "remapping",
    "info_drop",
    "field_size_goal",
    "node_vs_edge",
)


def get(name):
    if name not in MODULES:
        raise KeyError(f"unknown analysis {name!r}; choose from {', '.join(MODULES)}")
    return import_module(f"{__name__}.{name}")


def all_analyses():
    return {m: get(m) for m in MODULES}
