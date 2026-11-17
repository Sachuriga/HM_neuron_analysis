# hm_neuron: HexMaze single-unit analysis

Place-field, goal-coding and theta analyses for the hippocampal recordings on
the HexMaze (`/Volumes/genzel/Rat/HM/Rat_HM_Neuron`, 2 rats × 20 task days).
The input is the session NWBs produced by HM_Tracker_2025. See
[`docs/data_description.md`](docs/data_description.md) for the data and
[`docs/methods.md`](docs/methods.md) for the definitions used.

## Install

```bash
conda activate HM_neuron          # Py ≥ 3.10 with h5py
pip install -e ".[dev]"
pytest                            # synthetic-data tests, < 5 s
```

## Quickstart

```bash
hm sessions                       # catalog: rat, day, goal, goal block, switch/recall days
hm analyses                       # list analyses
hm cache                          # read every NWB once into ~/hm_cache (slow, network volume)
hm run field_count                # one analysis, all training days (repeat > 0; repeat 0 = habituation)
hm run all                        # everything -> results/<analysis>/
hm run theta --rat Rat5 --date 20260709          # one session
hm run goal_coding --query "block == 2"          # any pandas query on the catalog
hm --config my.yaml --out results_v2 run all     # override parameters
```

Each analysis writes `results/<name>/*.csv` (tidy tables), `<name>.pdf/.png`,
and `run_info.json` (sessions, the full config used, and a code fingerprint).

On Windows the data root is the SMB share: `hm --config config_windows.yaml ...`.

## LaTeX report

```bash
hm report field_examples field_count field_density_usage   # chosen analyses (default: all)
hm report --watch field_count                              # keep running, rebuild on every source change
hm report --force                                          # re-run everything
```

`hm report` writes `report/main.tex`, one `report/sections/<analysis>.tex` per analysis
(text, tables and figure captions generated from the result CSVs) and copies the figures to
`report/figures/`. An analysis is re-run only when its fingerprint changed: a hash of its own
module, the core modules (`hm_neuron/*.py`, `analyses/common.py`, `data/`), the config and
the selected sessions. Otherwise the existing results are reused, so a rebuild after an
unrelated edit takes seconds. The PDF is compiled when `latexmk`, `tectonic` or `pdflatex`
is on PATH or in the conda env (`conda install -c conda-forge tectonic`); without one,
`report/` can be uploaded to Overleaf as is.

To put text or tables into an analysis' section, define `latex(tables) -> str` and
`CAPTIONS = {figure_name: caption}` in its module (see `analyses/field_count.py`).

From Python:

```python
from hm_neuron.config import Config
from hm_neuron import io, fields
cfg = Config()
s = io.load_session(cfg, "Rat5", "20260709")      # position, spikes, trials, units
R = fields.session_fields(s, cfg)                  # rate maps, SI + shuffle p, fields
R["units"], R["fields"]
```

See `notebooks/quickstart.ipynb`.

## Analyses

| `hm run …` | Question |
|---|---|
| `field_examples` | How is a field detected? Example cells step by step: spikes on path, rate map, thresholded bins grouped into fields, and the spike-shift surrogate |
| `field_count` | How many fields does a cell have? Fits ZT-Poisson, ZT-negative binomial, geometric, discretised log-normal and Gaussian, with AIC/BIC and chi-square goodness of fit per rat; compared with spike-shift surrogates |
| `field_density_usage` | Does field density per node/edge scale with how often it is visited or occupied? Poisson GLM, session-clustered, observed vs spike-shift surrogates |
| `goal_coding` | Goal-zone enrichment (current and previous goal) over days and within-day thirds, goal cells vs null, and field recruitment lag vs goal distance |
| `stability` | Visit-to-visit reliability, rate CV, and within-edge profile correlation (same vs opposite direction, vs shifted null, drift with time) |
| `theta` | Theta/delta power and frequency vs speed; spike-phase MRL, Rayleigh test (FDR) and preferred phase per unit |
| `remapping` | Population map correlation across consecutive days: same goal vs goal switch, occupancy-controlled; within-day split-half baseline |
| `info_drop` | SI drop on switch days, and whether it reflects poor selectivity (low gain over a flat model) or coexisting early/late maps (bimodal per-spike LLR) |
| `field_size_goal` | Field size vs track distance to goal, mixed model controlling for peak rate and node location |
| `node_vs_edge` | More fields on nodes or on edges? Observed vs expected by track length and by occupancy |
| `cell_classes` | How many goal, start, long-/short-bridge and outside-maze cells per session and across learning? Exact circular-shift test per unit, counts vs matched control locations |

## Design notes

* **Linearised maze.** Positions are projected onto the maze graph. Positions within
  `lin.node_radius_m` (8 cm) of a node fall in that node's zone; the rest fall in 4 cm bins
  along edges. Smoothing and all distances use track distance, so firing never
  leaks between neighbouring corridors.
* **Units.** By default only `good` pyramidal units are used. Theta uses good units of both cell types.
  Place cell = SI above the shuffle null (p < 0.05, 500 circular shifts) and ≥ 1 field.
* **Statistics.** With only two rats, results are given per rat and pooled with
  session as the random/cluster effect. Units are never treated as independent across sessions.
* **No cross-day cell tracking.** Each day is sorted separately, so cross-goal analyses
  are population-level. Within-day analyses (recruitment, stability, info_drop
  mechanism) follow single cells.
* **Caching.** Sessions are cached in `~/hm_cache/sessions` and fields in `~/hm_cache/fields`.
  The field cache is keyed by a hash of the field, linearisation and unit parameters, so changing
  a parameter recomputes automatically. Changing `fields.speed_smooth_s` needs
  `rm -r ~/hm_cache/sessions`.

## Tracker unit PDFs (`hm_neuron.tracker_viz`)

A copy of the Genzel tracker's step `v` (HM_Tracker_2025, `src/nwb/visualize_nwb.py`
and the modules it needs). For one op folder it writes `visualization/summary.pdf`
and one `Unit_<id>.pdf` per good unit: 2D rate maps, place-field outlines, event
rasters / PETHs (reward, start, bridge, ...), theta phase and precession.

```
pip install -e ".[viz]"
hm-visualize --output_folder <op_folder>
```

Place fields here are connected regions of the 2D rate map (5 cm bins, 2-bin
Gaussian smoothing, speed > 0.05 m/s) at >= 50% of the cell's peak, with >= 6
bins and an in-field peak >= 1 Hz (`MIN_FIELD_PEAK_HZ`; the tracker uses 0.5 Hz).
There is no shuffle test. This is separate from the linearised-maze fields used
by the `hm run` analyses (`hm_neuron.fields`).
