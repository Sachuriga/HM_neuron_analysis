"""Per-session unit visualisation copied from the Genzel tracker (HM_Tracker_2025).

``visualize_nwb`` is runner step [v] of the tracker: it writes ``summary.pdf`` and
one ``Unit_<id>.pdf`` per good unit into ``<op>/visualization/``. The per-unit
PDFs include the event rasters / PETHs (reward, start, bridge, ...) and the
theta phase and precession pages from ``theta_events``. It works on a 2D rate
map of an op folder and is independent of the linearised-maze field code in
:mod:`hm_neuron.fields` (different thresholds, no shuffle test).

Copied from HM_Tracker_2025 commit 8a2d5af (2026-11-13):

  src/nwb/visualize_nwb.py      summary + per-unit PDFs (entry point)
  src/nwb/theta_events.py       event rasters / PETHs, theta phase, precession
  src/nwb/neural_umap.py        maze helpers used by theta_events (_hex_assign, _bridges)
  src/nwb/sleep_nwb_writer.py   NWB names and the theta/delta ratio
  src/nwb/spike_metrics.py      waveform / ACG metrics and cell-type rule
  src/nwb/session_prefix.py     rat_sessiondate_ output prefix
  src/sorter/sorting.py         -> sorting_config.py, config parsing part only

Only imports, the node-list path and the entry point were changed.

Run with ``hm-visualize --output_folder <op>`` or
``python -m hm_neuron.tracker_viz.visualize_nwb --output_folder <op>``.
Needs the ``viz`` extra: ``pip install -e ".[viz]"``.
"""
