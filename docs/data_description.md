# HM neuron data: description (Phase 1 survey, 2026-11-12)

Root: `/Volumes/genzel/Rat/HM/Rat_HM_Neuron` (read-only network volume)

## Layout
```
task/<rat>_<id>/<YYYYMMDD>/   20 sessions per rat; Rat5 (491390), Rat6 (491391)
    <Rat>_<date>.nwb                     main source, about 1-5 GB including LFP
    *_mountainsort4_sorting_output/      sorting_analyzer + phy_export
    *_Coordinates_Full*.csv, *_framewise_ts.csv, stitched_framewise_seconds.csv
    log_*.log, <date>_<Rat>.txt (node sequences), RecordingMeta.xlsx
    LFP_Output/, decoding/, predictive_coding/, umap/, visualization/  (old derived outputs)
pre/Rat{5,6}/<date>/   pre-task sleep sorting output (2-4 days per rat)
post/Rat{5,6}/<date>/  post-task sleep: <Rat>_<date>_post.nwb, *.eegstates.npz
task/session_summary*.xlsx|pdf   old cross-session summary (HM_Rat_Analysis)
```

## Task NWB contents (example Rat5 20260709)
| Path | Contents |
|---|---|
| `session_description` | `Rat5 ..., Repeat 3, Day 13, Session 4, Goal_Node 107` |
| `units/` | 393 units; `spike_times` (ragged), `quality_label` (good/mua/noise), `cell_type` (pyramidal/interneuron), waveform metrics, `waveform_mean` (90×124: samples × channels?), `phy_cluster_id` |
| `processing/Behavior/Position/Rat` | (N,2) xy at about 30 Hz, timestamps start negative (before ephys t0). Also tracked: `JP_L`, `JP_S`, `Researcher` |
| `processing/Behavior/Metrics/region_id` | per-frame maze region (node/edge id?) |
| `processing/Behavior/Trials_Data` | **one row per node-to-node step** (661 rows). Columns: `Trial_Num`, `Start_node`, `Next_node`, `Time_start/end`, `Trial_start/end_s`, `Goal_node`, `Velocity_m_s`, `Length_m` |
| `acquisition/lfp` | (5.76M, 32) float32 µV at 1500 Hz (about 64 min). Channel map in `processing/sleep/session_info` |
| `scratch/trial_unit_metrics_*` | precomputed per-trial × unit metrics from the old pipeline |

## Session / goal schedule
Exactly one goal per NWB session. Goal changes are between Session 1 and Session 2 of each Repeat:

| Rat | Day 0 | Repeat 1 (D1-4) + R2 S1 (D5) | R2 S2-5 (D6-9) + R3 S1 (D10) | R3 S2-5 (D11-14) + R4 S1 (D15) | R4 S2-5 (D16-19) |
|---|---|---|---|---|---|
| Rat5 | 101 | 410 | 314 | 107 | 219 |
| Rat6 | 101 | 204 | 109 | 318 | 421 |

Units per session: 30-137 good plus 6-126 MUA. 9-33 trials per session (Day 0 has only 1). LFP is present in all 40 sessions.

## Existing code (`HM_Rat_Analysis`) worth porting
- `maze.py`: node table (`data/node_list_new.csv`), networkx graph, islands, bridges, idealised positions, warp, shortest paths
- `nwb.py`, `behaviour.py`: position loading, trial building and alignment, node-sequence parsing, speed
- `place_fields.py` (1.2k lines): 2-D rate maps (2.5 cm bins, 5 cm smoothing, 0.3 s minimum occupancy), Skaggs information, fields
- `spike_metrics.py`: ACG tau-rise, cell-type classification
- Env: conda `HM_neuron` (Py 3.12, pynwb 4.0) has the required stack; base env lacks h5py.

## Ambiguities / open issues
1. **No cross-day unit tracking visible.** Each day is sorted separately, so "remapping across goals", "cell recruitment" and multi-session temporal evolution cannot follow single cells across days unless matching exists. Within a session there is only one goal.
2. Session 1 of each Repeat keeps the previous goal. Is it a probe or recall session with the old goal, and does the new goal start within Session 2 or from its first trial?
3. Rows in `Trials_Data`: confirm `Trial_Num` is the start-to-goal trial and that each row is one edge traversal.
4. `region_id` coding (node vs. edge ids) and the definition of node vs. edge ("hedge") zones in cm.
5. LFP: which channel(s) to use for theta (pyramidal layer?), and whether all 32 channels are in hippocampus.
6. Time base: position timestamps are negative before ephys start. Confirm everything in the NWB shares the ephys clock.
7. Unit inclusion: `good` only, or `good`+`mua`? Use `cell_type == pyramidal` for field analyses?
