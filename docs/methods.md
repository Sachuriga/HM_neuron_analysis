# Methods (definitions used in `hm_neuron`)

All parameters are in `src/hm_neuron/config.py`; defaults are given in brackets.

**Position.** Tracker pixels are converted to metres (2352/2/9 and 1424/2/5 px/m).
NaN gaps ≤ 0.5 s are linearly interpolated. Speed comes from Gaussian-smoothed
positions (σ 0.25 s).

**Linearisation.** Maze graph: 96 nodes, 125 edges (node table and adjacency rules
from HM_Rat_Analysis). Each sample is projected onto its nearest edge (dropped if
> 0.15 m away). It is assigned to the node zone when within 0.08 m of an end, otherwise
to one of the equal bins (~4 cm) along the edge, giving 921 bins in total. The
track distance between bins is the shortest path through the bin graph.

**Rate maps.** Only in-trial samples with speed ≥ 0.05 m/s are used. Spike counts and
occupancy are smoothed separately with a Gaussian over track distance (σ 5 cm,
truncated at 3σ) and then divided. A bin is valid when smoothed occupancy ≥ 0.3 s.

**Spatial information.** Skaggs bits/spike over valid bins. Significance comes from 500
circular shifts of the spike train within concatenated running time (shift ≥ 20 s);
the place-cell criterion is p < 0.05 plus at least one field.

**Fields.** A field is a connected set (on the bin graph) of valid bins with rate
≥ 30% of the map peak, peak ≥ 1 Hz, and length ≥ 0.10 m. Its location is the peak
bin, and its size is the summed bin length. `node_frac` is the fraction of its length
in node zones. Goal distance is the track distance from the peak to the goal node.

**Visits.** A visit is a maximal run of in-trial samples on one element (node or edge),
at least 0.2 s of running. An edge visit's direction is the sign of net movement along
the edge. A cell counts as active on a visit when its running rate exceeds 0.5 Hz.

**Theta.** The reference channel is the tetrode LFP with the largest theta (6–10 Hz)
to delta (2–4 Hz) power ratio during running in a 10-min mid-session window. It is
decimated to 250 Hz and band-passed (Butterworth order 3, zero-phase); the Hilbert
phase gives 0 at the theta peak. Phase locking is measured as MRL, with a bias-corrected MRL
√((n·R² − 1)/(n − 1)), and tested with the Rayleigh test, BH-FDR within session.

**Goal zone.** The goal zone is every bin within 0.5 m of track distance from the goal node.
Enrichment = (fraction of field peaks in the zone) / (fraction of valid track length
in the zone).

**Cell classes** (`cell_classes`). Every good unit with ≥ 100 spikes is tested for a firing
preference for each state; classes are not exclusive. Spatial classes use in-trial running
samples and compare the zone with the rest of the maze: *goal* (≤ 0.5 m track distance from
the goal node), *start* (≤ 0.5 m from the start node of the current trial), *long bridge*
(inter-island corridors > 1 m: 121–302, 223–404) and *short bridge* (124–201, 220–305,
324–401). Temporal classes compare with all in-trial time: *outside maze* (rat carried from
the goal to the next start, plus before the first and after the last trial) and *at goal after
arrival* (goal arrival until pick-up). The rat's track is frozen between trials, so the
pick-up is the last moment the tracked researcher was within 1.5 m of where the rat ended the
trial. A unit is in a class when its rate in the state is ≥ 0.5 Hz and exceeds the reference
with p < 0.05 against every circular shift (≥ 20 s) of its spike train relative to behaviour
(exact, computed by FFT cross-correlation). Goal and bridge counts are compared with control
locations of similar running occupancy (0.5–2×): zones around other nodes, and random sets of
within-island corridors of the same total length. All states are restricted to the span of
the spike record.

**Day types.** A switch day is the first day with a new goal (days 1, 6, 11, 16). A recall day
is Session 1 of a Repeat, the last day with the old goal (days 5, 10, 15). Day 0 is a
single free-exploration trial with goal 101 and is excluded by default.

**Statistics.** There are 2 rats, so every test is given per rat. Pooled models use
session as the random intercept (`statsmodels.mixedlm`) or as the cluster for robust SEs
(Poisson GLM), with rat as a fixed effect. Multiple tests across units are BH-FDR corrected.
