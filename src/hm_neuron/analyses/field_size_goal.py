"""8. Is place-field size a function of distance from the goal?

Field size (track length, m) vs track distance from the field PEAK to the goal
node. Pooled with a mixed model (random intercept per session); covariates
control for peak rate and for whether the peak sits on a node. Also per rat
(Spearman) and as binned medians.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .. import plotting as P, stats as S
from . import common

NAME = "field_size_goal"
TITLE = "Field size vs track distance to goal"
DEFAULT_QUERY = "repeat > 0"  # repeat 0 = habituation, not training


def run(sessions, cfg):
    U, F = common.gather(sessions, cfg)
    f = F[F["is_place"]].copy()
    f["log_size"] = np.log(f["size_m"])
    f["log_peak"] = np.log(f["peak_hz"])
    f["on_node"] = (f["peak_kind"] == "node").astype(int)
    models = pd.concat([
        S.mixed("log_size ~ goal_dist_m + C(rat)", f).assign(model="simple"),
        S.mixed("log_size ~ goal_dist_m + log_peak + on_node + C(rat)", f).assign(model="controlled"),
    ], ignore_index=True)
    per_rat = S.per_group(f, "rat", lambda g: S.spearman(g["goal_dist_m"].to_numpy(), g["size_m"].to_numpy()))
    per_ses = S.per_group(f, ["rat", "session_id", "day"],
                          lambda g: S.spearman(g["goal_dist_m"].to_numpy(), g["size_m"].to_numpy()))
    f["dist_bin"] = pd.cut(f["goal_dist_m"], np.arange(0, f["goal_dist_m"].max() + 1, 1.0))
    binned = (f.groupby(["rat", "dist_bin"], observed=True)["size_m"]
              .agg(["median", "count", lambda x: x.quantile(.25), lambda x: x.quantile(.75)])
              .reset_index())
    binned.columns = ["rat", "dist_bin", "median", "n", "q25", "q75"]
    binned["dist_mid"] = binned["dist_bin"].apply(lambda b: b.mid).astype(float)
    binned["dist_bin"] = binned["dist_bin"].astype(str)
    return dict(fields=f, models=models, per_rat=per_rat, per_session=per_ses, binned=binned)


def plot(res, cfg):
    f, b = res["fields"], res["binned"]
    fig, ax = P.figure(120, 55, ncols=2)
    for rat, g in f.groupby("rat"):
        ax[0].scatter(g["goal_dist_m"], g["size_m"], s=2, alpha=0.3, color=P.RAT_COLORS.get(rat))
    for rat, g in b.groupby("rat"):
        ax[0].plot(g["dist_mid"], g["median"], "-", color=P.RAT_COLORS.get(rat), lw=1.5, label=rat)
    ax[0].set(yscale="log", xlabel="peak distance to goal (m, track)", ylabel="field size (m)")
    ax[0].legend(frameon=False)
    ps = res["per_session"]
    for rat, g in ps.groupby("rat"):
        ax[1].plot(g["day"], g["rho"], "o-", ms=3, color=P.RAT_COLORS.get(rat))
    ax[1].axhline(0, color="k", lw=0.6)
    ax[1].set(xlabel="day", ylabel="Spearman ρ (size vs goal distance)")
    return [("field_size_goal", fig)]
