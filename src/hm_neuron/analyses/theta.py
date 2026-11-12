"""5. Theta strength and spike-phase modulation.

Per session: the theta reference channel (highest theta/delta ratio during
running), theta/delta ratio and peak theta frequency as a function of running
speed. Per unit (all ``quality`` units, pyramidal AND interneurons): theta
phase of spikes during running (speed >= threshold, inside trials), mean
resultant length (MRL), Rayleigh test (BH-FDR within session), preferred phase.
Phase 0 = theta peak of the reference channel.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .. import fields as fl, lfp as LF, plotting as P, stats as S
from . import common

NAME = "theta"
TITLE = "Theta power vs speed; spike-theta phase locking"
DEFAULT_QUERY = "day > 0"


def run(sessions, cfg):
    srows, urows, sp_rows = [], [], []
    for row, s, R in common.iter_loaded(sessions, cfg, with_fields=False):
        sid = common.session_id(row)
        L = LF.load_reference_lfp(s, cfg)
        phase, amp = LF.theta_phase_amp(L, cfg.theta.band_hz)
        tv = LF.theta_vs_speed(L, s, cfg)
        b = tv["bins"]
        cat = np.digitize(tv["speed"], b)
        for i in range(1, len(b)):
            m = cat == i
            if m.sum() >= 5:
                sp_rows.append(dict(session_id=sid, rat=row["rat"], day=row["day"],
                                    speed_lo=b[i - 1], speed_hi=b[i], n_win=int(m.sum()),
                                    theta_delta=float(np.median(tv["ratio"][m])),
                                    theta_freq=float(np.median(tv["freq"][m]))))
        run = tv["speed"] >= 0.1
        srows.append(dict(session_id=sid, rat=row["rat"], day=row["day"], channel=int(L["channel"]),
                          theta_delta_run=float(np.median(tv["ratio"][run])) if run.any() else np.nan,
                          theta_delta_still=float(np.median(tv["ratio"][tv["speed"] < 0.02])),
                          theta_freq_run=float(np.median(tv["freq"][run])) if run.any() else np.nan))
        runmask = fl.running_mask(s, cfg)
        for u in s.select_units(cfg, cell_type=False):
            st = s.spikes[u]
            st = st[runmask[fl.spike_samples(s, st)]]
            if len(st) < cfg.theta.min_spikes:
                continue
            ph = LF.at_times(L, phase, st)
            ph = ph[np.isfinite(ph)]
            z, p = S.rayleigh(ph)
            urows.append(dict(session_id=sid, rat=row["rat"], day=row["day"], unit=u,
                              cell_type=s.units.loc[u, "cell_type"], n_spikes=len(ph),
                              mrl=S.mrl(ph), pref_phase=S.circ_mean(ph), rayleigh_z=z, p=p,
                              # MRL is biased upwards for small n; ppc-like correction
                              mrl_unbiased=np.sqrt(max((len(ph) * S.mrl(ph) ** 2 - 1) / (len(ph) - 1), 0))))
    U = pd.DataFrame(urows)
    U["q"] = U.groupby("session_id")["p"].transform(S.fdr_bh)
    U["locked"] = U["q"] < cfg.fields.alpha
    summ = U.groupby(["rat", "cell_type"]).agg(n=("unit", "size"), frac_locked=("locked", "mean"),
                                               mrl_median=("mrl_unbiased", "median")).reset_index()
    return dict(sessions=pd.DataFrame(srows), speed=pd.DataFrame(sp_rows), units=U, summary=summ)


def plot(res, cfg):
    sp, U = res["speed"], res["units"]
    fig = P.figure(180, 55)[0]
    fig.clf()
    ax = [fig.add_subplot(1, 4, i + 1) for i in range(3)] + [fig.add_subplot(1, 4, 4, projection="polar")]
    for rat, g in sp.groupby("rat"):
        m = g.groupby("speed_hi")[["theta_delta", "theta_freq"]].median()
        ax[0].plot(m.index, m["theta_delta"], "o-", ms=3, color=P.RAT_COLORS.get(rat), label=rat)
        ax[1].plot(m.index, m["theta_freq"], "o-", ms=3, color=P.RAT_COLORS.get(rat))
    ax[0].set(xscale="log", xlabel="speed (m/s)", ylabel="theta / delta power")
    ax[0].legend(frameon=False)
    ax[1].set(xscale="log", xlabel="speed (m/s)", ylabel="theta peak frequency (Hz)")
    for ct, g in U.groupby("cell_type"):
        ax[2].hist(g["mrl_unbiased"], bins=30, histtype="step", density=True, label=ct)
    ax[2].set(xlabel="theta MRL (bias-corrected)", ylabel="density")
    ax[2].legend(frameon=False)
    for ct, g in U[U["locked"]].groupby("cell_type"):
        h, e = np.histogram(g["pref_phase"], bins=18, range=(-np.pi, np.pi))
        ax[3].plot(np.r_[e[:-1], e[0]] + np.pi / 18, np.r_[h, h[0]] / max(h.sum(), 1), label=ct)
    ax[3].set_title("preferred phase (locked units)")
    return [("theta", fig)]
