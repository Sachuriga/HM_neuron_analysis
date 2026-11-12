"""LFP access, theta reference-channel selection, phase and power.

The task NWB holds one LFP channel per tetrode (32 channels, 1500 Hz). The
theta reference is the channel with the highest theta/delta power ratio during
running, measured on a ``select_window_s`` window from the middle of the
session. Only that channel is then read in full, decimated to
``lfp_down_hz`` and cached.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
from scipy import signal


def _bandpower(x, fs, band):
    f, p = signal.welch(x, fs, nperseg=int(fs * 2), axis=0)
    m = (f >= band[0]) & (f <= band[1])
    return p[m].mean(0)


def load_reference_lfp(s, cfg, refresh=False):
    """Returns dict(lfp, fs, t0, channel, ratio_per_channel)."""
    th = cfg.theta
    cache = Path(cfg.cache_dir) / "lfp" / f"{s.label}_{int(th.lfp_down_hz)}hz.npz"
    if cache.exists() and not refresh:
        d = np.load(cache)
        return {k: (d[k].item() if d[k].ndim == 0 else d[k]) for k in d.files}
    with h5py.File(s.path, "r") as f:
        ds = f["acquisition/lfp/data"]
        fs = float(f["acquisition/lfp/starting_time"].attrs["rate"])
        t0 = float(f["acquisition/lfp/starting_time"][()])
        n = ds.shape[0]
        # selection window centred on the middle of the running time
        run_t = s.t[s.in_trial & (np.nan_to_num(s.speed) > cfg.fields.speed_min_m_s)]
        mid = np.median(run_t) if len(run_t) else (t0 + n / fs / 2)
        i0 = int(np.clip((mid - th.select_window_s / 2 - t0) * fs, 0, n - 1))
        i1 = int(min(n, i0 + th.select_window_s * fs))
        X = ds[i0:i1, :].astype(float)
        tt = t0 + np.arange(i0, i1) / fs
        sp = np.interp(tt, s.t, np.nan_to_num(s.speed))
        run = sp > cfg.fields.speed_min_m_s
        good = X.std(0) > 1e-6
        ratio = np.full(X.shape[1], np.nan)
        if run.sum() > fs * 10:
            Xr = X[run]
            ratio[good] = _bandpower(Xr[:, good], fs, th.band_hz) / _bandpower(Xr[:, good], fs, th.delta_hz)
        ch = int(np.nanargmax(ratio))
        x = ds[:, ch].astype(float)
    q = int(round(fs / th.lfp_down_hz))
    x = signal.decimate(x, q, ftype="fir", zero_phase=True)
    out = dict(lfp=x.astype(np.float32), fs=fs / q, t0=t0, channel=ch, ratio_per_channel=ratio)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, **out)
    return out


def bandpass(x, fs, band, order=3):
    sos = signal.butter(order, band, btype="band", fs=fs, output="sos")
    return signal.sosfiltfilt(sos, x)


def theta_phase_amp(lfp: dict, band):
    """Instantaneous theta phase (rad, 0 = peak) and amplitude."""
    z = signal.hilbert(bandpass(lfp["lfp"], lfp["fs"], band))
    return np.angle(z), np.abs(z)


def at_times(lfp: dict, values, times):
    i = np.round((np.asarray(times) - lfp["t0"]) * lfp["fs"]).astype(int)
    ok = (i >= 0) & (i < len(values))
    out = np.full(len(i), np.nan)
    out[ok] = values[i[ok]]
    return out


def theta_vs_speed(lfp: dict, s, cfg, speed_bins=(0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0), win_s=2.0):
    """Theta/delta ratio and peak theta frequency in windows, binned by running speed."""
    fs = lfp["fs"]
    x = lfp["lfp"]
    n = int(win_s * fs)
    starts = np.arange(0, len(x) - n, n)
    tc = lfp["t0"] + (starts + n / 2) / fs
    sp = np.interp(tc, s.t, np.nan_to_num(s.speed))
    intr = np.interp(tc, s.t, s.in_trial.astype(float)) > 0.5
    segs = np.stack([x[i:i + n] for i in starts[intr]])
    f, p = signal.welch(segs, fs, nperseg=n, axis=1)
    th = (f >= cfg.theta.band_hz[0]) & (f <= cfg.theta.band_hz[1])
    de = (f >= cfg.theta.delta_hz[0]) & (f <= cfg.theta.delta_hz[1])
    ratio = p[:, th].mean(1) / p[:, de].mean(1)
    fpk = f[th][p[:, th].argmax(1)]
    return dict(speed=sp[intr], ratio=ratio, freq=fpk, bins=np.asarray(speed_bins))
