"""Tracker config parsing (hm_tracker_paths.txt): bad channels, reference and
sleep-channel settings per rat.

Copied from HM_Tracker_2025 src/sorter/sorting.py: only the part above its
spikeinterface imports, so reading the config needs no sorting stack.
"""
import os
import shutil
import traceback
from datetime import datetime
from pathlib import Path
import numpy as np

# Built-in fallback used only when the config file has no entry for a rat.
# Keys are matched against the lowercased file stem prefix (e.g. "rat1").
DEFAULT_BAD_CHANNELS = {
    "rat1": [0, 1, 2, 3,
             4, 5, 6, 7,
             28, 29, 30, 31,
             36, 37, 38, 39,
             41, 42, 43,
             44, 45, 46, 47, 51,
             56, 59,
             68, 69, 70, 71,
             80, 92, 93, 94, 95,
             96, 97, 98, 99,
             100, 101, 102, 103,
             108, 109, 110, 111,
             124, 125, 126, 127],
}


import re

# Sorter selection. Stored in the config dict under a reserved key that can
# never collide with a rat token (rat keys are matched against file stems).
SORTER_CONFIG_KEY = "__sorter__"
DEFAULT_SORTER = "mountainsort5"
SUPPORTED_SORTERS = ("mountainsort5", "mountainsort4")


def resolve_sorter(config):
    """Return the configured sorter name, falling back to the default."""
    sorter = (config or {}).get(SORTER_CONFIG_KEY, DEFAULT_SORTER)
    if sorter not in SUPPORTED_SORTERS:
        print(f"Warning: unsupported sorter '{sorter}' in config; "
              f"using {DEFAULT_SORTER}. Supported: {', '.join(SUPPORTED_SORTERS)}.")
        sorter = DEFAULT_SORTER
    return sorter


# Global numeric sorting settings. Stored in the config dict under a reserved
# key. Maps the config-file key (upper case) -> (internal name, converter,
# default value).
SORTING_PARAMS_KEY = "__params__"
SORTING_PARAM_SPEC = {
    "FREQ_MIN": ("freq_min", float, 600.0),          # band-pass high-pass cutoff (Hz)
    "FREQ_MAX": ("freq_max", float, 8000.0),         # band-pass low-pass cutoff (Hz)
    "DETECT_THRESHOLD": ("detect_threshold", float, 5.0),  # spike detection threshold
    "DETECT_SIGN": ("detect_sign", int, 0),          # -1 neg, 0 both, 1 pos
}


def resolve_sorting_params(config):
    """Return the numeric sorting settings, filling in defaults for any
    keys not present in the config file."""
    stored = (config or {}).get(SORTING_PARAMS_KEY, {})
    return {name: stored.get(name, default)
            for (name, _conv, default) in SORTING_PARAM_SPEC.values()}


# 128 channels are wired as 32 tetrodes of 4: NT1..NT32, each with ch1..ch4.
# A token like "NT5ch3" maps to hardware channel (5-1)*4 + (3-1) = 18.
N_TETRODES = 32
CH_PER_TETRODE = 4
_NT_RE = re.compile(r"^nt(\d+)ch(\d+)$", re.IGNORECASE)


def nt_to_channel(tetrode, ch):
    """Map a 1-based NT tetrode / 1-based channel to a 0-based hardware id."""
    if not (1 <= tetrode <= N_TETRODES):
        raise ValueError(f"tetrode {tetrode} out of range 1..{N_TETRODES}")
    if not (1 <= ch <= CH_PER_TETRODE):
        raise ValueError(f"channel {ch} out of range 1..{CH_PER_TETRODE}")
    return (tetrode - 1) * CH_PER_TETRODE + (ch - 1)


def channel_to_nt(channel):
    """Inverse of nt_to_channel: 0-based hardware id -> (tetrode, ch), 1-based."""
    return channel // CH_PER_TETRODE + 1, channel % CH_PER_TETRODE + 1


def _parse_one_channel(token):
    """Parse a single token (plain int or NTxchY) into a 0-based channel id."""
    m = _NT_RE.match(token)
    if m:
        return nt_to_channel(int(m.group(1)), int(m.group(2)))
    return int(token)


def _parse_channel_list(raw_value):
    """
    Parse a space/comma separated list of channel ids into a list of ints.
    Each token may be a plain 0-based id (e.g. "18") or NT notation
    (e.g. "NT5ch3"). The two styles can be mixed.
    """
    if raw_value is None:
        return []
    tokens = raw_value.replace(",", " ").split()
    channels = []
    for tok in tokens:
        try:
            channels.append(_parse_one_channel(tok))
        except ValueError as e:
            print(f"Warning: ignoring invalid channel token '{tok}' in config ({e}).")
    return channels


# A whole-tetrode token: "NT5" or just "5" (1-based tetrode number).
_NT_TETRODE_RE = re.compile(r"^(?:nt)?(\d+)$", re.IGNORECASE)


def _parse_tetrode_token(token):
    """Parse a tetrode token ("NT5" or "5") into its four 0-based channel ids."""
    m = _NT_TETRODE_RE.match(token)
    if not m:
        raise ValueError(f"'{token}' is not a tetrode (expected NT<t> or <t>)")
    tetrode = int(m.group(1))
    if not (1 <= tetrode <= N_TETRODES):
        raise ValueError(f"tetrode {tetrode} out of range 1..{N_TETRODES}")
    base = (tetrode - 1) * CH_PER_TETRODE
    return [base + c for c in range(CH_PER_TETRODE)]


def _parse_tetrode_list(raw_value):
    """
    Parse a space/comma separated list of whole-tetrode tokens into the flat
    list of 0-based channel ids they cover. Each token is "NT<t>" or "<t>"
    (1-based tetrode number). Used for EEG tetrodes that are excluded from
    spike sorting entirely.
    """
    if raw_value is None:
        return []
    tokens = raw_value.replace(",", " ").split()
    channels = []
    for tok in tokens:
        try:
            channels.extend(_parse_tetrode_token(tok))
        except ValueError as e:
            print(f"Warning: ignoring invalid tetrode token '{tok}' in config ({e}).")
    return channels


# Sleep-scoring channels: one tetrode per role.
#   cortex -> NREM slow-wave/delta ; sr (stratum radiatum) -> REM theta ;
#   pyr (pyramidal) -> step 8's theta/delta ratio and step v's theta phase
#                      (sleep_role_column picks its channel for both).
SLEEP_ROLES = ("cortex", "sr", "pyr")


def _parse_sleep_channels(raw_value):
    """Parse ``cortex:NT28 sr:NT10 pyr:NT5`` (or plain tetrode numbers) into
    ``{'cortex': 28, 'sr': 10, 'pyr': 5}`` — 1-based tetrode numbers per role."""
    out = {}
    if not raw_value:
        return out
    for tok in raw_value.replace(",", " ").split():
        if ":" not in tok:
            print(f"Warning: ignoring sleep-channel token '{tok}' (want role:tetrode).")
            continue
        role, ch = tok.split(":", 1)
        role = role.strip().lower()
        m = _NT_TETRODE_RE.match(ch.strip())
        if role not in SLEEP_ROLES:
            print(f"Warning: unknown sleep-channel role '{role}' (use {SLEEP_ROLES}).")
        elif not m or not (1 <= int(m.group(1)) <= N_TETRODES):
            print(f"Warning: invalid tetrode '{ch}' for sleep role '{role}'.")
        else:
            out[role] = int(m.group(1))
    return out


def recording_belongs_to_rat(file_stem, rat_token):
    """True if a recording name starts with this rat's token.

    A plain startswith() is not enough: "rat10_hm_neurons_..." starts with
    "rat1" too, so Rat10..Rat19 would silently get Rat1's channel settings.
    The character right after the token must therefore not be a digit.

        recording_belongs_to_rat("Rat1_HM_20260101", "rat1")   -> True
        recording_belongs_to_rat("Rat10_HM_20260101", "rat1")  -> False
    """
    stem = file_stem.lower()
    if not stem.startswith(rat_token):
        return False
    next_char = stem[len(rat_token):len(rat_token) + 1]   # "" at the end of the name
    return not next_char.isdigit()


def resolve_sleep_channels(file_stem, config):
    """Return the ``{cortex, sr, pyr}`` sleep tetrodes for a recording (by rat), or {}."""
    for rat in sorted(config, key=len, reverse=True):
        if isinstance(config[rat], dict) and recording_belongs_to_rat(file_stem, rat):
            return dict(config[rat].get("sleep_channels", {}))
    return {}


def sleep_role_column(channel_map, sleep_channels, role, bad_channels=()):
    """The LFP column that stands for one sleep role's tetrode, or None.

    ``channel_map``    the LFP's columns, ``[{'index', 'ntrode', 'channel'}, ...]``
    ``sleep_channels`` ``{role: tetrode}`` from resolve_sleep_channels
    ``role``           "cortex", "sr" or "pyr"
    ``bad_channels``   hardware channel ids from BAD_CHANNELS_<RAT>

    The four channels of a tetrode carry nearly the same LFP, so one is enough:
    the lowest-numbered channel that is not bad. Step 8 (theta/delta ratio) and
    step v (theta phase) both use this, so they always pick the same channel.

    None when the role has no tetrode in the config, the tetrode is not in
    this LFP, or all of its channels are bad.
    """
    tetrode = (sleep_channels or {}).get(role)
    if tetrode is None:
        return None

    bad = set(int(c) for c in bad_channels)
    candidates = []                      # (channel number, LFP column)
    for position, entry in enumerate(channel_map or []):
        ntrode, channel = entry.get("ntrode"), entry.get("channel")
        if ntrode is None or channel is None or int(ntrode) != int(tetrode):
            continue
        hardware_id = (int(ntrode) - 1) * CH_PER_TETRODE + (int(channel) - 1)
        if hardware_id in bad:
            continue                     # a bad channel never stands for the tetrode
        candidates.append((int(channel), int(entry.get("index", position))))

    if not candidates:
        return None
    _channel, column = min(candidates)   # lowest channel number first
    return column


def load_sorting_config(config_path, quiet=False):
    """
    Read per-rat sorting settings from an hm_tracker_paths.txt style file.

    Recognised keys (RAT token is anything, e.g. RAT1, RAT2, MOUSEA):
        SORTER=mountainsort5             spike sorter for all rats
                                         (mountainsort5 or mountainsort4)
        FREQ_MIN=600                     band-pass high-pass cutoff (Hz)
        FREQ_MAX=8000                    band-pass low-pass cutoff (Hz)
        DETECT_THRESHOLD=5               spike detection threshold
        DETECT_SIGN=0                    -1 neg, 0 both, 1 pos
        BAD_CHANNELS_<RAT>=0 1 2 3 ...   channels to interpolate
        REF_CHANNEL_<RAT>=64             single (or space separated) reference
                                         channel(s) for common_reference.
                                         If omitted, global median is used.
        EEG_TETRODES_<RAT>=NT5 17 ...    whole tetrodes ("NT<t>" or "<t>") used
                                         for EEG; their channels are dropped
                                         from the recording before sorting.

    Returns a dict like:
        { "rat1": {"bad_channels": [...], "ref_channels": [...],
                   "eeg_channels": [...]}, ... }
    Rat tokens are lowercased so they can be matched against the file stem.
    """
    config = {}
    if not config_path:
        return config
    config_path = Path(config_path)
    if not config_path.exists():
        print(f"Sorting config not found at {config_path} — using built-in defaults.")
        return config

    def _entry(rat):
        return config.setdefault(
            rat, {"bad_channels": [], "ref_channels": [], "eeg_channels": [],
                  "sleep_channels": {}}
        )

    with open(config_path, "r", encoding="utf-8-sig") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip().upper()
            value = value.strip()
            if key == "SORTER":
                config[SORTER_CONFIG_KEY] = value.lower()
            elif key in SORTING_PARAM_SPEC:
                name, conv, default = SORTING_PARAM_SPEC[key]
                if value == "":
                    continue  # empty -> keep default
                try:
                    config.setdefault(SORTING_PARAMS_KEY, {})[name] = conv(value)
                except ValueError:
                    print(f"Warning: invalid value '{value}' for {key}; "
                          f"using default {default}.")
            elif key.startswith("BAD_CHANNELS_"):
                rat = key[len("BAD_CHANNELS_"):].lower()
                _entry(rat)["bad_channels"] = _parse_channel_list(value)
            elif key.startswith("REF_CHANNEL_"):
                rat = key[len("REF_CHANNEL_"):].lower()
                _entry(rat)["ref_channels"] = _parse_channel_list(value)
            elif key.startswith("EEG_TETRODES_"):
                rat = key[len("EEG_TETRODES_"):].lower()
                _entry(rat)["eeg_channels"] = _parse_tetrode_list(value)
            elif key.startswith("SLEEP_CHANNELS_"):
                rat = key[len("SLEEP_CHANNELS_"):].lower()
                _entry(rat)["sleep_channels"] = _parse_sleep_channels(value)

    # `quiet` for callers that only want the per-rat channel settings (the LFP
    # and EMG exports); announcing the sorter and its params there is noise,
    # since nothing is being sorted.
    rats = [k for k in config if not k.startswith("__")]
    if not quiet:
        if rats:
            print(f"Loaded sorting config for rats: {', '.join(sorted(rats))}")
        if SORTER_CONFIG_KEY in config:
            print(f"Sorter selected in config: {config[SORTER_CONFIG_KEY]}")
        if SORTING_PARAMS_KEY in config:
            print(f"Sorting params from config: {config[SORTING_PARAMS_KEY]}")
    return config


def resolve_rat_settings(file_stem, config):
    """
    Match a recording's file stem against the configured rats and return its
    bad channels, reference channels and EEG channels (channels belonging to
    EEG tetrodes, excluded from sorting). Falls back to DEFAULT_BAD_CHANNELS.
    """
    # Find the configured rat this recording belongs to. The reserved keys
    # ("__sorter__", "__params__") never match a recording name.
    matched = None
    for rat in sorted(config, key=len, reverse=True):
        if recording_belongs_to_rat(file_stem, rat):
            matched = rat
            break

    if matched is not None:
        settings = config[matched]
        bad_channels = list(settings.get("bad_channels", []))
        ref_channels = list(settings.get("ref_channels", []))
        eeg_channels = list(settings.get("eeg_channels", []))
        eeg_note = ""
        if eeg_channels:
            eeg_note = f", {len(eeg_channels)} EEG channel(s) excluded"
        print(f"Using config for '{matched}': "
              f"{len(bad_channels)} bad channel(s), "
              f"ref={ref_channels if ref_channels else 'global median'}"
              f"{eeg_note}")
        return bad_channels, ref_channels, eeg_channels

    # No config entry: fall back to built-in defaults (bad channels only).
    for rat, channels in DEFAULT_BAD_CHANNELS.items():
        if recording_belongs_to_rat(file_stem, rat):
            print(f"No config entry for '{rat}'; using built-in default bad channels.")
            return list(channels), [], []

    return [], [], []
