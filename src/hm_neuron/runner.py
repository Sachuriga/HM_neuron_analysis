"""Run an analysis end-to-end: select sessions, compute, save tables + figures."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd

from . import analyses, io, plotting


def select(cfg, rats=None, dates=None, query=None, default_query=None):
    cat = io.catalog(cfg)
    rats = rats or list(cfg.rats)
    cat = cat[cat["rat"].isin(rats)]
    if dates:
        cat = cat[cat["date"].astype(str).isin([str(d) for d in dates])]
    q = query if query is not None else default_query
    if q:
        cat = cat.query(q)
    return cat.reset_index(drop=True)


def run_analysis(name, cfg, rats=None, dates=None, query=None, make_plots=True):
    mod = analyses.get(name)
    sessions = select(cfg, rats, dates, query, getattr(mod, "DEFAULT_QUERY", None))
    if sessions.empty:
        raise ValueError("no sessions selected")
    print(f"[{name}] {mod.TITLE} — {len(sessions)} sessions", flush=True)
    t0 = time.time()
    res = mod.run(sessions, cfg)
    out = Path(cfg.out_dir) / name
    out.mkdir(parents=True, exist_ok=True)
    for k, df in res.items():
        if isinstance(df, pd.DataFrame):
            df.to_csv(out / f"{k}.csv", index=False)
    (out / "run_info.json").write_text(json.dumps(dict(
        analysis=name, sessions=sessions[["rat", "date"]].astype(str).values.tolist(),
        query=query, config=cfg.to_dict(), runtime_s=round(time.time() - t0, 1)),
        indent=1, default=str))
    if make_plots:
        for fname, fig in mod.plot(res, cfg):
            plotting.save(fig, out, fname, cfg)
    print(f"[{name}] done in {time.time() - t0:.0f}s -> {out}", flush=True)
    return res
