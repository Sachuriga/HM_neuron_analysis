"""Command line: ``hm <command>``.

  hm sessions                         list sessions in the catalog
  hm analyses                         list analyses
  hm cache [--rat R] [--date D]       pre-load sessions + fields into the local cache
  hm run <analysis|all> [--rat R ...] [--date D ...] [--query "day > 5"] [--no-plot]

Common options: ``--config my.yaml``, ``--out results_dir``.
"""

from __future__ import annotations

import argparse
import sys
import traceback

from . import analyses, fields, io, runner
from .config import Config


def _cfg(a):
    cfg = Config.from_yaml(a.config)
    if a.out:
        from pathlib import Path
        cfg.out_dir = Path(a.out)
    return cfg


def main(argv=None):
    p = argparse.ArgumentParser(prog="hm", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config")
    p.add_argument("--out")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sessions")
    s.add_argument("--refresh", action="store_true")
    sub.add_parser("analyses")
    for name in ("cache", "run"):
        q = sub.add_parser(name)
        if name == "run":
            q.add_argument("analysis")
            q.add_argument("--no-plot", action="store_true")
        q.add_argument("--rat", nargs="*")
        q.add_argument("--date", nargs="*")
        q.add_argument("--query")
    a = p.parse_args(argv)
    cfg = _cfg(a)

    if a.cmd == "sessions":
        cat = io.catalog(cfg, refresh=a.refresh)
        cols = ["rat", "date", "day", "repeat", "session", "goal", "block", "day_in_block",
                "is_switch_day", "is_recall_day"]
        print(cat[cols].to_string(index=False))
    elif a.cmd == "analyses":
        for k, m in analyses.all_analyses().items():
            print(f"{k:22s} {m.TITLE}")
    elif a.cmd == "cache":
        sel = runner.select(cfg, a.rat, a.date, a.query)
        for r in sel.itertuples():
            print(r.rat, r.date, flush=True)
            fields.session_fields(io.load_session(cfg, r.rat, str(r.date)), cfg)
    elif a.cmd == "run":
        names = analyses.MODULES if a.analysis == "all" else [a.analysis]
        failed = []
        for n in names:
            try:
                runner.run_analysis(n, cfg, a.rat, a.date, a.query, make_plots=not a.no_plot)
            except Exception:
                traceback.print_exc()
                failed.append(n)
        if failed:
            print("FAILED:", ", ".join(failed), file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
