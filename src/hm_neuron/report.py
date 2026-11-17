"""LaTeX report that rebuilds itself when the code changes.

``hm report [analysis ...]`` does, for each analysis:

1. Fingerprint = hash of the analysis module, every core module it can use
   (``hm_neuron/*.py``, ``analyses/common.py``, ``data/*``), the config and the
   selected sessions.
2. If ``results/<name>/run_info.json`` holds a different fingerprint (or none),
   the analysis is re-run. Otherwise its CSVs and figures are reused.
3. ``report/sections/<name>.tex`` is regenerated from the CSVs, figures are
   copied to ``report/figures/`` and ``report/main.tex`` is rewritten.
4. The PDF is compiled when a LaTeX engine (latexmk, pdflatex or tectonic) is on PATH.

``hm report --watch`` polls the source tree and repeats this after every change.

An analysis can add text and tables to its section by defining
``latex(tables: dict[str, DataFrame]) -> str`` and figure captions with
``CAPTIONS = {figure_name: latex_caption}``. Analyses without them get only their figures.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import analyses, runner

PKG = Path(__file__).resolve().parent
# files whose change does not affect analysis results
_IGNORE = {"cli.py", "report.py"}


# ----------------------------------------------------------------------------- fingerprint
def _source_files(name: str) -> list[Path]:
    files = [p for p in PKG.glob("*.py") if p.name not in _IGNORE]
    files += [PKG / "analyses" / "common.py", PKG / "analyses" / f"{name}.py"]
    files += sorted((PKG / "data").glob("*"))
    return sorted(set(files))


def fingerprint(name: str, cfg, sessions: pd.DataFrame) -> str:
    h = hashlib.md5()
    for p in _source_files(name):
        h.update(p.name.encode())
        h.update(p.read_bytes().replace(b"\r\n", b"\n"))
    d = cfg.to_dict()
    d.pop("out_dir", None)
    h.update(json.dumps(d, sort_keys=True, default=str).encode())
    h.update(json.dumps(sessions[["rat", "date"]].astype(str).values.tolist()).encode())
    return h.hexdigest()[:12]


def _stored_fingerprint(out: Path):
    try:
        return json.loads((out / "run_info.json").read_text()).get("code_hash")
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def ensure_results(name, cfg, force=False):
    """Re-run `name` if its fingerprint changed; return (tables, run_info, rerun?)."""
    mod = analyses.get(name)
    sessions = runner.select(cfg, default_query=getattr(mod, "DEFAULT_QUERY", None))
    fp = fingerprint(name, cfg, sessions)
    out = Path(cfg.out_dir) / name
    rerun = force or _stored_fingerprint(out) != fp
    if rerun:
        print(f"[report] {name}: code/config/sessions changed -> re-running", flush=True)
        runner.run_analysis(name, cfg)
    else:
        print(f"[report] {name}: up to date ({fp})", flush=True)
    tables = {p.stem: pd.read_csv(p) for p in sorted(out.glob("*.csv"))}
    return tables, json.loads((out / "run_info.json").read_text()), rerun


# ----------------------------------------------------------------------------- LaTeX helpers
_TEX_ESC = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
            "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\^{}",
            "ρ": r"$\rho$", "≥": r"$\geq$", "Δ": r"$\Delta$", "→": r"$\to$", "—": "---", "–": "--"}


def esc(s) -> str:
    return "".join(_TEX_ESC.get(c, c) for c in str(s))


def fmt(v, digits=3) -> str:
    if isinstance(v, (bool, np.bool_)):
        return "yes" if v else "no"
    if isinstance(v, (int, np.integer)):
        return f"{int(v):,}".replace(",", "{,}")
    if isinstance(v, (float, np.floating)):
        if not np.isfinite(v):
            return "--"
        if v == 0:
            return "0"
        a = abs(v)
        if a < 1e-3 or a >= 1e5:
            m, e = f"{v:.2e}".split("e")
            return rf"${m}\times10^{{{int(e)}}}$"
        if float(v).is_integer() and a >= 100:
            return fmt(int(v))
        return f"{v:.{digits}g}" if a < 1 else f"{v:.{max(digits, len(str(int(a))))}g}"
    return esc(v)


def table(df: pd.DataFrame, caption: str, label: str, cols=None, headers=None,
          digits=3, size=r"\small") -> str:
    """booktabs table without the jinja2 dependency of DataFrame.to_latex."""
    cols = list(cols or df.columns)
    headers = headers or {}
    align = "".join("r" if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])
                    else "l" for c in cols)
    lines = [r"\begin{table}[htbp]", r"\centering", size,
             rf"\caption{{{caption}}}", rf"\label{{{label}}}",
             rf"\begin{{tabular}}{{{align}}}", r"\toprule",
             " & ".join(headers.get(c, esc(c)) for c in cols) + r" \\", r"\midrule"]
    for _, r in df[cols].iterrows():
        lines.append(" & ".join(fmt(r[c], digits) for c in cols) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def figure(path: str, caption: str, label: str, width=r"\textwidth") -> str:
    return "\n".join([r"\begin{figure}[htbp]", r"\centering",
                      rf"\includegraphics[width={width}]{{{path}}}",
                      rf"\caption{{{caption}}}", rf"\label{{{label}}}", r"\end{figure}", ""])


# ----------------------------------------------------------------------------- build
_PREAMBLE = r"""\documentclass[11pt,a4paper]{article}
\usepackage[margin=2cm]{geometry}
\usepackage{graphicx,booktabs,amsmath,hyperref}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\graphicspath{{figures/}}
\title{HexMaze place-field analyses}
\author{hm\_neuron (auto-generated)}
\date{%(date)s}
\begin{document}
\maketitle
\noindent Generated by \texttt{hm report} from commit \texttt{%(commit)s}%(dirty)s.
Every section is rebuilt when its code, configuration or session set changes;
the fingerprint of each analysis is given in its section. Habituation
(Repeat~0, Day~0) is excluded throughout.

"""


def _git(*args):
    try:
        return subprocess.run(["git", *args], cwd=PKG, capture_output=True, text=True,
                              timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def _section(name, tables, info, fig_dir: Path, res_dir: Path) -> str:
    mod = analyses.get(name)
    n_ses = len(info.get("sessions", []))
    parts = [rf"\section{{{esc(mod.TITLE)}}}", rf"\label{{sec:{name}}}",
             rf"\noindent\textit{{Analysis \texttt{{{esc(name)}}}, {n_ses} sessions, "
             rf"fingerprint \texttt{{{info.get('code_hash', '?')}}}.}}", ""]
    if hasattr(mod, "latex"):
        parts.append(mod.latex(tables))
    captions = getattr(mod, "CAPTIONS", {})
    for f in sorted(res_dir.glob("*.pdf")):
        dst = fig_dir / f"{name}__{f.name}"
        shutil.copy2(f, dst)
        parts.append(figure(dst.name, captions.get(f.stem, esc(mod.TITLE)), f"fig:{f.stem}"))
    parts.append(r"\clearpage")
    return "\n".join(parts)


_ENV_BIN = os.pathsep.join(str(Path(sys.prefix) / d) for d in ("Library/bin", "Scripts", "bin"))


def compile_pdf(tex_dir: Path):
    for cmd in (["latexmk", "-pdf", "-interaction=nonstopmode", "-quiet", "main.tex"],
                ["tectonic", "main.tex"],
                ["pdflatex", "-interaction=nonstopmode", "main.tex"]):
        exe = shutil.which(cmd[0]) or shutil.which(cmd[0], path=_ENV_BIN)   # also the conda env's own bin
        if exe:
            cmd = [exe] + cmd[1:]
            runs = 2 if "pdflatex" in Path(exe).name else 1
            for _ in range(runs):
                r = subprocess.run(cmd, cwd=tex_dir, capture_output=True, text=True)
            if r.returncode == 0:
                print(f"[report] compiled -> {tex_dir / 'main.pdf'}")
                return tex_dir / "main.pdf"
            print(f"[report] {cmd[0]} failed:\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")
            return None
    print("[report] no LaTeX engine on PATH (latexmk / tectonic / pdflatex); "
          f"wrote {tex_dir / 'main.tex'} only")
    return None


def build(cfg, names=None, tex_dir="report", force=False, compile=True):
    names = list(names or analyses.MODULES)
    tex_dir = Path(tex_dir)
    (tex_dir / "sections").mkdir(parents=True, exist_ok=True)
    (tex_dir / "figures").mkdir(parents=True, exist_ok=True)
    built = []
    for n in names:
        try:
            tables, info, _ = ensure_results(n, cfg, force=force)
        except Exception as e:  # keep the rest of the report
            import traceback
            traceback.print_exc()
            print(f"[report] {n}: FAILED ({e}); keeping previous section if any")
            if (tex_dir / "sections" / f"{n}.tex").exists():
                built.append(n)
            continue
        (tex_dir / "sections" / f"{n}.tex").write_text(
            _section(n, tables, info, tex_dir / "figures", Path(cfg.out_dir) / n), encoding="utf-8")
        built.append(n)
    dirty = _git("status", "--porcelain", "--", str(PKG))
    head = _PREAMBLE % dict(date=dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
                            commit=esc(_git("rev-parse", "--short", "HEAD") or "?"),
                            dirty=" (with uncommitted changes)" if dirty else "")
    body = "\n".join(rf"\input{{sections/{n}}}" for n in built)
    (tex_dir / "main.tex").write_text(head + body + "\n\\end{document}\n", encoding="utf-8")
    print(f"[report] wrote {tex_dir / 'main.tex'} ({len(built)} sections)")
    return compile_pdf(tex_dir) if compile else None


def _tree_state():
    return {p: p.stat().st_mtime for p in PKG.rglob("*") if p.suffix in (".py", ".csv", ".yaml")
            and "__pycache__" not in p.parts}


def watch(cli_args, interval=5.0):
    """Rebuild whenever a source file changes (Ctrl-C to stop).

    Each rebuild runs ``hm <cli_args>`` in a fresh interpreter so edited modules are re-imported.
    """
    import sys
    state = None
    print(f"[report] watching {PKG} every {interval:.0f}s (Ctrl-C to stop)")
    while True:
        new = _tree_state()
        if new != state:
            if state is not None:
                changed = sorted(p.name for p in set(new) | set(state) if new.get(p) != state.get(p))
                print(f"[report] changed: {', '.join(changed)}", flush=True)
            state = new
            subprocess.run([sys.executable, "-m", "hm_neuron.cli", *cli_args])
        time.sleep(interval)
