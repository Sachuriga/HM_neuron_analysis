"""LaTeX report: fingerprinting, table writer, section text from tables."""

import numpy as np
import pandas as pd

from hm_neuron import report
from hm_neuron.analyses import field_count
from hm_neuron.config import Config


def test_fingerprint_tracks_config_and_sessions():
    ses = pd.DataFrame(dict(rat=["Rat5", "Rat5"], date=["20260623", "20260624"]))
    cfg = Config()
    fp = report.fingerprint("field_count", cfg, ses)
    assert fp == report.fingerprint("field_count", cfg, ses)          # deterministic
    assert fp != report.fingerprint("theta", cfg, ses)                # other analysis source
    assert fp != report.fingerprint("field_count", cfg, ses.iloc[:1])  # session set
    cfg2 = Config()
    cfg2.fields.field_frac = 0.5
    assert fp != report.fingerprint("field_count", cfg2, ses)         # parameter
    cfg3 = Config()
    cfg3.out_dir = "elsewhere"
    assert fp == report.fingerprint("field_count", cfg3, ses)         # output location is irrelevant


def test_table_and_escaping():
    df = pd.DataFrame(dict(name=["a_b", "50%"], x=[1.23456, 1e-7], n=[3, 12000]))
    tex = report.table(df, "cap", "tab:x")
    assert r"a\_b" in tex and r"50\%" in tex
    assert r"\begin{tabular}{lrr}" in tex
    assert r"\times10^{-7}" in tex and "12{,}000" in tex
    assert tex.count(r"\\") == 3


def test_field_count_latex_section():
    rng = np.random.default_rng(0)
    k = 1 + rng.negative_binomial(2, 0.4, 300)
    U = pd.DataFrame(dict(unit=np.arange(300), rat=np.where(np.arange(300) < 150, "Rat5", "Rat6"),
                          session_id="s", is_place=True, n_fields=k,
                          n_fields_null=k * 0.5, n_fields_null_1=k))
    fits = pd.concat([field_count.fit(k).assign(group="pooled")])
    fits["params"] = fits["params"].apply(lambda v: ";".join(f"{x:.4g}" for x in v))
    summ = pd.DataFrame(dict(rat=["Rat5"], n_cells=[150], n_sessions=[1], mean=[2.0], median=[2.0],
                             var=[3.0], fano=[1.5]))
    conf = pd.DataFrame(dict(test=["n_fields ~ day (Spearman)"], rho=[0.1], p=[0.2], n=[300]))
    tex = field_count.latex(dict(units=U, fits=fits, summary=summ, confounds=conf))
    assert r"\begin{table}" in tex and "best model by AIC" in tex
    import re
    plain = re.sub(r"\$[^$]*\$", "", tex).replace(r"\_", "").replace("tab:field_count", "")
    assert "_" not in plain   # no unescaped underscores outside labels and math
