"""
_clinical.py — shared data loading + dot-plot drawing for the two clinical
association blocks (MIMIC covariates and factor–outcome correlations) that
make up Figure 4's panels A and B (fig4_clinical_merged.py).

Both blocks share the exact same visual grammar (4 panels = 3 domain factors +
general-cognition composite; horizontal lollipop per predictor/outcome; colour =
predictor/outcome domain; filled dot = p < .05). Keeping it here means panels A
and B are guaranteed identical in look. (The formerly-separate standalone
fig3_mimic.py / fig4_outcomes.py scripts that also used this module have been
retired — superseded by the merged Figure 4.)

Data contracts (committed CSVs):
  MIMIC     : psychometrics/mimic/mimic_std_effects.csv    (+ difftest_results.csv)
              (STDYX for continuous covariates, STDY for binary ones)
  Outcomes  : psychometrics/outcomes/outcomes_associations.csv

In both files the three domain factors live in the 3-factor model
(model == "3F", factor in {WM, GDP, FS}) and the composite lives in the
unidimensional model (model == "UNIDIM", factor == "COG").
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

import lancet_style as ls

# The four panels, left to right (canonical order + header text)
PANELS = [
    ("WM",      "Working\nMemory"),
    ("GDPS",    "Goal-Directed\nProblem Solving"),
    ("FS",      "Flexibility/\nShift"),
    ("GENERAL", "General\nCognition"),
]

# ── MIMIC predictor metadata (order = top→bottom on the y-axis) ──────────────
PRED_META = [
    # code,      label,                                  domain
    ("GA",       "Gestational age (wks)",                "Neonatal"),
    ("BIRTHWT",  "Birth weight (kg)",                    "Neonatal"),
    ("HDCIRC",   "Head circumference (cm)",              "Neonatal"),
    ("SEX",      "Sex (male vs female)",                 "Neonatal"),
    ("IUGR",     "IUGR",                                 "Obstetric"),
    ("GDIAB",    "Gestational diabetes",                 "Obstetric"),
    ("PRECL",    "Pre-eclampsia",                        "Obstetric"),
    ("ANTCOR_P", "Antenatal corticosteroids (partial)",  "Obstetric"),
    ("ANTCOR_C", "Antenatal corticosteroids (complete)", "Obstetric"),
    ("MOMAGE",   "Maternal age (yrs)",                   "Sociodemographic"),
    ("MOMEDU",   "Maternal education (yrs)",             "Sociodemographic"),
    ("DADEDU",   "Paternal education (yrs)",             "Sociodemographic"),
    ("MOMENG",   "Maternal English first language",      "Sociodemographic"),
    ("EPDSTOT",  "Postnatal depression (EPDS total)",    "Sociodemographic"),
    ("PRLAX",    "Parenting laxness",                    "Sociodemographic"),
    ("PROVR",    "Parenting overreactivity",             "Sociodemographic"),
    ("PRVRB",    "Parenting verbosity",                  "Sociodemographic"),
    ("PRSTY",    "Parenting style (overall)",            "Sociodemographic"),
    ("STIMENV",  "Stimulating home environment",         "Sociodemographic"),
]

# difftest constraint-prefix → predictor code (bold labels = sig. factor contrast)
DIFF_PRED_MAP = {
    "AntCor_P": "ANTCOR_P", "AntCor_C": "ANTCOR_C", "BirthWt": "BIRTHWT",
    "DadEdu": "DADEDU", "GA": "GA", "GDiab": "GDIAB", "HdCirc": "HDCIRC",
    "IUGR": "IUGR", "MomAge": "MOMAGE", "MomEdu": "MOMEDU", "MomEng": "MOMENG",
    "PrEcl": "PRECL", "Sex": "SEX", "epds": "EPDSTOT", "prlax": "PRLAX",
    "provr": "PROVR", "prvrb": "PRVRB", "prsty": "PRSTY", "StimEnv": "STIMENV",
}

# ── Outcome metadata (key = subfolder) ───────────────────────────────────────
OUTCOME_META = [
    ("BayLan",   "Bayley Language",     "Bayley"),
    ("BayMot",   "Bayley Motor",        "Bayley"),
    ("CBInt",    "CBCL Internalising",  "CBCL Broad"),
    ("CBExt",    "CBCL Externalising",  "CBCL Broad"),
    ("CBTot",    "CBCL Total Problems", "CBCL Broad"),
    ("CBEmR",    "CBCL Emot. Reactive", "CBCL Narrow"),
    ("CBAnx",    "CBCL Anxious/Dep.",   "CBCL Narrow"),
    ("CBWth",    "CBCL Withdrawn/Dep.", "CBCL Narrow"),
    ("CBSom",    "CBCL Somatic",        "CBCL Narrow"),
    ("CBSlp",    "CBCL Sleep",          "CBCL Narrow"),
    ("CBAtt",    "CBCL Attention",      "CBCL Narrow"),
    ("CBAgg",    "CBCL Aggressive",     "CBCL Narrow"),
    ("QchatTot", "Q-CHAT Total",        "Q-CHAT"),
]


# ============================================================================
# Loading
# ============================================================================
def _pick(df, key_col, key, panel_factor):
    """Return (estimate, z, p) for one cell, or (nan, nan, nan) if absent."""
    if panel_factor == "GENERAL":
        sub = df[(df[key_col] == key) & (df["model"] == "UNIDIM") & (df["factor"] == "COG")]
    else:
        fac = "GDP" if panel_factor == "GDPS" else panel_factor
        sub = df[(df[key_col] == key) & (df["model"] == "3F") & (df["factor"] == fac)]
    if len(sub) == 0:
        return (np.nan, np.nan, np.nan)
    r = sub.iloc[0]
    return (float(r["estimate"]), float(r["z_ratio"]), float(r["p_value"]))


def load_mimic(root: Path):
    df = pd.read_csv(root / "psychometrics" / "mimic" / "mimic_std_effects.csv")
    diff = pd.read_csv(root / "psychometrics" / "mimic" / "difftest_results.csv")
    prefix = diff["constraint"].astype(str).str.split(":").str[0].str.strip()
    sig = diff["significant"].isin([True, "True"])
    sig_codes = {DIFF_PRED_MAP[p] for p in prefix[sig] if p in DIFF_PRED_MAP}
    return df, "predictor", PRED_META, sig_codes


def load_outcomes(root: Path):
    df = pd.read_csv(root / "psychometrics" / "outcomes" / "outcomes_associations.csv")
    return df, "subfolder", OUTCOME_META, set()   # no difftest for outcomes


# ============================================================================
# Drawing
# ============================================================================
def draw_block(axes4, df, key_col, meta, domain_palette, *, metric="estimate",
               sig_codes=None, x_lim=None, show_ylabels=True, ylabel_fs=8.0,
               dot_size=34, header_text_color="white", header_h=0.055,
               header_fs=8.5, plain_label_color="#8a8a8a"):
    """Draw one 4-panel association block into the 4 axes in `axes4`.

    metric : "estimate" (standardised β / latent correlation r) or "z"
             (z-statistic).
    Returns the (small, medium) reference-line magnitudes actually used.
    """
    sig_codes = sig_codes or set()
    n = len(meta)
    ys = list(range(n - 1, -1, -1))            # top row = highest y
    codes   = [m[0] for m in meta]
    labels  = [m[1] for m in meta]
    domains = [m[2] for m in meta]

    # group boundaries (between consecutive different domains)
    boundaries = [i for i in range(1, n) if domains[i] != domains[i - 1]]
    boundary_y = [ys[i] + 0.5 for i in boundaries]

    val_col = "estimate" if metric == "estimate" else "z"
    ref_small, ref_med = (0.10, 0.30) if metric == "estimate" else (1.96, 2.58)

    # collect all values to auto-scale x
    allvals = []
    cell = {}   # (panel_idx, code) -> (val, p)
    for pi, (fac, _) in enumerate(PANELS):
        for code in codes:
            est, z, p = _pick(df, key_col, code, fac)
            v = est if metric == "estimate" else z
            cell[(pi, code)] = (v, p)
            if not np.isnan(v):
                allvals.append(v)
    if x_lim is None:
        vmax = max(abs(min(allvals)), abs(max(allvals)))
        pad = 0.14 * vmax
        x_lim = (min(allvals) - pad, max(allvals) + pad)

    for pi, (ax, (fac, header)) in enumerate(zip(axes4, PANELS)):
        color_head = ls.factor_color(fac)
        ls.factor_header(ax, header, color_head, height=header_h,
                         text_color=header_text_color, fontsize=header_fs)

        # reference furniture
        for by in boundary_y:
            ax.axhline(by, color="#d9d9d9", lw=0.5, zorder=0)
        ax.axvline(0, color="#444444", lw=0.8, zorder=1)
        for r in (ref_small,):
            ax.axvline(r,  color="#9aa0a6", lw=0.5, ls=(0, (4, 3)), zorder=1)
            ax.axvline(-r, color="#9aa0a6", lw=0.5, ls=(0, (4, 3)), zorder=1)
        for r in (ref_med,):
            ax.axvline(r,  color="#b7bcc2", lw=0.5, ls=(0, (1, 2.2)), zorder=1)
            ax.axvline(-r, color="#b7bcc2", lw=0.5, ls=(0, (1, 2.2)), zorder=1)

        for code, dom, y in zip(codes, domains, ys):
            v, p = cell[(pi, code)]
            if np.isnan(v):
                continue
            c = domain_palette[dom]
            sig = (not np.isnan(p)) and (p < 0.05)
            seg_alpha = 1.0 if sig else 0.30
            ax.plot([0, v], [y, y], color=c, lw=1.6, alpha=seg_alpha,
                    solid_capstyle="round", zorder=2)
            if sig:
                ax.scatter([v], [y], s=dot_size, facecolor=c, edgecolor="white",
                           linewidth=0.6, zorder=3)
            else:
                ax.scatter([v], [y], s=dot_size, facecolor="white", edgecolor=c,
                           linewidth=1.0, alpha=0.9, zorder=3)

        ax.set_xlim(x_lim)
        ax.set_ylim(-0.7, n - 0.3)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.xaxis.set_major_formatter(
            ls.decimal_formatter(decimals=(2 if metric == "estimate" else None)))

        # Predictor / outcome labels: drawn manually on the leftmost panel so we
        # control per-label weight and colour (bold+dark = sig. factor contrast).
        if pi == 0 and show_ylabels:
            for code, label, dom, y in zip(codes, labels, domains, ys):
                bold = code in sig_codes
                ax.text(-0.035, y, label, transform=ax.get_yaxis_transform(),
                        ha="right", va="center", fontsize=ylabel_fs,
                        color="#111111" if bold else plain_label_color,
                        fontweight="bold" if bold else "normal", clip_on=False)

    return ref_small, ref_med


def domain_legend_handles(domain_palette):
    """Coloured dot handles for each domain (for the shared legend)."""
    return [mlines.Line2D([], [], marker="o", linestyle="none", markersize=6.5,
                          markerfacecolor=c, markeredgecolor=c, label=d)
            for d, c in domain_palette.items()]


def significance_legend_handles():
    return [
        mlines.Line2D([], [], marker="o", linestyle="none", markersize=6.5,
                      markerfacecolor="#555", markeredgecolor="#555", label="p < ·05"),
        mlines.Line2D([], [], marker="o", linestyle="none", markersize=6.5,
                      markerfacecolor="white", markeredgecolor="#555",
                      label="not significant"),
    ]
