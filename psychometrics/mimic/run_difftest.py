# =============================================================================
# run_difftest.py  —  WLSMV chi-square difference tests for demographics MIMIC
# =============================================================================
# Purpose:
#   Tests whether constraining two factor paths equal (e.g. WM = FS for BirthWt)
#   significantly worsens model fit, using Mplus DIFFTEST for the WLSMV estimator.
#   Standard chi-square subtraction is invalid under WLSMV; DIFFTEST accounts for
#   the mean-and-variance correction via a scaling matrix saved from the free model.
#
# Workflow (per predictor subfolder):
#   1. Inject  SAVEDATA: DIFFTEST = "difftest_free.dat";  into free model .inp
#      (in-place, idempotent — skipped if line already present)
#   2. Run free model  →  produces difftest_free.dat (scaling matrix)
#   3. Inject  ANALYSIS: DIFFTEST = "difftest_free.dat";  into each constrained .inp
#      (in-place, idempotent)
#   4. Run constrained model  →  .out contains "Chi-Square Test for Difference Testing"
#   5. Parse chi2_diff, df, p from that block
#
# Inputs (per subfolder of this directory):
#   <PRED>/model_mimic_<PRED>.inp              free MIMIC model (modified in-place)
#   <PRED>/model_mimic_<PRED>_constraint*.inp  equality-constrained models (modified in-place)
#
# Intermediary files produced (per subfolder):
#   <PRED>/difftest_free.dat                   Mplus scaling matrix required by DIFFTEST
#   <PRED>/model_mimic_<PRED>.out              free model output (re-run with SAVEDATA block)
#   <PRED>/model_mimic_<PRED>_constraint*.out  constrained output with DIFFTEST section
#
# Outputs (saved in this directory):
#   difftest_results.csv   table: constraint, chi2_diff, df, p_value, sig for every test
#   difftest_results.png   horizontal bar chart sorted by chi2_diff, coloured by significance
#
# Usage:
#   python psychometrics/mimic/run_difftest.py                # run Mplus + parse
#   python psychometrics/mimic/run_difftest.py --parse-only   # only re-parse the
#        committed .out files (no Mplus licence needed) and rebuild the CSV/PNG
#
# The Mplus executable is taken from the MPLUS_EXE environment variable
# (default: C:\Program Files\Mplus\Mplus.exe).
# =============================================================================

import argparse
import os, re, subprocess
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import chi2 as chi2dist

MPLUS = os.environ.get("MPLUS_EXE", r"C:\Program Files\Mplus\Mplus.exe")
BASE  = os.path.dirname(os.path.abspath(__file__))

_ap = argparse.ArgumentParser(description="WLSMV DIFFTEST for the MIMIC factor contrasts")
_ap.add_argument("--parse-only", action="store_true",
                 help="do not run Mplus; parse the existing .out files only")
ARGS = _ap.parse_args()

# (predictor_folder, free_inp, [(constrained_inp, label), ...])
PAIRS = [
    ("AntCor", "model_mimic_AntCor.inp", [
        ("model_mimic_AntCor_constraint_a.inp", "AntCor_P: WM = GDP"),
        ("model_mimic_AntCor_constraint_b.inp", "AntCor_C: WM = FS"),
    ]),
    ("BirthWt", "model_mimic_BirthWt.inp", [
        ("model_mimic_BirthWt_constraint.inp", "BirthWt: WM = FS"),
    ]),
    ("DadEdu", "model_mimic_DadEdu.inp", [
        ("model_mimic_DadEdu_constraint.inp", "DadEdu: WM = GDP"),
    ]),
    ("GA", "model_mimic_GA.inp", [
        ("model_mimic_GA_constraint.inp", "GA: WM = FS"),
    ]),
    ("GDiab", "model_mimic_GDiab.inp", [
        ("model_mimic_GDiab_constraint.inp", "GDiab: WM = GDP"),
    ]),
    ("HdCirc", "model_mimic_HdCirc.inp", [
        ("model_mimic_HdCirc_constraint.inp", "HdCirc: WM = FS"),
    ]),
    ("IUGR", "model_mimic_IUGR.inp", [
        ("model_mimic_IUGR_constraint.inp", "IUGR: GDP = FS"),
    ]),
    ("MomAge", "model_mimic_MomAge.inp", [
        ("model_mimic_MomAge_constraint.inp", "MomAge: WM = GDP"),
    ]),
    ("MomEdu", "model_mimic_MomEdu.inp", [
        ("model_mimic_MomEdu_constraint.inp", "MomEdu: WM = GDP"),
    ]),
    ("MomEng", "model_mimic_MomEng.inp", [
        ("model_mimic_MomEng_constraint.inp", "MomEng: WM = FS"),
    ]),
    ("PrEcl", "model_mimic_PrEcl.inp", [
        ("model_mimic_PrEcl_constraint.inp", "PrEcl: WM = GDP"),
    ]),
    ("Sex", "model_mimic_Sex.inp", [
        ("model_mimic_Sex_constraint.inp", "Sex: GDP = FS"),
    ]),
    ("StimEnv", "model_mimic_StimEnv.inp", [
        ("model_mimic_StimEnv_constraint.inp", "StimEnv: WM = FS"),
    ]),
    ("EPDSTot", "model_mimic_EPDSTot.inp", [
        ("model_mimic_EPDSTot_constraint.inp", "epds: GDP = FS"),
    ]),
    ("PrLax", "model_mimic_PrLax.inp", [
        ("model_mimic_PrLax_constraint.inp", "prlax: WM = GDP"),
    ]),
    ("PrOvr", "model_mimic_PrOvr.inp", [
        ("model_mimic_PrOvr_constraint.inp", "provr: WM = GDP"),
    ]),
    ("PrVrb", "model_mimic_PrVrb.inp", [
        ("model_mimic_PrVrb_constraint.inp", "prvrb: GDP = FS"),
    ]),
    ("PrSty", "model_mimic_PrSty.inp", [
        ("model_mimic_PrSty_constraint.inp", "prsty: WM = FS"),
    ]),
]

# ── helpers ──────────────────────────────────────────────────────────────────

def inject_difftest_save(path):
    """Ensure SAVEDATA: DIFFTEST line is in the free model .inp (idempotent)."""
    txt = open(path, encoding="utf-8", errors="ignore").read()
    if 'DIFFTEST = "difftest_free.dat"' in txt:
        return
    if "SAVEDATA:" in txt:
        txt = txt.replace("SAVEDATA:", 'SAVEDATA:\n  DIFFTEST = "difftest_free.dat";', 1)
    else:
        txt += '\nSAVEDATA:\n  DIFFTEST = "difftest_free.dat";\n'
    open(path, "w", encoding="utf-8").write(txt)


def inject_difftest_load(path):
    """Ensure ANALYSIS: DIFFTEST line is in the constrained model .inp (idempotent)."""
    txt = open(path, encoding="utf-8", errors="ignore").read()
    if 'DIFFTEST = "difftest_free.dat"' in txt:
        return
    txt = re.sub(
        r"(ESTIMATOR\s*=\s*WLSMV\s*;)",
        r'\1\n  DIFFTEST = "difftest_free.dat";',
        txt, count=1, flags=re.IGNORECASE
    )
    open(path, "w", encoding="utf-8").write(txt)


def run_mplus(folder, inp):
    if ARGS.parse_only:
        print(f"    (parse-only) skipping mplus {inp}", flush=True)
        return None
    print(f"    mplus {inp}", flush=True)
    r = subprocess.run([MPLUS, inp], cwd=folder,
                       capture_output=True, text=True, timeout=600)
    if r.returncode not in (0, 1):          # Mplus often returns 1 on warnings
        print(f"      [stderr] {r.stderr[:200]}")
    return r


def find_output_path(inp_path):
    """Return matching .out path, allowing for Mplus filename case changes."""
    expected = os.path.splitext(inp_path)[0] + ".out"
    if os.path.exists(expected):
        return expected

    folder = os.path.dirname(inp_path)
    wanted = os.path.basename(expected).lower()
    for name in os.listdir(folder):
        if name.lower() == wanted:
            return os.path.join(folder, name)

    return expected


def parse_difftest_out(out_path):
    """
    Return (chi2_diff, df, p_value) from a constrained .out run with DIFFTEST.
    Mplus 8 writes the block:
        Chi-Square Test for Difference Testing
          Value                  xx.xxx*
          Degrees of Freedom         1
          P-Value               x.xxxx
    """
    if not os.path.exists(out_path):
        print(f"      [parse] .out not found: {out_path}")
        return None, None, None

    txt = open(out_path, encoding="utf-8", errors="ignore").read()

    patterns = [
        # Mplus 8 actual format: labeled rows
        r"Chi-Square Test for Difference Testing\s+Value\s+([\d]+\.[\d]+)\*?\s+Degrees of Freedom\s+(\d+)\s+P-Value\s+([\d]+\.[\d]+)",
        # Compact table variant: Value Df P-Value on same line
        r"Chi-Square Test for Difference Testing\s+Value\s+Df\s+P-Value\s+([\d]+\.[\d]+)\*?\s+(\d+)\s+([\d]+\.[\d]+)",
        # Older Mplus layout without "for"
        r"Chi-Square\s+Difference\s+Test(?:ing)?\s+Value\s+([\d]+\.[\d]+)\*?\s+Degrees of Freedom\s+(\d+)\s+P-Value\s+([\d]+\.[\d]+)",
        # Generic fallback
        r"(?:Difference\s+Test|DIFFTEST)[^\n]*\n(?:[^\n]*\n){0,5}\s+([\d]+\.[\d]+)\*?\s+(\d+)\s+([\d]+\.[\d]+)",
    ]
    for p in patterns:
        m = re.search(p, txt, re.IGNORECASE | re.DOTALL)
        if m:
            return float(m.group(1)), int(m.group(2)), float(m.group(3))

    # Line-by-line fallback scanning for the labeled block
    lines = txt.split("\n")
    for i, line in enumerate(lines):
        if "DIFFERENCE TESTING" in line.upper():
            chi2v = dfv = pv = None
            for j in range(i + 1, min(i + 15, len(lines))):
                lj = lines[j]
                mv = re.search(r"Value\s+([\d]+\.[\d]+)", lj)
                md = re.search(r"Degrees of Freedom\s+(\d+)", lj)
                mp = re.search(r"P-Value\s+([\d]+\.[\d]+)", lj)
                if mv: chi2v = float(mv.group(1))
                if md: dfv   = int(md.group(1))
                if mp: pv    = float(mp.group(1))
            if chi2v is not None and dfv is not None and pv is not None:
                return chi2v, dfv, pv

    print(f"      [parse] DIFFTEST block not found in {os.path.basename(out_path)}")
    return None, None, None


# ── main loop ─────────────────────────────────────────────────────────────────

rows = []

for pred, free_inp, cons_list in PAIRS:
    folder = os.path.join(BASE, pred)
    print(f"\n{'='*55}\n  Predictor: {pred}\n{'='*55}")

    free_src   = os.path.join(folder, free_inp)
    difftest_f = os.path.join(folder, "difftest_free.dat")

    # 1. inject DIFFTEST save into free model (in-place, idempotent)
    if not ARGS.parse_only:
        inject_difftest_save(free_src)

    # 2. run free model
    run_mplus(folder, os.path.basename(free_src))

    if not os.path.exists(difftest_f):
        print(f"  [!] difftest_free.dat not created — skipping {pred}")
        for _, lbl in cons_list:
            rows.append(dict(predictor=pred, constraint=lbl,
                             chi2_diff=None, df=None, p_value=None))
        continue

    # 3+4. each constrained model
    for c_inp, c_lbl in cons_list:
        c_src = os.path.join(folder, c_inp)
        if not ARGS.parse_only:
            inject_difftest_load(c_src)
        run_mplus(folder, os.path.basename(c_src))

        c_out = find_output_path(c_src)
        chi2v, dfv, pv = parse_difftest_out(c_out)
        print(f"    {c_lbl:35s}  chi2={chi2v}  df={dfv}  p={pv}")
        rows.append(dict(predictor=pred, constraint=c_lbl,
                         chi2_diff=chi2v, df=dfv, p_value=pv))

# ── save table ────────────────────────────────────────────────────────────────

df = pd.DataFrame(rows)

def sig_stars(p):
    if p is None:  return "—"
    if p < .001:   return "***"
    if p < .01:    return "**"
    if p < .05:    return "*"
    return "ns"

df["sig"] = df["p_value"].apply(sig_stars)
df["significant"] = df["p_value"].apply(lambda p: p < .05 if p is not None else False)

csv_path = os.path.join(BASE, "difftest_results.csv")
df.to_csv(csv_path, index=False, float_format="%.4f")
print(f"\nResults table saved -> {csv_path}")
print(df.to_string(index=False))

# ── figure ────────────────────────────────────────────────────────────────────

valid = df.dropna(subset=["chi2_diff"]).copy()
valid = valid.sort_values("chi2_diff", ascending=True).reset_index(drop=True)

CRIT      = chi2dist.ppf(0.95, df=1)   # 3.841
SIG_COL   = "#B03A2E"
NSIG_COL  = "#2471A3"
CRIT_COL  = "#1C2833"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 14,
    "axes.titlesize": 18,
    "axes.labelsize": 15,
})

n = len(valid)
fig, ax = plt.subplots(figsize=(13, max(7, n * 0.62 + 2.5)))
fig.patch.set_facecolor("white")
ax.set_facecolor("#F8F9FA")

for i, row in valid.iterrows():
    col   = SIG_COL  if row.significant else NSIG_COL
    alpha = 1.0      if row.significant else 0.55
    ax.barh(i, row.chi2_diff, color=col, alpha=alpha,
            height=0.58, zorder=3, linewidth=0)

ax.axvline(CRIT, color=CRIT_COL, lw=1.6, ls="--", zorder=4)
ax.axvspan(CRIT, valid["chi2_diff"].max() * 1.55,
           alpha=0.04, color=SIG_COL, zorder=1)

ax.set_yticks(range(n))
ax.set_yticklabels(valid["constraint"], fontsize=14)
ax.set_xlabel("WLSMV \u03c7\u00b2 Difference  (df\u00a0=\u00a01)", fontsize=15, labelpad=10)
ax.set_title(
    "WLSMV Chi-Square Difference Tests\n"
    "Constrained vs. Free MIMIC Models",
    fontsize=18, fontweight="bold", pad=16
)
ax.set_xlim(0, valid["chi2_diff"].max() * 1.55)
ax.tick_params(axis="y", length=0, pad=6)
ax.tick_params(axis="x", labelsize=13)

ax.grid(axis="x", color="white", linewidth=1.2, zorder=2)
for spine in ["top", "right", "left"]:
    ax.spines[spine].set_visible(False)
ax.spines["bottom"].set_color("#AAAAAA")

for i, row in valid.iterrows():
    p_str  = "p\u00a0<\u00a0.001" if row.p_value < .001 else f"p\u00a0=\u00a0{row.p_value:.3f}"
    stars  = f"  {row.sig}" if row.sig != "ns" else ""
    label  = f"\u03c7\u00b2 = {row.chi2_diff:.2f}   {p_str}{stars}"
    tcol   = "#1a1a1a" if row.significant else "#666666"
    ax.text(row.chi2_diff + 0.18, i, label,
            va="center", ha="left", fontsize=13, color=tcol, zorder=5,
            bbox=dict(facecolor="#F8F9FA", edgecolor="none", alpha=0.85, pad=0.5))

ax.text(CRIT + 0.12, -0.7,
        f"critical\u00a0\u03c7\u00b2\u00a0=\u00a0{CRIT:.2f}",
        fontsize=12, color=CRIT_COL, va="top", ha="left", style="italic")

from matplotlib.lines import Line2D
handles = [
    mpatches.Patch(color=SIG_COL,  label="Significant  (p\u00a0<\u00a0.05)"),
    mpatches.Patch(color=NSIG_COL, alpha=0.55, label="Non-significant  (p\u00a0\u2265\u00a0.05)"),
    Line2D([0], [0], color=CRIT_COL, ls="--", lw=1.6,
           label=f"\u03b1\u00a0=\u00a0.05 threshold  (\u03c7\u00b2\u00a0=\u00a0{CRIT:.2f})"),
]
ax.legend(handles=handles, fontsize=13, framealpha=0.92,
          edgecolor="#CCCCCC", loc="lower right")

plt.tight_layout(pad=1.8)
fig_path = os.path.join(BASE, "difftest_results.png")
fig.savefig(fig_path, dpi=180, bbox_inches="tight", facecolor="white")
print(f"Figure saved -> {fig_path}")

plt.close()

print("\nDone.")
