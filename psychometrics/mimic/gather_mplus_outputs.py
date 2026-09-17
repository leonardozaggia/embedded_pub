# =============================================================================
# gather_mplus_outputs.py  —  Gather the STDY MIMIC effects from the Mplus outputs
# =============================================================================
# Purpose:
#   Parses free MIMIC model .out files from all predictor subfolders and
#   generates radar (spider) charts visualising z-ratios (Est./S.E.) for each
#   predictor across the three cognitive factors (WM, GDP, FS) and, where
#   available, the unidimensional factor (COG).
#   Z-ratios are scale-invariant and comparable across continuous and binary
#   predictors. A constant offset is added so all values stay positive
#   (inner ring = null effect, z = 0).
#
# Inputs:
#   <THIS_DIR>/<PRED>/model_mimic_<PRED>.out      free 3-factor MIMIC output
#   <THIS_DIR>/<PRED>/model_mimic_<PRED>_unidim.out  unidim output (if present)
#
# Intermediary file produced:
#   <THIS_DIR>/mimic_stdy_effects.csv   parsed STDY rows; used as input by
#                                       plot_mimic_pub.R for publication figures
#
# Outputs:
#   <THIS_DIR>/radar_plots/radar_<PRED>.png   individual radar chart per predictor
#   <THIS_DIR>/radar_plots/radar_combined.png all predictors on one figure
#
# Usage:
#   python psychometrics/mimic/gather_mplus_outputs.py
# =============================================================================

import os
import sys, re, glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from math import pi

# ── Paths ─────────────────────────────────────────────────────────────────────
HERE     = os.path.dirname(os.path.abspath(__file__))
DEMO_DIR = HERE                                  # subfolders live here
PLOT_DIR = os.path.join(HERE, 'radar_plots')
os.makedirs(PLOT_DIR, exist_ok=True)

# ── Predictor configuration ────────────────────────────────────────────────────
# subfolder -> Mplus output predictor names (<=8 chars, uppercase)
# EthSAsian (9 chars) -> ETHSASIA in Mplus output
SUBFOLDER_PREDS = {
    'GA':      ['GA'],
    'BirthWt': ['BIRTHWT'],
    'Sex':     ['SEX'],
    'HdCirc':  ['HDCIRC'],
    'IUGR':    ['IUGR'],
    'GDiab':   ['GDIAB'],
    'PrEcl':   ['PRECL'],
    'AntCor':  ['ANTCOR_P', 'ANTCOR_C'],
    'MomAge':  ['MOMAGE'],
    'MomEdu':  ['MOMEDU'],
    'DadEdu':  ['DADEDU'],
    'Eth':     ['ETHSASIAN', 'ETHBLACK', 'ETHMIXED', 'ETHOTHER'],
    'MomEng':  ['MOMENG'],
    'EPDSTot': ['EPDSTOT'],
    'StimEnv': ['STIMENV'],
    'PrLax':   ['PRLAX'],
    'PrOvr':   ['PROVR'],
    'PrVrb':   ['PRVRB'],
    'PrSty':   ['PRSTY'],
}

PRED_LABEL = {
    'GA':       'GA (wks)',
    'BIRTHWT':  'Birth Wt (g)',
    'SEX':      'Sex (M vs F)',
    'HDCIRC':   'Head Circ (cm)',
    'IUGR':     'IUGR (Yes vs No)',
    'GDIAB':    'Gest. Diabetes',
    'PRECL':    'Pre-eclampsia',
    'ANTCOR_P': 'AntCor Partial',
    'ANTCOR_C': 'AntCor Complete',
    'MOMAGE':   'Maternal Age (y)',
    'MOMEDU':   'Maternal Edu (y)',
    'DADEDU':   'Paternal Edu (y)',
    'ETHSASIAN': 'S. Asian vs White',
    'ETHBLACK': 'Black vs White',
    'ETHMIXED': 'Mixed vs White',
    'ETHOTHER': 'Other vs White',
    'MOMENG':   'Maternal English',
    'EPDSTOT':  'Postnatal depression (EPDS)',
    'STIMENV':  'Stimulating home environment',
    'PRLAX':    'Parenting laxness',
    'PROVR':    'Parenting overreactivity',
    'PRVRB':    'Parenting verbosity',
    'PRSTY':    'Parenting Style',
}

PRED_SHORT = {
    'GA':       'GA',
    'BIRTHWT':  'BirthWt',
    'SEX':      'Sex',
    'HDCIRC':   'HdCirc',
    'IUGR':     'IUGR',
    'GDIAB':    'GDiab',
    'PRECL':    'PrEcl',
    'ANTCOR_P': 'AntCor\n(Partial)',
    'ANTCOR_C': 'AntCor\n(Complete)',
    'MOMAGE':   'MomAge',
    'MOMEDU':   'MomEdu',
    'DADEDU':   'DadEdu',
    'ETHSASIAN': 'Eth:\nS.Asian',
    'ETHBLACK': 'Eth:\nBlack',
    'ETHMIXED': 'Eth:\nMixed',
    'ETHOTHER': 'Eth:\nOther',
    'MOMENG':   'MomEng',
    'EPDSTOT':  'EPDSTot',
    'STIMENV':  'StimEnv',
    'PRLAX':    'PrLax',
    'PROVR':    'PrOvr',
    'PRVRB':    'PrVrb',
    'PRSTY':    'PrSty',
}

ALL_FACTORS   = ['WM', 'GDP', 'FS', 'COG']
FACTOR_COLOR  = {'WM': '#1565C0', 'GDP': '#E65100', 'FS': '#2E7D32', 'COG': '#6A1B9A'}
FACTOR_FILL   = {'WM': '#90CAF9', 'GDP': '#FFCC80', 'FS': '#A5D6A7', 'COG': '#CE93D8'}
FACTOR_LABEL  = {
    'WM':  'Working Memory (WM)',
    'GDP': 'Goal-Directed PS (GDP)',
    'FS':  'Flexibility / Shift (FS)',
    'COG': 'COG Unidim',
}

MULTI_COLORS = ['#1f77b4', '#d62728', '#2ca02c', '#9467bd',
                '#8c564b', '#e377c2', '#17becf', '#bcbd22']


# ── Parser ─────────────────────────────────────────────────────────────────────
def parse_stdy_on(out_path):
    """Return list of dicts from the first STDY Standardization > ON section."""
    try:
        with open(out_path, encoding='utf-8', errors='replace') as fh:
            lines = fh.readlines()
    except Exception:
        return []

    start = None
    for i, ln in enumerate(lines):
        if 'STDY Standardization' in ln:
            start = i
            break
    if start is None:
        return []

    end = len(lines)
    for i in range(start + 5, len(lines)):
        ln = lines[i]
        if ('Standardization' in ln or 'R-SQUARE' in ln
                or 'CONFIDENCE INTERVALS' in ln):
            end = i
            break

    results, factor, in_on = [], None, False

    for ln in lines[start:end]:
        s = ln.strip()
        if not s:
            in_on = False
            continue

        m = re.match(r'^([A-Z][A-Z0-9_]*)\s+ON\s*$', s, re.I)
        if m:
            factor  = m.group(1).upper()
            in_on   = True
            continue

        if (re.match(r'^[A-Z][A-Z0-9_]*\s+(BY|WITH)\s*$', s, re.I)
                or re.match(r'^(Thresholds|Intercepts|Means|Variances|Residual)\b', s, re.I)):
            in_on  = False
            factor = None
            continue

        if in_on and factor:
            parts = s.split()
            if len(parts) == 5:
                try:
                    est, se, z, p = (float(parts[1]), float(parts[2]),
                                     float(parts[3]), float(parts[4]))
                    if abs(z) < 900:
                        results.append({
                            'factor':    factor,
                            'predictor': parts[0].upper(),
                            'estimate':  est,
                            'se':        se,
                            'z_ratio':   z,
                            'p_value':   p,
                        })
                except ValueError:
                    pass
    return results


def sig_stars(pv):
    if pv < 0.001: return '***'
    if pv < 0.010: return '**'
    if pv < 0.050: return '*'
    if pv < 0.100: return '\u2020'
    return ''


def load_effects():
    rows = []
    for sf, preds in SUBFOLDER_PREDS.items():
        folder = os.path.join(DEMO_DIR, sf)
        if not os.path.isdir(folder):
            print(f'  WARNING: subfolder not found: {sf}')
            continue

        all_out = glob.glob(os.path.join(folder, '*.out'))
        f3_out  = [f for f in all_out
                   if '_UNIDIM' not in os.path.basename(f).upper()
                   and 'constraint' not in os.path.basename(f).lower()]
        ud_out  = [f for f in all_out
                   if '_UNIDIM' in os.path.basename(f).upper()]

        for paths, model, valid_factors in [
                (f3_out, '3F',     ['WM', 'GDP', 'FS']),
                (ud_out, 'UNIDIM', ['COG'])]:
            for path in paths:
                for rec in parse_stdy_on(path):
                    if rec['predictor'] in preds and rec['factor'] in valid_factors:
                        rec.update({
                            'subfolder': sf,
                            'model':     model,
                            'label':     PRED_LABEL.get(rec['predictor'], rec['predictor']),
                            'short':     PRED_SHORT.get(rec['predictor'], rec['predictor']),
                            'sig':       sig_stars(rec['p_value']),
                        })
                        rows.append(rec)

    df = pd.DataFrame(rows).drop_duplicates(
        subset=['subfolder', 'model', 'factor', 'predictor'])
    return df.sort_values(['subfolder', 'predictor', 'factor']).reset_index(drop=True)


# ── Radar utilities ────────────────────────────────────────────────────────────
def closed_angles(n):
    a = [k / n * 2 * pi for k in range(n)]
    return np.array(a + [a[0]])


def shifted(z_list, offset):
    return [offset + z for z in z_list] + [offset + z_list[0]]


def ref_ring(ax, r, color='#888888', lw=0.8, ls='--', alpha=0.5):
    t = np.linspace(0, 2 * pi, 360)
    ax.plot(t, [r] * 360, color=color, lw=lw, ls=ls, alpha=alpha, zorder=1)


def setup_polar(ax, labels, v_max, fontsize=11):
    n = len(labels)
    angles = [k / n * 2 * pi for k in range(n)]
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles)
    ax.set_xticklabels(labels, size=fontsize, fontweight='bold',
                       multialignment='center')
    ax.set_ylim(0, v_max)
    ax.set_yticks([])
    ax.grid(color='#dddddd', linestyle='--', linewidth=0.5, alpha=0.7)
    ax.spines['polar'].set_visible(False)
    return np.array(angles + [angles[0]])


# ── Individual plots (one per subfolder) ──────────────────────────────────────
def plot_individual(df, sf, preds):
    axes  = ALL_FACTORS
    n_ax  = len(axes)

    z_tbl = {}
    p_tbl = {}
    for pred in preds:
        zv, pv = [], []
        for fac in axes:
            model = 'UNIDIM' if fac == 'COG' else '3F'
            m = df[(df['subfolder'] == sf) & (df['predictor'] == pred)
                   & (df['factor'] == fac) & (df['model'] == model)]
            if len(m):
                zv.append(m.iloc[0]['z_ratio'])
                pv.append(m.iloc[0]['p_value'])
            else:
                zv.append(0.0); pv.append(1.0)
        z_tbl[pred] = zv
        p_tbl[pred] = pv

    all_z   = [z for zv in z_tbl.values() for z in zv]
    max_abs = max((abs(z) for z in all_z), default=2.5)
    max_abs = max(max_abs, 2.5)
    OFFSET  = max_abs + 1.8
    v_max   = OFFSET + max_abs + 2.0

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    angle_arr = setup_polar(ax, axes, v_max, fontsize=14)

    ref_ring(ax, OFFSET,        color='#888888', lw=1.4, ls='-',  alpha=0.55)
    for thr, col in [(1.96, '#FFA040'), (2.58, '#FF5555'), (3.29, '#CC0000')]:
        ref_ring(ax, OFFSET + thr,             col, lw=0.8)
        ref_ring(ax, max(0.02*v_max, OFFSET - thr), col, lw=0.8)

    for idx, pred in enumerate(preds):
        col   = MULTI_COLORS[idx % len(MULTI_COLORS)]
        lbl   = PRED_LABEL.get(pred, pred)
        vals  = shifted(z_tbl[pred], OFFSET)
        ax.plot(angle_arr, vals, color=col, lw=2.5, label=lbl, zorder=3+idx)
        ax.fill(angle_arr, vals, color=col, alpha=0.15, zorder=2+idx)

        for ai, (ang, z, pv) in enumerate(
                zip(angle_arr[:-1], z_tbl[pred], p_tbl[pred])):
            dv  = OFFSET + z
            sz  = 80 if pv < 0.05 else 35
            ec  = 'black' if pv < 0.05 else col
            ax.scatter(ang, dv, s=sz, color=col, edgecolors=ec, lw=1.2, zorder=6)
            sig = sig_stars(pv)
            if sig:
                ax.text(ang, dv + v_max * 0.045, sig,
                        ha='center', va='bottom', fontsize=11,
                        color=col, fontweight='bold', zorder=7)

    top_ang = 0.0
    for z_val in [-1.96, 0, 1.96, 3.29]:
        rv = OFFSET + z_val
        if 0 < rv < v_max:
            lbl_str = f'z={z_val:+.2f}' if z_val != 0 else 'z = 0'
            ax.text(top_ang + 0.08, rv, lbl_str,
                    fontsize=7, color='#555555', ha='left', va='center')

    ax.set_title(
        f'{sf}  \u2014  z-ratio profile across cognitive factors\n'
        f'(radius = z-ratio; grey ring = null effect)',
        fontsize=11, pad=22, fontweight='bold')

    if len(preds) > 1:
        ax.legend(loc='upper right', bbox_to_anchor=(1.50, 1.18),
                  fontsize=9, framealpha=0.88, title='Predictor')

    fig.text(0.02, 0.01,
             'Reference rings:  grey = z=0   orange = |z|=1.96 (p<.05)'
             '   red = |z|=2.58 (p<.01)   dark red = |z|=3.29 (p<.001)',
             fontsize=7, color='#555555')

    plt.tight_layout()
    out_path = os.path.join(PLOT_DIR, f'radar_{sf}.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'  Saved: {out_path}')


# ── Combined plot ──────────────────────────────────────────────────────────────
def plot_combined(df):
    all_preds = [p for sf in SUBFOLDER_PREDS for p in SUBFOLDER_PREDS[sf]]
    n_pred    = len(all_preds)

    z_mat = {}
    p_mat = {}
    for fac in ALL_FACTORS:
        model = 'UNIDIM' if fac == 'COG' else '3F'
        zv, pv = [], []
        for pred in all_preds:
            sf = next(s for s, ps in SUBFOLDER_PREDS.items() if pred in ps)
            m  = df[(df['predictor'] == pred) & (df['factor'] == fac)
                    & (df['model'] == model)]
            zv.append(m.iloc[0]['z_ratio'] if len(m) else 0.0)
            pv.append(m.iloc[0]['p_value'] if len(m) else 1.0)
        z_mat[fac] = zv
        p_mat[fac] = pv

    flat_z  = [z for zv in z_mat.values() for z in zv]
    max_abs = max((abs(z) for z in flat_z), default=3.0)
    OFFSET  = max_abs + 1.5
    v_max   = OFFSET + max_abs + 2.5

    fig = plt.figure(figsize=(22, 22), facecolor='white')
    ax  = fig.add_subplot(111, polar=True)

    short_labels = [PRED_SHORT.get(p, p) for p in all_preds]
    angle_arr    = setup_polar(ax, short_labels, v_max, fontsize=11)

    ref_ring(ax, OFFSET, color='#888888', lw=1.6, ls='-', alpha=0.55)
    for thr, col in [(1.96, '#FFA040'), (2.58, '#FF5555'), (3.29, '#CC0000')]:
        ref_ring(ax, OFFSET + thr,                  col, lw=1.0)
        ref_ring(ax, max(0.02*v_max, OFFSET - thr), col, lw=1.0)

    group_boundaries = []
    idx = 0
    for sf in SUBFOLDER_PREDS:
        idx += len(SUBFOLDER_PREDS[sf])
        group_boundaries.append(idx)

    for fac in ALL_FACTORS:
        col      = FACTOR_COLOR[fac]
        fill_col = FACTOR_FILL[fac]
        vals     = shifted(z_mat[fac], OFFSET)

        ax.plot(angle_arr, vals, color=col, lw=3.0,
                label=FACTOR_LABEL[fac], zorder=4)
        ax.fill(angle_arr, vals, color=fill_col, alpha=0.12, zorder=3)

        for ai, (ang, z, pv) in enumerate(
                zip(angle_arr[:-1], z_mat[fac], p_mat[fac])):
            dv  = OFFSET + z
            sz  = max(18, min(220, abs(z) * 32))
            ec  = 'black' if pv < 0.05 else col
            lw  = 1.8   if pv < 0.05 else 0.4
            ax.scatter(ang, dv, s=sz, color=fill_col, edgecolors=ec,
                       lw=lw, alpha=0.92, zorder=6)

    for z_val, lbl_str in [(-3.29, 'z=-3.29'), (-1.96, 'z=-1.96'),
                            (0, 'z = 0'), (1.96, 'z=+1.96'), (3.29, 'z=+3.29')]:
        rv = OFFSET + z_val
        if 0 < rv < v_max:
            ax.text(0.06, rv, lbl_str, fontsize=8, color='#555555',
                    ha='left', va='center')

    group_starts = [0] + group_boundaries[:-1]
    for sf, g_start, g_end in zip(SUBFOLDER_PREDS.keys(),
                                   group_starts, group_boundaries):
        if len(SUBFOLDER_PREDS[sf]) <= 1:
            continue
        mid_i = (g_start + g_end - 1) / 2
        mid_a = mid_i / n_pred * 2 * pi
        ax.text(mid_a, v_max * 1.08, sf,
                ha='center', va='center', fontsize=9,
                color='#333333', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.2', fc='#f0f0f0', ec='none', alpha=0.7))

    ax.legend(loc='lower left', bbox_to_anchor=(-0.15, -0.15),
              fontsize=13, framealpha=0.95,
              title='Cognitive Factor', title_fontsize=13, ncol=2)

    ax.set_title(
        'MIMIC Model \u2014 All Predictors \u00d7 Cognitive Factors\n'
        'Radius = z-ratio (Est./S.E.)  \u00b7  Grey ring = null (z=0)'
        '  \u00b7  Outlined dots = p < .05',
        fontsize=16, pad=40, fontweight='bold')

    out_path = os.path.join(PLOT_DIR, 'radar_combined.png')
    plt.savefig(out_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'  Saved: {out_path}')


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print('Loading STDY ON effects from all per-predictor subfolders...\n')
    df = load_effects()

    csv_path = os.path.join(HERE, 'mimic_stdy_effects.csv')
    col_order = ['subfolder', 'predictor', 'label', 'model', 'factor',
                 'estimate', 'se', 'z_ratio', 'p_value', 'sig']
    df[col_order].to_csv(csv_path, index=False)
    print(f'Saved CSV: {csv_path}  ({len(df)} rows)\n')

    pd.set_option('display.max_rows', 200)
    pd.set_option('display.width', 120)
    summary = df[['subfolder', 'predictor', 'model', 'factor',
                  'estimate', 'z_ratio', 'p_value', 'sig']].copy()
    summary['estimate'] = summary['estimate'].round(3)
    summary['z_ratio']  = summary['z_ratio'].round(3)
    summary['p_value']  = summary['p_value'].round(3)
    print(summary.to_string(index=False))

    # Diagnostic radar charts (not part of the manuscript): opt in with --radar
    if '--radar' in sys.argv:
        print('\n-- Individual radar plots --')
        for sf, preds in SUBFOLDER_PREDS.items():
            if len(df[df['subfolder'] == sf]) == 0:
                print(f'  No data for {sf}, skipping')
                continue
            plot_individual(df, sf, preds)

        print('\n-- Combined radar plot --')
        plot_combined(df)
        print(f'\nDone.  Plots saved to: {PLOT_DIR}')
    else:
        print('\nDone.  (pass --radar to also draw the diagnostic radar charts)')
