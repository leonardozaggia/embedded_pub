# =============================================================================
# gather_std_effects.py  —  Mixed-standardization MIMIC effects table
# =============================================================================
# Builds mimic_std_effects.csv: the covariate -> factor regressions from the
# free MIMIC models using the reporting convention
#
#   continuous covariates  -> STDYX  (fully standardized: x and factor in SD units)
#   binary covariates      -> STDY   (semistandardized: factor SD units per group)
#
# Standardizing a 0/1 dummy by its own SD has no substantive interpretation,
# so binary covariates keep STDY (Mplus's own recommendation).
#
# The script also re-parses the STDY sections and cross-checks them against the
# committed mimic_stdy_effects.csv, refusing to write if anything mismatches
# (guards against the .out files and the CSV drifting apart, and against
# parser errors).
#
# Inputs:
#   <THIS_DIR>/<PRED>/model_mimic_<pred>.out         free 3-factor MIMIC output
#   <THIS_DIR>/<PRED>/model_mimic_<pred>_unidim.out  unidimensional output
#   <THIS_DIR>/mimic_stdy_effects.csv                reference for cross-check
#
# Output:
#   <THIS_DIR>/mimic_std_effects.csv
#     same columns as mimic_stdy_effects.csv plus `standardization`
#     (STDYX or STDY per row), so fig4_clinical_merged can switch to it by
#     repointing _clinical.load_mimic at this file.
#
# Usage:
#   python psychometrics/mimic/gather_std_effects.py
# =============================================================================

import glob
import os
import re
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))

# subfolder -> Mplus predictor names (mirrors gather_mplus_outputs.py)
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
    'MOMENG':   'Maternal English',
    'EPDSTOT':  'Postnatal depression (EPDS)',
    'STIMENV':  'Stimulating home environment',
    'PRLAX':    'Parenting laxness',
    'PROVR':    'Parenting overreactivity',
    'PRVRB':    'Parenting verbosity',
    'PRSTY':    'Parenting Style',
}

# 0/1 dummies keep STDY; everything else is a continuous scale -> STDYX
BINARY_PREDS = {'SEX', 'IUGR', 'GDIAB', 'PRECL', 'ANTCOR_P', 'ANTCOR_C', 'MOMENG'}


def parse_on_section(out_path, section):
    """Parse factor-ON-covariate rows from one standardization section.

    section : 'STDYX' or 'STDY'. Returns list of dicts.
    """
    try:
        with open(out_path, encoding='utf-8', errors='replace') as fh:
            lines = fh.readlines()
    except OSError:
        return []

    header = f'{section} Standardization'
    start = None
    for i, ln in enumerate(lines):
        if ln.strip() == header:
            start = i
            break
    if start is None:
        return []

    end = len(lines)
    for i in range(start + 1, len(lines)):
        s = lines[i].strip()
        if (s.endswith('Standardization') and s != header) \
                or s.startswith('R-SQUARE') or s.startswith('CONFIDENCE INTERVALS'):
            end = i
            break

    results, factor, in_on = [], None, False
    for ln in lines[start:end]:
        s = ln.strip()
        if not s:
            in_on = False
            continue
        m = re.match(r'^([A-Z][A-Z0-9_]*)\s+ON\s*$', s)
        if m:
            factor, in_on = m.group(1), True
            continue
        if (re.match(r'^[A-Z][A-Z0-9_]*\s+(BY|WITH)\s*$', s)
                or re.match(r'^(Thresholds|Intercepts|Means|Variances|Residual)\b', s)):
            factor, in_on = None, False
            continue
        if in_on and factor:
            parts = s.split()
            if len(parts) == 5:
                try:
                    est, se, z, p = (float(parts[1]), float(parts[2]),
                                     float(parts[3]), float(parts[4]))
                except ValueError:
                    continue
                if abs(z) < 900:
                    results.append({'factor': factor, 'predictor': parts[0],
                                    'estimate': est, 'se': se,
                                    'z_ratio': z, 'p_value': p})
    return results


def sig_stars(pv):
    if pv < 0.001: return '***'
    if pv < 0.010: return '**'
    if pv < 0.050: return '*'
    if pv < 0.100: return '†'
    return ''


def load_all():
    rows = []
    for sf, preds in SUBFOLDER_PREDS.items():
        folder = os.path.join(HERE, sf)
        if not os.path.isdir(folder):
            print(f'  WARNING: subfolder not found: {sf}')
            continue
        all_out = glob.glob(os.path.join(folder, '*.out'))
        f3_out = [f for f in all_out
                  if '_UNIDIM' not in os.path.basename(f).upper()
                  and 'constraint' not in os.path.basename(f).lower()]
        ud_out = [f for f in all_out if '_UNIDIM' in os.path.basename(f).upper()]

        for paths, model, valid_factors in [(f3_out, '3F', ['WM', 'GDP', 'FS']),
                                            (ud_out, 'UNIDIM', ['COG'])]:
            for path in paths:
                for section in ('STDYX', 'STDY'):
                    for rec in parse_on_section(path, section):
                        if rec['predictor'] in preds and rec['factor'] in valid_factors:
                            rec.update({'subfolder': sf, 'model': model,
                                        'standardization': section,
                                        'label': PRED_LABEL.get(rec['predictor'],
                                                                rec['predictor']),
                                        'sig': sig_stars(rec['p_value'])})
                            rows.append(rec)

    df = pd.DataFrame(rows).drop_duplicates(
        subset=['subfolder', 'model', 'factor', 'predictor', 'standardization'])
    return df.sort_values(['subfolder', 'predictor', 'model',
                           'factor', 'standardization']).reset_index(drop=True)


def crosscheck_stdy(df):
    """Freshly parsed STDY rows must match the committed mimic_stdy_effects.csv."""
    ref = pd.read_csv(os.path.join(HERE, 'mimic_stdy_effects.csv'))
    mine = df[df['standardization'] == 'STDY']
    n_bad = 0
    for _, r in ref.iterrows():
        m = mine[(mine['subfolder'] == r['subfolder'])
                 & (mine['predictor'] == r['predictor'])
                 & (mine['model'] == r['model'])
                 & (mine['factor'] == r['factor'])]
        if len(m) != 1:
            print(f'  MISSING: {r["subfolder"]} {r["predictor"]} '
                  f'{r["model"]} {r["factor"]}')
            n_bad += 1
            continue
        got = m.iloc[0]
        for col in ('estimate', 'se', 'z_ratio', 'p_value'):
            if abs(float(got[col]) - float(r[col])) > 1e-6:
                print(f'  MISMATCH: {r["subfolder"]} {r["predictor"]} {r["model"]} '
                      f'{r["factor"]} {col}: csv={r[col]} out={got[col]}')
                n_bad += 1
    if len(mine) != len(ref):
        print(f'  ROW-COUNT: parsed {len(mine)} STDY rows, reference has {len(ref)}')
        n_bad += 1
    return n_bad == 0


def main():
    df = load_all()

    print('Cross-checking freshly parsed STDY against mimic_stdy_effects.csv ...')
    if not crosscheck_stdy(df):
        sys.exit('FAILED: STDY cross-check mismatches above — not writing CSV.')
    print(f'  OK: all STDY rows match the committed CSV.\n')

    pick = df[((~df['predictor'].isin(BINARY_PREDS))
               & (df['standardization'] == 'STDYX'))
              | ((df['predictor'].isin(BINARY_PREDS))
                 & (df['standardization'] == 'STDY'))].copy()

    csv_path = os.path.join(HERE, 'mimic_std_effects.csv')
    col_order = ['subfolder', 'predictor', 'label', 'model', 'factor',
                 'standardization', 'estimate', 'se', 'z_ratio', 'p_value', 'sig']
    pick[col_order].to_csv(csv_path, index=False)
    print(f'Saved: {csv_path}  ({len(pick)} rows)\n')

    # Side-by-side old (STDY) vs new value for every continuous predictor
    wide = df.pivot_table(index=['subfolder', 'predictor', 'model', 'factor'],
                          columns='standardization',
                          values=['estimate', 'p_value']).round(3)
    wide.columns = [f'{a}_{b}' for a, b in wide.columns]
    wide = wide.reset_index()
    cont = wide[~wide['predictor'].isin(BINARY_PREDS)]
    pd.set_option('display.max_rows', 200)
    pd.set_option('display.width', 140)
    print('Continuous covariates — STDY (old, in text) vs STDYX (new):')
    print(cont[['subfolder', 'predictor', 'model', 'factor', 'estimate_STDY',
                'p_value_STDY', 'estimate_STDYX', 'p_value_STDYX']]
          .to_string(index=False))


if __name__ == '__main__':
    main()
