# =============================================================================
# extract_outcomes_associations.py
# =============================================================================
# Purpose:
#   Parses Mplus .out files from each outcome subdirectory and extracts the
#   STDY-standardised factor–outcome correlations (WITH statements).
#   Saves results to psychometrics/outcomes/outcomes_associations.csv.
#
# Usage (from repo root):
#   python psychometrics/outcomes/extract_outcomes_associations.py
# =============================================================================

import os, re, glob
import pandas as pd

HERE    = os.path.dirname(os.path.abspath(__file__))
OUT_CSV = os.path.join(HERE, 'outcomes_associations.csv')

OUTCOMES = {
    'BayLan': {'var': 'BAYLAN', 'label': 'Bayley Language'},
    'BayMot': {'var': 'BAYMOT', 'label': 'Bayley Motor'},
    'CBAgg':  {'var': 'CBAGG',  'label': 'CBCL Aggressive Behaviour'},
    'CBAnx':  {'var': 'CBANX',  'label': 'CBCL Anxious/Depressed'},
    'CBAtt':  {'var': 'CBATT',  'label': 'CBCL Attention Problems'},
    'CBEmR':  {'var': 'CBEMR',  'label': 'CBCL Emotionally Reactive'},
    'CBExt':  {'var': 'CBEXT',  'label': 'CBCL Externalising'},
    'CBInt':  {'var': 'CBINT',  'label': 'CBCL Internalising'},
    'CBSlp':  {'var': 'CBSLP',  'label': 'CBCL Sleep Problems'},
    'CBSom':  {'var': 'CBSOM',  'label': 'CBCL Somatic Complaints'},
    'CBTot':  {'var': 'CBTOT',  'label': 'CBCL Total Problems'},
    'CBWth':  {'var': 'CBWTH',  'label': 'CBCL Withdrawn/Depressed'},
    'ECEff':  {'var': 'ECEFF',  'label': 'Effortful Control'},
    'ECNeg':  {'var': 'ECNEG',  'label': 'Negative Affect'},
    'ECSur':  {'var': 'ECSUR',  'label': 'Surgency/Extraversion'},
    'QchatTot': {'var': 'QCHATTOT', 'label': 'Q-CHAT Total'},
}

DOMAIN = {
    'BayLan': 'Bayley',
    'BayMot': 'Bayley',
    'CBInt':  'CBCL Broad',
    'CBEmR':  'CBCL Narrow',
    'CBExt':  'CBCL Broad',
    'CBTot':  'CBCL Broad',
    'CBAnx':  'CBCL Narrow',
    'CBWth':  'CBCL Narrow',
    'CBSom':  'CBCL Narrow',
    'CBSlp':  'CBCL Narrow',
    'CBAtt':  'CBCL Narrow',
    'CBAgg':  'CBCL Narrow',
    'ECSur':  'Emotion Control',
    'ECNeg':  'Emotion Control',
    'ECEff':  'Emotion Control',
    'QchatTot': 'Q-CHAT',
}

FACTORS_3F     = {'WM', 'GDP', 'FS'}
FACTORS_UNIDIM = {'COG'}


def sig_stars(pv):
    if pv < 0.001: return '***'
    if pv < 0.010: return '**'
    if pv < 0.050: return '*'
    if pv < 0.100: return '\u2020'
    return ''


def parse_stdy_with(out_path, outcome_var_upper, valid_factors):
    """Extract STDY WITH rows linking the outcome to any of valid_factors."""
    try:
        with open(out_path, encoding='utf-8', errors='replace') as fh:
            lines = fh.readlines()
    except Exception:
        return []

    # First STDY Standardization block (not the CI block)
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
        if 'R-SQUARE' in ln or 'CONFIDENCE INTERVALS' in ln:
            end = i
            break

    results = []
    current_block_var = None
    in_with = False

    for ln in lines[start:end]:
        s = ln.strip()
        if not s:
            in_with = False
            current_block_var = None
            continue

        m_with = re.match(r'^([A-Z][A-Z0-9_]*)\s+WITH\s*$', s, re.I)
        if m_with:
            current_block_var = m_with.group(1).upper()
            in_with = True
            continue

        if (re.match(r'^[A-Z][A-Z0-9_]*\s+(BY|ON)\s*$', s, re.I)
                or re.match(r'^(Thresholds|Intercepts|Means|Variances|Residual)\b', s, re.I)):
            in_with = False
            current_block_var = None
            continue

        if in_with and current_block_var:
            parts = s.split()
            if len(parts) == 5:
                try:
                    var_name = parts[0].upper()
                    est, se, z, p = (float(parts[1]), float(parts[2]),
                                     float(parts[3]), float(parts[4]))
                    if abs(z) >= 900:
                        continue
                    # FACTOR WITH ... OUTCOME
                    if current_block_var in valid_factors and var_name == outcome_var_upper:
                        results.append({'factor': current_block_var,
                                        'estimate': est, 'se': se,
                                        'z_ratio': z, 'p_value': p})
                    # OUTCOME WITH ... FACTOR
                    elif current_block_var == outcome_var_upper and var_name in valid_factors:
                        results.append({'factor': var_name,
                                        'estimate': est, 'se': se,
                                        'z_ratio': z, 'p_value': p})
                except ValueError:
                    pass

    return results


def load_all():
    rows = []
    for sf, meta in OUTCOMES.items():
        folder = os.path.join(HERE, sf)
        if not os.path.isdir(folder):
            print(f'  WARNING: folder not found: {sf}')
            continue

        all_out = glob.glob(os.path.join(folder, '*.out'))
        f3_out  = [f for f in all_out if '_unidim' not in os.path.basename(f).lower()]
        ud_out  = [f for f in all_out if '_unidim' in os.path.basename(f).lower()]

        for paths, model, valid_factors in [
                (f3_out, '3F',     FACTORS_3F),
                (ud_out, 'UNIDIM', FACTORS_UNIDIM)]:
            for path in paths:
                recs = parse_stdy_with(path, meta['var'], valid_factors)
                if not recs:
                    print(f'  WARNING: no WITH rows found in {os.path.basename(path)}')
                for rec in recs:
                    rec.update({
                        'subfolder': sf,
                        'outcome':   meta['var'],
                        'label':     meta['label'],
                        'domain':    DOMAIN[sf],
                        'model':     model,
                        'sig':       sig_stars(rec['p_value']),
                    })
                    rows.append(rec)

    df = pd.DataFrame(rows).drop_duplicates(subset=['subfolder', 'model', 'factor'])
    return df.sort_values(['domain', 'subfolder', 'model', 'factor']).reset_index(drop=True)


if __name__ == '__main__':
    print('Extracting factor–outcome correlations from STDY WITH sections...\n')
    df = load_all()

    col_order = ['subfolder', 'outcome', 'label', 'domain', 'model', 'factor',
                 'estimate', 'se', 'z_ratio', 'p_value', 'sig']
    df[col_order].to_csv(OUT_CSV, index=False)
    print(f'\nSaved: {OUT_CSV}  ({len(df)} rows)\n')

    pd.set_option('display.max_rows', 200)
    pd.set_option('display.width', 120)
    summary = df[['subfolder', 'label', 'model', 'factor',
                  'estimate', 'z_ratio', 'p_value', 'sig']].copy()
    summary['estimate'] = summary['estimate'].round(3)
    summary['z_ratio']  = summary['z_ratio'].round(3)
    summary['p_value']  = summary['p_value'].round(3)
    print(summary.to_string(index=False))
