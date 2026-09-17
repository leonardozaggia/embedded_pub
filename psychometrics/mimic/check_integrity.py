"""
_check_integrity.py  —  Verify mimic_stdy_effects.csv matches current .out files.

For each row in the CSV:
  1. Find the corresponding .out file in the subfolder
  2. Parse STDY ON section
  3. Compare estimate, se, z_ratio, p_value (tolerance 0.002)
  4. Check predictor variable distribution for wrong-variable-type errors

Run from repo root:
  python psychometrics/mimic/check_integrity.py
"""
import os, re, glob
import pandas as pd

HERE     = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(HERE, 'mimic_stdy_effects.csv')

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
    'PrLax':   ['PRLAX'],
    'PrOvr':   ['PROVR'],
    'PrVrb':   ['PRVRB'],
    'PrSty':   ['PRSTY'],
}

CONTINUOUS_PREDS = {
    'GA', 'BIRTHWT', 'HDCIRC', 'MOMAGE', 'MOMEDU', 'DADEDU',
    'EPDSTOT', 'PRLAX', 'PROVR', 'PRVRB', 'PRSTY',
}
BINARY_PREDS = {'SEX', 'IUGR', 'GDIAB', 'PRECL', 'MOMENG', 'ANTCOR_P', 'ANTCOR_C'}

TOLERANCE = 0.002


def parse_stdy_on(out_path):
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
        if 'Standardization' in ln or 'R-SQUARE' in ln or 'CONFIDENCE INTERVALS' in ln:
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
            factor = m.group(1).upper()
            in_on = True
            continue
        if (re.match(r'^[A-Z][A-Z0-9_]*\s+(BY|WITH)\s*$', s, re.I)
                or re.match(r'^(Thresholds|Intercepts|Means|Variances|Residual)\b', s, re.I)):
            in_on = False
            factor = None
            continue
        if in_on and factor:
            parts = s.split()
            if len(parts) == 5:
                try:
                    est, se, z, p = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
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


def get_predictor_distribution(out_path, predictor_upper):
    """Parse the univariate sample statistics table and return distribution info."""
    try:
        with open(out_path, encoding='utf-8', errors='replace') as fh:
            lines = fh.readlines()
    except Exception:
        return None

    section_start = None
    for i in range(len(lines) - 1):
        if 'Variable/' in lines[i] and 'Sample Size' in lines[i + 1]:
            section_start = i
            break
    if section_start is None:
        return None

    for i in range(section_start + 2, min(section_start + 120, len(lines))):
        s = lines[i].strip()
        parts = s.split()
        if not parts:
            continue
        if parts[0].upper() == predictor_upper and len(parts) >= 5:
            try:
                min_val = float(parts[3])
                min_pct = float(parts[4].replace('%', ''))
                if i + 1 < len(lines):
                    parts2 = lines[i + 1].strip().split()
                    if len(parts2) >= 5:
                        max_val = float(parts2[3])
                        max_pct = float(parts2[4].replace('%', ''))
                        return {
                            'mean':    float(parts[1]),
                            'min_val': min_val,
                            'max_val': max_val,
                            'min_pct': min_pct,
                            'max_pct': max_pct,
                        }
            except (ValueError, IndexError):
                pass
    return None


def looks_binary(dist):
    if dist is None:
        return False
    return (abs(dist['min_val']) < 0.01
            and abs(dist['max_val'] - 1.0) < 0.01
            and dist['min_pct'] + dist['max_pct'] > 90)


def find_out_file(subfolder, model):
    folder = os.path.join(HERE, subfolder)
    if not os.path.isdir(folder):
        return None
    all_out = glob.glob(os.path.join(folder, '*.out'))
    if model == 'UNIDIM':
        candidates = [f for f in all_out
                      if '_unidim' in os.path.basename(f).lower()
                      and 'constraint' not in os.path.basename(f).lower()]
    else:
        candidates = [f for f in all_out
                      if '_unidim' not in os.path.basename(f).lower()
                      and 'constraint' not in os.path.basename(f).lower()]
    return candidates[0] if candidates else None


def main():
    df = pd.read_csv(CSV_PATH, encoding='cp1252')
    ok_count = 0
    mismatches = []
    wrong_var   = []
    missing_out = []
    checked_out = {}     # out_path -> list of predictor names already distribution-checked

    for _, row in df.iterrows():
        sf    = row['subfolder']
        pred  = row['predictor']
        fac   = row['factor']
        model = row['model']

        if sf not in SUBFOLDER_PREDS:
            continue  # Skip archived predictors (Eth)

        out_file = find_out_file(sf, model)
        if out_file is None:
            missing_out.append(f'{sf}/{model}')
            continue

        records = parse_stdy_on(out_file)
        matched = [r for r in records if r['predictor'] == pred and r['factor'] == fac]

        if not matched:
            mismatches.append({
                'sf': sf, 'pred': pred, 'fac': fac, 'model': model,
                'issue': 'NOT FOUND in .out STDY ON section',
                'csv': f"est={row['estimate']}, z={row['z_ratio']}",
                'out': 'N/A',
            })
            continue

        rec = matched[0]
        est_ok = abs(rec['estimate'] - row['estimate']) <= TOLERANCE
        se_ok  = abs(rec['se']       - row['se'])       <= TOLERANCE
        z_ok   = abs(rec['z_ratio']  - row['z_ratio'])  <= TOLERANCE
        p_ok   = abs(rec['p_value']  - row['p_value'])  <= TOLERANCE

        if est_ok and se_ok and z_ok and p_ok:
            ok_count += 1
        else:
            mismatches.append({
                'sf': sf, 'pred': pred, 'fac': fac, 'model': model,
                'issue': 'VALUE MISMATCH',
                'csv': f"est={row['estimate']:.3f} se={row['se']:.3f} z={row['z_ratio']:.3f} p={row['p_value']:.3f}",
                'out': f"est={rec['estimate']:.3f} se={rec['se']:.3f} z={rec['z_ratio']:.3f} p={rec['p_value']:.3f}",
            })

        # Distribution check — once per unique (out_file, predictor)
        key = (out_file, pred)
        if key not in checked_out:
            checked_out[key] = True
            dist = get_predictor_distribution(out_file, pred)
            if pred in CONTINUOUS_PREDS and looks_binary(dist):
                wrong_var.append({
                    'sf': sf, 'pred': pred, 'model': model,
                    'file': os.path.basename(out_file),
                    'dist': (f"min={dist['min_val']}, max={dist['max_val']}, "
                             f"{dist['min_pct']}%/{dist['max_pct']}% [binary for continuous pred!]"),
                })
            elif pred in BINARY_PREDS and dist is not None and not looks_binary(dist):
                wrong_var.append({
                    'sf': sf, 'pred': pred, 'model': model,
                    'file': os.path.basename(out_file),
                    'dist': (f"min={dist['min_val']}, max={dist['max_val']}"
                             f" [not binary for a binary predictor?]"),
                })

    # ── Report ──────────────────────────────────────────────────────────────────
    W = 72
    print('=' * W)
    print('DEMOGRAPHICS INTEGRITY CHECK')
    print(f'CSV: {CSV_PATH}')
    print('=' * W)

    print(f'\n[OK]  OK rows : {ok_count}')

    if mismatches:
        print(f'\n[!!] VALUE MISMATCHES ({len(mismatches)}):')
        for m in mismatches:
            print(f"  {m['sf']:10s} | {m['pred']:10s} | {m['fac']:4s} | {m['model']:6s}")
            print(f"    Issue : {m['issue']}")
            print(f"    CSV   : {m['csv']}")
            print(f"    .out  : {m['out']}")
    else:
        print('\n[!!] VALUE MISMATCHES : none')

    if wrong_var:
        print(f'\n[WV] WRONG-VARIABLE DISTRIBUTIONS ({len(wrong_var)}):')
        for w in wrong_var:
            print(f"  {w['sf']:10s} | {w['pred']:10s} | {w['model']:6s}  [{w['file']}]")
            print(f"    {w['dist']}")
    else:
        print('\n[WV] WRONG-VARIABLE DISTRIBUTIONS : none')

    if missing_out:
        print(f'\n[--] MISSING .out FILES: {missing_out}')
    else:
        print('\n[--] MISSING .out FILES : none')

    print('=' * W)

    return mismatches, wrong_var


if __name__ == '__main__':
    main()
