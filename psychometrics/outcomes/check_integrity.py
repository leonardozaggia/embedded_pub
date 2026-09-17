"""
_check_integrity.py  —  Verify outcomes_associations.csv matches current .out files.

For each row in the CSV:
  1. Find the corresponding .out file in the outcome subfolder
  2. Parse STDY WITH section
  3. Compare estimate, se, z_ratio, p_value (tolerance 0.002)
  4. Check outcome variable distribution for wrong-variable errors

Run from repo root:
  python psychometrics/outcomes/check_integrity.py
"""
import os, re, glob
import pandas as pd

HERE     = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(HERE, 'outcomes_associations.csv')

OUTCOMES = {
    'BayLan':   'BAYLAN',
    'BayMot':   'BAYMOT',
    'CBAgg':    'CBAGG',
    'CBAnx':    'CBANX',
    'CBAtt':    'CBATT',
    'CBExt':    'CBEXT',
    'CBInt':    'CBINT',
    'CBSlp':    'CBSLP',
    'CBSom':    'CBSOM',
    'CBTot':    'CBTOT',
    'CBWth':    'CBWTH',
    'ECEff':    'ECEFF',
    'ECNeg':    'ECNEG',
    'ECSur':    'ECSUR',
    'QchatTot': 'QCHATTOT',
}

FACTORS_3F     = {'WM', 'GDP', 'FS'}
FACTORS_UNIDIM = {'COG'}
TOLERANCE = 0.002


def parse_stdy_with(out_path, outcome_var_upper, valid_factors):
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
                    if current_block_var in valid_factors and var_name == outcome_var_upper:
                        results.append({'factor': current_block_var,
                                        'estimate': est, 'se': se,
                                        'z_ratio': z, 'p_value': p})
                    elif current_block_var == outcome_var_upper and var_name in valid_factors:
                        results.append({'factor': var_name,
                                        'estimate': est, 'se': se,
                                        'z_ratio': z, 'p_value': p})
                except ValueError:
                    pass

    return results


def get_var_distribution(out_path, var_upper):
    """Parse sample statistics table for variable distribution."""
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

    for i in range(section_start + 2, min(section_start + 200, len(lines))):
        s = lines[i].strip()
        parts = s.split()
        if not parts:
            continue
        if parts[0].upper() == var_upper and len(parts) >= 5:
            try:
                mean_val = float(parts[1])
                min_val  = float(parts[3])
                min_pct  = float(parts[4].replace('%', ''))
                if i + 1 < len(lines):
                    parts2 = lines[i + 1].strip().split()
                    if len(parts2) >= 5:
                        max_val = float(parts2[3])
                        max_pct = float(parts2[4].replace('%', ''))
                        return {
                            'mean': mean_val, 'min_val': min_val,
                            'max_val': max_val, 'min_pct': min_pct,
                            'max_pct': max_pct,
                        }
            except (ValueError, IndexError):
                pass
    return None


def find_out_file(subfolder, model):
    folder = os.path.join(HERE, subfolder)
    if not os.path.isdir(folder):
        return None
    all_out = glob.glob(os.path.join(folder, '*.out'))
    if model == 'UNIDIM':
        candidates = [f for f in all_out if '_unidim' in os.path.basename(f).lower()]
    else:
        candidates = [f for f in all_out if '_unidim' not in os.path.basename(f).lower()]
    return candidates[0] if candidates else None


def main():
    df = pd.read_csv(CSV_PATH)
    ok_count   = 0
    mismatches = []
    wrong_var  = []
    missing_out = []
    dist_cache = {}     # (out_path, var) -> distribution

    for _, row in df.iterrows():
        sf      = row['subfolder']
        outcome = row['outcome']
        fac     = row['factor']
        model   = row['model']

        if sf not in OUTCOMES:
            continue

        out_file = find_out_file(sf, model)
        if out_file is None:
            missing_out.append(f'{sf}/{model}')
            continue

        valid_factors = FACTORS_UNIDIM if model == 'UNIDIM' else FACTORS_3F
        records = parse_stdy_with(out_file, outcome, valid_factors)
        matched = [r for r in records if r['factor'] == fac]

        if not matched:
            mismatches.append({
                'sf': sf, 'outcome': outcome, 'fac': fac, 'model': model,
                'issue': 'NOT FOUND in .out STDY WITH section',
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
                'sf': sf, 'outcome': outcome, 'fac': fac, 'model': model,
                'issue': 'VALUE MISMATCH',
                'csv': (f"est={row['estimate']:.3f} se={row['se']:.3f} "
                        f"z={row['z_ratio']:.3f} p={row['p_value']:.3f}"),
                'out': (f"est={rec['estimate']:.3f} se={rec['se']:.3f} "
                        f"z={rec['z_ratio']:.3f} p={rec['p_value']:.3f}"),
            })

        # Distribution check for outcome variable (once per .out file)
        key = (out_file, outcome)
        if key not in dist_cache:
            dist_cache[key] = get_var_distribution(out_file, outcome)
        d = dist_cache.get(key)
        if d is not None:
            # Outcomes should be standardized (mean ~ 0, range typically -4 to 4)
            # Flag if distribution looks binary or clearly wrong
            looks_binary_out = (abs(d['min_val']) < 0.01 and abs(d['max_val'] - 1.0) < 0.01
                                 and d['min_pct'] + d['max_pct'] > 90)
            if looks_binary_out:
                if key not in [w['key'] for w in wrong_var]:
                    wrong_var.append({
                        'sf': sf, 'outcome': outcome, 'model': model,
                        'file': os.path.basename(out_file),
                        'dist': (f"min={d['min_val']}, max={d['max_val']}, "
                                 f"{d['min_pct']}%/{d['max_pct']}% [binary for continuous outcome!]"),
                        'key': key,
                    })

    # Check 3F vs UNIDIM distribution consistency per subfolder
    for sf, outcome_var in OUTCOMES.items():
        f3_out = find_out_file(sf, '3F')
        ud_out = find_out_file(sf, 'UNIDIM')
        if not f3_out or not ud_out:
            continue
        d3 = get_var_distribution(f3_out, outcome_var)
        du = get_var_distribution(ud_out, outcome_var)
        if d3 and du and abs(d3['mean'] - du['mean']) >= 0.01:
            wrong_var.append({
                'sf': sf, 'outcome': outcome_var, 'model': '3F vs UNIDIM',
                'file': f'{os.path.basename(f3_out)} vs {os.path.basename(ud_out)}',
                'dist': (f"3F mean={d3['mean']:.3f} range [{d3['min_val']},{d3['max_val']}] | "
                         f"UNIDIM mean={du['mean']:.3f} range [{du['min_val']},{du['max_val']}]"
                         f" [DISTRIBUTION MISMATCH]"),
                'key': (f3_out, ud_out),
            })

    # ── Report ──────────────────────────────────────────────────────────────────
    W = 72
    print('=' * W)
    print('OUTCOMES INTEGRITY CHECK')
    print(f'CSV: {CSV_PATH}')
    print('=' * W)

    print(f'\n[OK]  OK rows : {ok_count}')

    if mismatches:
        print(f'\n[!!] VALUE MISMATCHES ({len(mismatches)}):')
        for m in mismatches:
            print(f"  {m['sf']:10s} | {m['outcome']:10s} | {m['fac']:4s} | {m['model']:6s}")
            print(f"    Issue : {m['issue']}")
            print(f"    CSV   : {m['csv']}")
            print(f"    .out  : {m['out']}")
    else:
        print('\n[!!] VALUE MISMATCHES : none')

    if wrong_var:
        print(f'\n[WV] WRONG-VARIABLE / DISTRIBUTION ISSUES ({len(wrong_var)}):')
        for w in wrong_var:
            print(f"  {w['sf']:10s} | {w['outcome']:10s} | {w['model']:6s}  [{w['file']}]")
            print(f"    {w['dist']}")
    else:
        print('\n[WV] WRONG-VARIABLE / DISTRIBUTION ISSUES : none')

    if missing_out:
        print(f'\n[--] MISSING .out FILES: {missing_out}')
    else:
        print('\n[--] MISSING .out FILES : none')

    print('=' * W)

    return mismatches, wrong_var


if __name__ == '__main__':
    main()
