"""Q2 Part B: does the fault exist in the raw GPS sensor during late hover? (outside the 26-ch input)

Protocol: configs/experiments/q2_late_hover_gps_observability_v1.json. Train cases only.
Reads raw RflyMAD ULog vehicle_gps_position and the causal_input_v1 case grid (baro, timestamps).
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
from pyulog import ULog
from sklearn.metrics import roc_auc_score

DATA = Path('D:/UAV-Data/causal_input_v1')
RAW = Path('D:/RflyMAD')
OUT = Path('D:/UAV-Fault-Diagnosis/artifacts/q2_late_hover_gps_observability_v1')
M_PER_DEG = 111_320.0


def case_features(uid):
    grid = np.load(DATA / 'cases' / (uid.replace('/', '__') + '.npz'))
    ts = grid['timestamp_us'].astype(np.int64)
    baro = grid['canonical_rows'][:, 14]
    ulg = glob.glob(str(RAW / uid / 'Log' / '*.ulg'))[0]
    g = ULog(ulg, message_name_filter_list=['vehicle_gps_position']).get_dataset('vehicle_gps_position').data
    gt = g['timestamp'].astype(np.int64)
    lat, lon, alt = g['lat'] * 1e-7, g['lon'] * 1e-7, g['alt'] * 1e-3
    ref = slice(0, 50)                       # first second of the case grid
    late = np.arange(len(ts)) * 0.02 > 30    # late hover, same definition as the windows
    if late.sum() < 50:
        return None
    gi = np.clip(np.searchsorted(gt, ts, side='right') - 1, 0, len(gt) - 1)   # last GPS sample at or before grid time
    north = (lat[gi] - lat[gi][ref].mean()) * M_PER_DEG
    east = (lon[gi] - lon[gi][ref].mean()) * M_PER_DEG * np.cos(np.radians(lat[gi][ref].mean()))
    dalt = alt[gi] - alt[gi][ref].mean()
    dbaro = baro - np.nanmean(baro[ref])
    ok = late & np.isfinite(dbaro)
    return {'gps_minus_baro_alt_m': float(np.mean(dalt[ok] - dbaro[ok])),
            'gps_over_baro_ratio': float(np.mean(dalt[ok]) / np.mean(dbaro[ok])),
            'horiz_std_m': float(np.sqrt(north[ok].var() + east[ok].var())),
            'horiz_offset_m': float(np.hypot(north[ok].mean(), east[ok].mean())),
            'baro_rel_alt_m': float(np.mean(dbaro[ok]))}


def main():
    L = pd.read_csv(DATA / 'ledgers/train_included_windows.csv', dtype={'association_group': str},
                    keep_default_na=False, low_memory=False)
    p = L.case_uid.str.split('/')
    late = (p.str[1] == 'hover') & (L.window_end_row * 0.02 > 30)
    pos = L[late & (p.str[2] == 'GPS') & (L.qualified_y == 1)].case_uid.unique()
    neg = L[late & (L.qualified_y == 0) & (L.normal_origin == 'normal_flight')].case_uid.unique()
    modes = pd.read_csv(OUT.parent / 'q2_gps_fault_modes.csv').set_index('case_uid')
    rows = []
    for uid in list(pos) + list(neg):
        f = case_features(uid)
        if f is None:
            continue
        if uid in modes.index:
            md, pr = int(modes.loc[uid, 'mode']), modes.loc[uid, 'params']
            st = 'mode3' if md == 3 else ('mode4_z_eq_1' if pr.split()[2] == '1.' else 'mode4_z_ne_1')
        else:
            md, pr, st = 0, '', 'normal'
        rows.append({'case_uid': uid, 'stratum': st, 'params': pr, 'identity': pr == '1. 1. 1.', **f})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / 'raw_gps_case_features.csv', index=False)
    dev = {'abs_gps_minus_baro_alt_m': df.gps_minus_baro_alt_m.abs(), 'abs_ratio_minus_1': (df.gps_over_baro_ratio - 1).abs(),
           'horiz_std_m': df.horiz_std_m, 'horiz_offset_m': df.horiz_offset_m}
    res = {'cases': df.stratum.value_counts().to_dict(), 'strata': {}}
    groups = {'all_gps': df.stratum != 'normal', 'mode3': df.stratum == 'mode3', 'mode4_z_ne_1': df.stratum == 'mode4_z_ne_1',
              'mode4_z_eq_1': df.stratum == 'mode4_z_eq_1', 'mode4_identity': df.identity}
    nm = df.stratum == 'normal'
    for st, sel in groups.items():
        mk = sel | nm
        y = sel[mk].astype(int)
        r = {k: float(roc_auc_score(y, v[mk])) for k, v in dev.items()}
        r['best'] = max(r, key=r.get)
        r['median'] = {k: float(v[sel].median()) for k, v in dev.items()}
        res['strata'][st] = r
    res['normal_median'] = {k: float(v[nm].median()) for k, v in dev.items()}
    res['normal_p95'] = {k: float(v[nm].quantile(0.95)) for k, v in dev.items()}
    (OUT / 'raw_gps_reference.json').write_text(json.dumps(res, indent=1), encoding='utf-8')
    print(json.dumps(res, indent=1))
    print(df.groupby('stratum')[['gps_minus_baro_alt_m', 'gps_over_baro_ratio', 'horiz_std_m', 'horiz_offset_m', 'baro_rel_alt_m']].median().round(3))


if __name__ == '__main__':
    main()
