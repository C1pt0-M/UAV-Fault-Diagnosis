"""Descriptive: late-hover altitude (baro, relative to case-grid start) by class/domain, train hover cases."""
import numpy as np, pandas as pd
D = 'D:/UAV-Data/causal_input_v1/'
L = pd.read_csv(D + 'ledgers/train_included_windows.csv', dtype={'association_group': str}, keep_default_na=False, low_memory=False)
uids = L[L.case_uid.str.split('/').str[1] == 'hover'].case_uid.unique()
rows = []
for u in uids:
    g = np.load(D + 'cases/' + u.replace('/', '__') + '.npz')
    b = g['canonical_rows'][:, 14]; n = len(b)
    if n < 1600: continue
    rows.append({'domain': u.split('/')[0], 'cls': u.split('/')[2], 'late_rel_alt_m': float(np.nanmean(b[1500:]) - np.nanmean(b[:50])),
                 'start_to_late_s': n * 0.02})
df = pd.DataFrame(rows)
t = df.groupby(['cls', 'domain']).late_rel_alt_m.describe()[['count', '25%', '50%', '75%']].round(1)
print(t.to_string())
df.to_csv('D:/UAV-Fault-Diagnosis/artifacts/q2_late_hover_gps_observability_v1/hover_altitude_census.csv', index=False)
