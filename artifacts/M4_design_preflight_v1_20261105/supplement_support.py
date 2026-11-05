"""Supplement existing Train-only ledger with independent-normal/hover support.

No model imports or source tensor reads. Exclusive-create outputs.
"""
from pathlib import Path
import pandas as pd

OUT = Path(__file__).resolve().parent


def main():
    f = pd.read_csv(OUT / 'window_state_ledger.csv.gz', dtype={'association_group': str}, keep_default_na=False)
    rows = []
    for (state, domain, flight), part in f[f.known_normal & f.fault_directory.eq('no_fault')].groupby(['state', 'domain', 'flight_directory']):
        rows.append({'state': state, 'domain': domain, 'flight_directory': flight,
                     'windows': len(part), 'cases': part.case_uid.nunique(), 'groups': part.association_group.nunique()})
    pd.DataFrame(rows).to_csv(OUT / 'independent_normal_state_domain_flight_support.csv', index=False, mode='x')
    rows = []
    for k in [5, 3]:
        assignments = pd.read_csv(OUT / f'fold_assignments_{k}.csv', dtype={'association_group': str})
        folds = f.association_group.map(assignments.set_index('association_group').outer_fold)
        for fold in range(k):
            heldout = f[folds.eq(fold) & f.flight_directory.eq('hover')]
            for state in ['low_motion_proxy', 'climb_proxy', 'maneuver_proxy', 'translation_proxy', 'other_motion', 'descent_proxy']:
                for domain in ['SIL', 'HIL']:
                    normal = heldout[heldout.state.eq(state) & heldout.domain.eq(domain) & heldout.known_normal & heldout.fault_directory.eq('no_fault')]
                    for fault in ['GPS', 'gyroscope', 'magnetometer', 'accelerometer', 'barometer', 'motor']:
                        fpart = heldout[heldout.state.eq(state) & heldout.domain.eq(domain) & heldout.pure_fault & heldout.fault_directory.eq(fault)]
                        rows.append({'k': k, 'fold': fold, 'state': state, 'domain': domain, 'fault_directory': fault,
                                     'independent_normal_windows': len(normal), 'independent_normal_cases': normal.case_uid.nunique(),
                                     'independent_normal_groups': normal.association_group.nunique(), 'fault_windows': len(fpart),
                                     'fault_cases': fpart.case_uid.nunique(), 'fault_groups': fpart.association_group.nunique(),
                                     'descriptive_support_pass': len(normal) >= 20 and normal.case_uid.nunique() >= 4 and normal.association_group.nunique() >= 2
                                     and len(fpart) >= 20 and fpart.case_uid.nunique() >= 4 and fpart.association_group.nunique() >= 2})
    pd.DataFrame(rows).to_csv(OUT / 'fold_hover_priority_support.csv', index=False, mode='x')


if __name__ == '__main__':
    main()
