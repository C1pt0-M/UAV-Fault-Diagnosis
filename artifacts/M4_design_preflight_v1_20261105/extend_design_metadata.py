"""Metadata-only nested split draft and reproducible same-case comparison.

Reads new Train-only summaries, never source tensors or a model. Does not
overwrite existing outputs. The existing pair table is reconciled numerically.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent


def save_json(name, value):
    with (OUT / name).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def population(frame):
    result = {'windows': len(frame), 'cases': frame.case_uid.nunique(),
              'groups': frame.association_group.nunique()}
    for name, mask in [('normal', frame.known_normal),
                       ('fault_endpoint', frame.qualified_y.eq(1)),
                       ('pure_fault', frame.pure_fault)]:
        part = frame[mask]
        result.update({name + '_windows': len(part),
                       name + '_cases': part.case_uid.nunique(),
                       name + '_groups': part.association_group.nunique()})
    return result


def main():
    plan = json.loads((OUT / 'nested_design_plan.json').read_text(encoding='utf-8'))
    f = pd.read_csv(OUT / 'window_state_ledger.csv.gz',
                    dtype={'association_group': str}, keep_default_na=False)
    groups = pd.read_csv(OUT / plan['outer_design'], dtype={'association_group': str})
    assignments, populations, normal_support, fault_support, singleton_strata = [], [], [], [], []
    for outer in range(3):
        assignment = groups.copy()
        assignment['outer_evaluation_fold'] = outer
        assignment['role'] = np.where(assignment.outer_fold.eq(outer), 'outer_held_out', 'inner_fit')
        for key, part in assignment[assignment.role.eq('inner_fit')].groupby(['fault_directory', 'flight_directory']):
            n = len(part)
            if n == 1:
                singleton_strata.append({'outer': outer, 'fault_directory': key[0], 'flight_directory': key[1]})
                continue
            count = max(1, min(n - 1, int(np.floor(.2 * n + .5))))
            ordered = sorted(part.index, key=lambda ix: hashlib.sha256(
                f"{plan['inner_seed']}|{outer}|{assignment.loc[ix, 'association_group']}".encode()).hexdigest())
            assignment.loc[ordered[:count], 'role'] = 'inner_dev'
        assignments.append(assignment)
        role_map = assignment.set_index('association_group').role
        roles = f.association_group.map(role_map)
        assert roles.notna().all()
        for role in ['inner_fit', 'inner_dev', 'outer_held_out']:
            part = f[roles.eq(role)]
            populations.append({'outer_fold': outer, 'role': role, **population(part)})
            for state in sorted(f.state.unique()):
                state_part = part[part.state.eq(state) & part.known_normal]
                normal_support.append({'outer_fold': outer, 'role': role, 'state': state,
                                       'normal_windows': len(state_part), 'normal_cases': state_part.case_uid.nunique(),
                                       'normal_groups': state_part.association_group.nunique(),
                                       'SIL_cases': state_part[state_part.domain.eq('SIL')].case_uid.nunique(),
                                       'HIL_cases': state_part[state_part.domain.eq('HIL')].case_uid.nunique()})
            for fault in ['GPS', 'accelerometer', 'barometer', 'gyroscope', 'magnetometer', 'motor']:
                for domain in ['SIL', 'HIL']:
                    fault_part = part[part.fault_directory.eq(fault) & part.domain.eq(domain) & part.qualified_y.eq(1)]
                    fault_support.append({'outer_fold': outer, 'role': role, 'fault_directory': fault,
                                          'domain': domain, 'fault_windows': len(fault_part),
                                          'fault_cases': fault_part.case_uid.nunique(),
                                          'fault_groups': fault_part.association_group.nunique()})
        # Case/group isolation holds at every level, not just outer groups.
        assert assignment.association_group.is_unique
        assert pd.DataFrame({'case': f.case_uid, 'role': roles}).groupby('case').role.nunique().eq(1).all()
    for name, value in [('nested_group_assignments_3.csv', pd.concat(assignments)),
                        ('nested_populations_3.csv', pd.DataFrame(populations)),
                        ('nested_normal_state_support_3.csv', pd.DataFrame(normal_support)),
                        ('nested_fault_domain_support_3.csv', pd.DataFrame(fault_support))]:
        value.to_csv(OUT / name, index=False, mode='x')

    summaries = pd.read_csv(OUT / 'case_state_feature_summaries.csv.gz',
                             dtype={'association_group': str}, keep_default_na=False)
    normal = summaries[summaries.fault_directory.eq('no_fault') & summaries.qualified_y.eq(0)]
    keys = ['case_uid', 'association_group', 'domain']
    features = ['actuator_ctrl_4__median', 'output_mean_median_native', 'acc_norm__median', 'baro_alt_relative__median']
    climb = normal[normal.state.eq('climb_proxy')][keys + features]
    low = normal[normal.state.eq('low_motion_proxy')][keys + features]
    merged = climb.merge(low, on=keys, suffixes=('_climb', '_low'), validate='one_to_one')
    pairs = merged[keys].copy()
    for feature in features:
        pairs[feature + '_low_minus_climb'] = merged[feature + '_low'] - merged[feature + '_climb']
    target = OUT / 'independent_normal_climb_low_case_pairs.csv'
    if target.exists():
        saved = pd.read_csv(target, dtype={'association_group': str}, keep_default_na=False)
        pd.testing.assert_frame_equal(pairs.sort_values(keys).reset_index(drop=True),
                                      saved.sort_values(keys).reset_index(drop=True),
                                      check_exact=False, atol=1e-12, rtol=1e-12)
    else:
        pairs.to_csv(target, index=False, mode='x')
    normal_table = pd.DataFrame(normal_support)
    train_table = normal_table[normal_table.role.eq('inner_fit')]
    train_table = train_table.assign(passed=(train_table.normal_windows.ge(1000) &
        train_table.normal_cases.ge(30) & train_table.normal_groups.ge(15) &
        train_table.SIL_cases.ge(10) & train_table.HIL_cases.ge(10)))
    fault_table = pd.DataFrame(fault_support)
    dev = fault_table[fault_table.role.eq('inner_dev')]
    save_json('nested_design_verification.json', {
        'status': 'PASS_GROUP_ISOLATION_DESIGN_ONLY', 'case_or_group_crossings': 0,
        'singleton_inner_strata': singleton_strata,
        'inner_fit_normal_support': train_table.groupby('state').passed.all().to_dict(),
        'inner_dev_min_fault_cases_per_fault_domain': int(dev.fault_cases.min()),
        'inner_dev_min_fault_windows_per_fault_domain': int(dev.fault_windows.min()),
        'same_case_pairs_reconciled': len(pairs), 'paired_groups': pairs.association_group.nunique(),
        'no_source_tensor_or_model_read': True})
    print(json.dumps({'nested_populations': populations,
                      'inner_fit_support': train_table.groupby('state').passed.all().to_dict(),
                      'inner_dev_min_fault_cases': int(dev.fault_cases.min())}, ensure_ascii=False))


if __name__ == '__main__':
    main()
