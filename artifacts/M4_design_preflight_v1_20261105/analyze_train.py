"""Source-Train-only M4 design statistics. No model import, forward or fitting.

Run with the existing backend environment and -B. Outputs are exclusive-create.
Frozen historical state functions are AST-extracted without importing modules.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('MKL_NUM_THREADS', '4')
import ast
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import time
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
DATA = Path('D:/UAV-Data/causal_input_v1')
FAULTS = ['GPS', 'accelerometer', 'barometer', 'gyroscope', 'magnetometer', 'motor']
STATES = ['maneuver_proxy', 'climb_proxy', 'descent_proxy', 'low_motion_proxy', 'translation_proxy', 'other_motion']


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(name, value):
    with (OUT / name).open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def write_csv(name, frame):
    frame.to_csv(OUT / name, index=False, mode='x', encoding='utf-8', compression='infer')


def extract_function(relative, name):
    source = (ROOT / relative).read_text(encoding='utf-8-sig')
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == name)
    module = ast.Module(body=[node], type_ignores=[])
    namespace = {'np': np}
    exec(compile(module, relative + ':' + name, 'exec'), namespace)
    return namespace[name], hashlib.sha256(ast.get_source_segment(source, node).encode()).hexdigest()


def counts(frame):
    result = {'windows': len(frame), 'cases': frame.case_uid.nunique(), 'groups': frame.association_group.nunique()}
    for label, mask in [('normal_endpoint', frame.qualified_y.eq(0)), ('fault_endpoint', frame.qualified_y.eq(1)),
                        ('known_normal', frame.known_normal), ('pure_fault', frame.pure_fault)]:
        part = frame[mask]
        result.update({label + '_windows': len(part), label + '_cases': part.case_uid.nunique(), label + '_groups': part.association_group.nunique()})
    return result


def grouped_counts(frame, keys):
    rows = []
    for key, part in frame.groupby(keys, sort=True, dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        rows.append(dict(zip(keys, key), **counts(part)))
    return pd.DataFrame(rows)


def normal_support(frame):
    p = frame[frame.known_normal]
    return {'normal_windows': len(p), 'normal_cases': p.case_uid.nunique(), 'normal_groups': p.association_group.nunique(),
            'normal_cases_SIL': p.loc[p.domain.eq('SIL'), 'case_uid'].nunique(),
            'normal_cases_HIL': p.loc[p.domain.eq('HIL'), 'case_uid'].nunique()}


def ref_pass(support, plan):
    m = plan['normal_reference_train_minimum']
    return bool(support['normal_windows'] >= m['windows'] and support['normal_cases'] >= m['cases'] and
                support['normal_groups'] >= m['association_groups'] and
                support['normal_cases_SIL'] >= m['cases_per_domain'] and support['normal_cases_HIL'] >= m['cases_per_domain'])


def descriptive_cell(normal, fault, plan):
    m = plan['heldout_description_minimum']
    return bool(len(normal) >= m['normal_windows'] and len(fault) >= m['fault_windows'] and
                normal.case_uid.nunique() >= m['normal_cases'] and fault.case_uid.nunique() >= m['fault_cases'] and
                normal.association_group.nunique() >= m['normal_association_groups'] and
                fault.association_group.nunique() >= m['fault_association_groups'])


def auc_fixed_direction(negative, positive):
    # One case-state scalar per row; no per-feature direction selection.
    if not len(negative) or not len(positive):
        return None
    z = np.r_[negative, positive]
    ranks = pd.Series(z).rank(method='average').to_numpy()
    n, m = len(negative), len(positive)
    return float((ranks[n:].sum() - m * (m + 1) / 2) / (n * m))


def main():
    started = time.perf_counter()
    if (OUT / 'completion.json').exists() or (OUT / 'window_state_ledger.csv.gz').exists():
        raise FileExistsError('Never rerun over existing output; preserve completed statistics.')
    plan = json.loads((OUT / 'analysis_plan.json').read_text(encoding='utf-8'))
    protocol = json.loads((ROOT / 'configs/experiments/causal_pure_mstca_m0_v1.json').read_text(encoding='utf-8'))
    allowed = plan['input_allowlist']
    identities = {relative: sha(DATA / relative) for relative in allowed}
    for relative in ['train_cache_complete.json', 'input_contract.json', 'norm_stats.json', 'ledgers/train_included_windows.csv', 'cache/train.pt']:
        assert identities[relative] == protocol['data_identities'][relative], relative
    identity = {'scope': 'SOURCE_TRAIN_ONLY', 'inputs': identities, 'plan_sha256': sha(OUT / 'analysis_plan.json'),
                'script_sha256': sha(__file__), 'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT).decode().strip(),
                'environment': {'python': platform.python_version(), 'numpy': np.__version__, 'pandas': pd.__version__, 'torch': str(torch.__version__), 'CUDA_VISIBLE_DEVICES': os.environ['CUDA_VISIBLE_DEVICES']}}
    write_json('input_identity.json', identity)
    print('SOURCE_TRAIN_IDENTITIES_PASS', flush=True)
    frame = pd.read_csv(DATA / 'ledgers/train_included_windows.csv', dtype={'association_group': str, 'case_uid': str}, keep_default_na=False)
    all_windows = pd.read_csv(DATA / 'ledgers/train_all_windows.csv', dtype={'association_group': str, 'case_uid': str}, keep_default_na=False)
    assert len(frame) == 162069 and frame.split.eq('train').all() and frame.role.eq('train').all()
    assert frame.domain.isin(['SIL', 'HIL']).all()
    assert frame.input_valid.all() and frame.label_valid.all() and frame.supervised_valid.all()
    assert np.array_equal(frame.compact_index, np.arange(len(frame))) and frame.original_cache_index.is_unique
    assert frame.qualified_y.isin([0, 1]).all() and (frame.window_end_row - frame.window_start_row).eq(150).all()
    frame['fault_directory'] = frame.case_uid.str.split('/').str[2]
    frame['flight_directory'] = frame.case_uid.str.split('/').str[1]
    frame['known_normal'] = frame.qualified_y.eq(0) & frame.whole_window_label_valid.eq(True) & frame.fault_fraction.eq(0)
    frame['pure_fault'] = frame.qualified_y.eq(1) & frame.whole_window_label_valid.eq(True) & frame.fault_fraction.eq(1)
    frame['state'] = ''
    stats = json.loads((DATA / 'norm_stats.json').read_text())
    payload = torch.load(DATA / 'cache/train.pt', map_location='cpu', mmap=True, weights_only=True)
    x = payload['x']
    assert payload['split'] == 'train' and payload['normalization_applied'] is False
    assert x.dtype == torch.float32 and tuple(x.shape) == (162069, 26, 150)
    assert np.array_equal(payload['y'].numpy(), frame.qualified_y) and np.array_equal(payload['original_cache_index'].numpy(), frame.original_cache_index)
    observed, state_sha = extract_function('scripts/research/collect_m3_formal_results.py', 'observed_states')
    scalar, scalar_sha = extract_function('scripts/research/analyze_13_feature_screening.py', 'observed_motion')
    feature_names = [name + '__median' for name in stats['columns']]
    extra = ['speed_median_mps', 'speed_std_mps', 'gyro_p90_radps', 'vz_median_mps', 'output_mean_median_native']
    feature_names += extra
    features = np.empty((len(frame), len(feature_names)), dtype=np.float64)
    ci = {name: i for i, name in enumerate(stats['columns'])}
    probes = []
    for begin in range(0, len(frame), 512):
        end = min(len(frame), begin + 512)
        raw = x[begin:end].numpy()
        assert np.isfinite(raw).all(), 'Nonfinite input; do not silently drop or fill'
        state = observed(raw, stats['columns'])
        frame.loc[begin:end - 1, 'state'] = state
        speed = np.linalg.norm(raw[:, [ci['vel_x'], ci['vel_y'], ci['vel_z']], :].astype(np.float64), axis=1)
        rotation = np.linalg.norm(raw[:, [ci['gyro_x'], ci['gyro_y'], ci['gyro_z']], :].astype(np.float64), axis=1)
        features[begin:end, :26] = np.median(raw.astype(np.float64), axis=2)
        features[begin:end, 26] = np.median(speed, axis=1)
        features[begin:end, 27] = np.std(speed, axis=1)
        features[begin:end, 28] = np.quantile(rotation, .9, axis=1)
        features[begin:end, 29] = np.median(raw[:, ci['vel_z'], :].astype(np.float64), axis=1)
        features[begin:end, 30] = np.median(raw[:, [ci['actuator_out_' + str(i)] for i in range(1, 5)], :].astype(np.float64).mean(axis=1), axis=1)
        for offset in [0, (end - begin) // 2, end - begin - 1]:
            q = scalar(raw[offset].T, stats['columns'])
            assert q['observed_state'] == state[offset]
            for key in ['speed_median_mps', 'speed_std_mps', 'gyro_p90_radps']:
                assert abs(q[key] - features[begin + offset, feature_names.index(key)]) < 1e-12
            probes.append({'compact_index': begin + offset, 'original_cache_index': int(frame.original_cache_index.iloc[begin + offset]), 'state': state[offset]})
        if begin % (512 * 40) == 0:
            print('TRAIN_WINDOWS_SUMMARIZED', end, '/', len(frame), flush=True)
    assert frame.state.isin(STATES).all()
    for i, name in enumerate(feature_names):
        frame[name] = features[:, i]
    write_json('state_rule_identity.json', {'vector_function_sha256': state_sha, 'scalar_function_sha256': scalar_sha,
                                          'scalar_vector_probe_checks': len(probes), 'no_parent_module_import': True})
    write_csv('state_function_probe_checks.csv', pd.DataFrame(probes))
    kept = ['original_cache_index', 'compact_index', 'case_uid', 'association_group', 'domain', 'fault_directory', 'flight_directory',
            'window_start_row', 'window_end_row', 'qualified_y', 'whole_window_label_valid', 'fault_fraction', 'phase', 'fault_onset_row',
            'normal_origin', 'known_normal', 'pure_fault', 'state'] + extra
    write_csv('window_state_ledger.csv.gz', frame[kept])
    for name, keys in [('domain', ['domain']), ('fault_directory', ['fault_directory']), ('flight_directory', ['flight_directory']),
                       ('state', ['state']), ('state_domain', ['state', 'domain']), ('state_fault_domain', ['state', 'fault_directory', 'domain']),
                       ('normal_origin', ['normal_origin']), ('flight_state', ['flight_directory', 'state'])]:
        write_csv('coverage_' + name + '.csv', grouped_counts(frame, keys))
    normal_rows = []
    for state in STATES:
        support = normal_support(frame[frame.state.eq(state)])
        normal_rows.append({'state': state, **support, 'reference_support_pass': ref_pass(support, plan)})
    normal_table = pd.DataFrame(normal_rows)
    write_csv('normal_state_support.csv', normal_table)
    # Cases and historical association groups are retained, not inferred from raw names.
    source_all = all_windows[all_windows.case_uid.ne('') & all_windows.domain.isin(['SIL', 'HIL'])].copy()
    case_cols = ['case_uid', 'association_group', 'domain', 'case_level2_label']
    cases_all = source_all[case_cols].drop_duplicates().copy()
    assert len(cases_all) == 2580 and cases_all.case_uid.is_unique
    cases_all['fault_directory'] = cases_all.case_uid.str.split('/').str[2]
    cases_all['flight_directory'] = cases_all.case_uid.str.split('/').str[1]
    cases_all['admitted_windows'] = cases_all.case_uid.map(frame.groupby('case_uid').size()).fillna(0).astype(int)
    cases_all['known_normal_windows'] = cases_all.case_uid.map(frame[frame.known_normal].groupby('case_uid').size()).fillna(0).astype(int)
    cases_all['fault_endpoint_windows'] = cases_all.case_uid.map(frame[frame.qualified_y.eq(1)].groupby('case_uid').size()).fillna(0).astype(int)
    cases_all['pure_fault_windows'] = cases_all.case_uid.map(frame[frame.pure_fault].groupby('case_uid').size()).fillna(0).astype(int)
    write_csv('case_identity_and_availability.csv', cases_all)
    cases = cases_all[cases_all.admitted_windows.gt(0)].copy()
    assert len(cases) == 2570
    assert not cases.duplicated(['association_group', 'domain']).any()
    assert cases.groupby('association_group').size().le(2).all()
    assert cases.groupby('association_group').fault_directory.nunique().eq(1).all()
    assert cases.groupby('association_group').flight_directory.nunique().eq(1).all()
    group_rows = []
    for group, part in cases_all.groupby('association_group'):
        group_rows.append({'association_group': group, 'case_count_original': len(part), 'cases_admitted': int(part.admitted_windows.gt(0).sum()),
                           'domains_original': ';'.join(sorted(part.domain)), 'domains_admitted': ';'.join(sorted(part.loc[part.admitted_windows.gt(0), 'domain'])),
                           'fault_directory': part.fault_directory.iloc[0], 'flight_directory': part.flight_directory.iloc[0],
                           'windows_admitted': int(part.admitted_windows.sum())})
    write_csv('association_groups.csv', pd.DataFrame(group_rows))
    # Normal trajectory descriptions use selected cap64 windows, not continuous frames.
    case_dynamics = []
    for case, part in frame[frame.known_normal].groupby('case_uid'):
        part = part.sort_values('window_end_row')
        state_counts = part.state.value_counts()
        case_dynamics.append({'case_uid': case, 'association_group': part.association_group.iloc[0], 'domain': part.domain.iloc[0],
                              'flight_directory': part.flight_directory.iloc[0], 'independent_normal': part.fault_directory.iloc[0] == 'no_fault',
                              'normal_windows': len(part), 'distinct_states': part.state.nunique(), 'dominant_state_fraction': float(state_counts.max() / len(part)),
                              'first_state': part.state.iloc[0], 'last_state': part.state.iloc[-1],
                              'selected_neighbor_state_changes': int(part.state.ne(part.state.shift()).sum() - 1),
                              'selected_endpoint_gap_median_s': float(np.median(np.diff(part.window_end_row)) * .02) if len(part) > 1 else None})
    write_csv('known_normal_case_dynamics.csv', pd.DataFrame(case_dynamics))
    pairs = []
    for case, part in frame[frame.fault_directory.ne('no_fault')].groupby('case_uid'):
        pre = part[part.known_normal & part.phase.eq('pre_fault')].sort_values('window_end_row')
        post = part[part.pure_fault & part.phase.eq('stable_fault')].sort_values('window_start_row')
        if not len(pre) or not len(post):
            continue
        a, b = pre.iloc[-1], post.iloc[0]
        pa = pre.state.value_counts(normalize=True).reindex(STATES, fill_value=0)
        pb = post.state.value_counts(normalize=True).reindex(STATES, fill_value=0)
        pairs.append({'case_uid': case, 'association_group': a.association_group, 'domain': a.domain, 'fault_directory': a.fault_directory,
                      'flight_directory': a.flight_directory, 'pre_ID': int(a.original_cache_index), 'post_ID': int(b.original_cache_index),
                      'pre_state': a.state, 'post_state': b.state, 'state_changed': a.state != b.state,
                      'pre_windows': len(pre), 'post_windows': len(post), 'endpoint_gap_s': float((b.window_end_row - a.window_end_row) * .02),
                      'pre_distance_to_onset_s': float((a.window_end_row - 1 - a.fault_onset_row) * .02),
                      'post_distance_to_onset_s': float((b.window_end_row - 1 - b.fault_onset_row) * .02),
                      'state_distribution_TV': float(.5 * np.abs(pa - pb).sum()),
                      **{name + '_delta': float(post[name].median() - pre[name].median()) for name in extra[:4]}})
    pairs = pd.DataFrame(pairs)
    write_csv('case_pre_post_state_pairs.csv', pairs)
    write_csv('pre_post_state_transitions.csv', pairs.groupby(['fault_directory', 'domain', 'pre_state', 'post_state']).agg(cases=('case_uid', 'nunique'), groups=('association_group', 'nunique')).reset_index())
    write_csv('pre_post_by_fault_domain.csv', pairs.groupby(['fault_directory', 'domain']).agg(cases=('case_uid', 'nunique'), groups=('association_group', 'nunique'), state_change_fraction=('state_changed', 'mean'), median_TV=('state_distribution_TV', 'median'), median_endpoint_gap_s=('endpoint_gap_s', 'median')).reset_index())
    # AUC and distribution descriptions do not train a reference or classifier.
    description = frame[frame.known_normal | frame.pure_fault].copy()
    description['known_label'] = np.where(description.known_normal, 'normal', 'fault')
    ckeys = ['case_uid', 'association_group', 'domain', 'fault_directory', 'flight_directory', 'state', 'known_label']
    case_features = description.groupby(ckeys)[feature_names].median().reset_index()
    write_csv('case_state_feature_summaries.csv.gz', case_features)
    distr = []
    for (state, domain), part in case_features[case_features.known_label.eq('normal')].groupby(['state', 'domain']):
        for feature in feature_names:
            values = part[feature].to_numpy()
            distr.append({'state': state, 'domain': domain, 'feature': feature, 'normal_cases': part.case_uid.nunique(),
                          'normal_groups': part.association_group.nunique(), 'case_q05': float(np.quantile(values, .05)),
                          'case_q25': float(np.quantile(values, .25)), 'case_median': float(np.median(values)),
                          'case_q75': float(np.quantile(values, .75)), 'case_q95': float(np.quantile(values, .95))})
    write_csv('normal_case_state_distributions.csv', pd.DataFrame(distr))
    aucs = []
    for state in STATES:
        for domain in ['SIL', 'HIL']:
            cf = case_features[case_features.state.eq(state) & case_features.domain.eq(domain)]
            n = cf[cf.known_label.eq('normal') & cf.fault_directory.eq('no_fault')]
            nf = frame[frame.state.eq(state) & frame.domain.eq(domain) & frame.known_normal & frame.fault_directory.eq('no_fault')]
            for fault in FAULTS:
                f = cf[cf.known_label.eq('fault') & cf.fault_directory.eq(fault)]
                ff = frame[frame.state.eq(state) & frame.domain.eq(domain) & frame.pure_fault & frame.fault_directory.eq(fault)]
                adequate = descriptive_cell(nf, ff, plan)
                for feature in feature_names:
                    a = auc_fixed_direction(n[feature].to_numpy(), f[feature].to_numpy())
                    aucs.append({'state': state, 'domain': domain, 'fault_directory': fault, 'feature': feature,
                                 'normal_cases': len(n), 'fault_cases': len(f), 'normal_groups': n.association_group.nunique(),
                                 'fault_groups': f.association_group.nunique(), 'adequate_descriptive_cell': adequate,
                                 'fixed_signed_case_AUC': a, 'normal_case_median': float(n[feature].median()) if len(n) else None,
                                 'fault_case_median': float(f[feature].median()) if len(f) else None})
    write_csv('conditional_case_AUC_descriptive.csv', pd.DataFrame(aucs))
    print('TRAIN_COVERAGE_AND_DISTRIBUTIONS_COMPLETE', flush=True)
    # Only admitted groups enter design folds. All original unavailable cases remain above.
    groups = cases.groupby('association_group').agg(fault_directory=('fault_directory', 'first'), flight_directory=('flight_directory', 'first'), cases=('case_uid', 'size')).reset_index()
    fold_summaries = []
    for k in plan['folds']:
        assigned = {}; loads = np.zeros(k, dtype=int)
        for (fault, flight), part in groups.groupby(['fault_directory', 'flight_directory'], sort=True):
            ordered = sorted(part.association_group, key=lambda g: hashlib.sha256(f"{plan['fold_seed']}|{k}|{fault}|{flight}|{g}".encode()).hexdigest())
            offset = int(np.argmin(loads))
            for j, group in enumerate(ordered):
                fold = (offset + j) % k
                assigned[group] = fold; loads[fold] += 1
        gf = groups.copy(); gf['outer_fold'] = gf.association_group.map(assigned).astype(int)
        write_csv(f'fold_assignments_{k}.csv', gf)
        membership = frame.association_group.map(assigned)
        fold_rows, support_rows, core_rows = [], [], []
        for fold in range(k):
            tr, ho = frame[membership.ne(fold)], frame[membership.eq(fold)]
            assert not set(tr.association_group) & set(ho.association_group)
            assert not set(tr.case_uid) & set(ho.case_uid)
            fold_rows += [{'fold': fold, 'role': role, **counts(part)} for role, part in [('outer_train', tr), ('outer_held_out', ho)]]
            for fault in FAULTS:
                for domain in ['SIL', 'HIL']:
                    q = ho[ho.qualified_y.eq(1) & ho.fault_directory.eq(fault) & ho.domain.eq(domain)]
                    m = plan['core_heldout_fault_domain_minimum']
                    core_rows.append({'fold': fold, 'fault_directory': fault, 'domain': domain, 'fault_windows': len(q),
                                      'fault_cases': q.case_uid.nunique(), 'fault_groups': q.association_group.nunique(),
                                      'pass': len(q) >= m['fault_windows'] and q.case_uid.nunique() >= m['fault_cases'] and q.association_group.nunique() >= m['fault_association_groups']})
            for state in STATES:
                trs = tr[tr.state.eq(state)]; hos = ho[ho.state.eq(state)]
                support = normal_support(trs); passes = ref_pass(support, plan)
                for fault in FAULTS:
                    for domain in ['SIL', 'HIL']:
                        n = hos[hos.known_normal & hos.domain.eq(domain)]
                        f = hos[hos.pure_fault & hos.fault_directory.eq(fault) & hos.domain.eq(domain)]
                        support_rows.append({'fold': fold, 'state': state, 'domain': domain, 'fault_directory': fault,
                                             **support, 'train_reference_pass': passes,
                                             'heldout_normal_windows': len(n), 'heldout_normal_cases': n.case_uid.nunique(),
                                             'heldout_normal_groups': n.association_group.nunique(), 'heldout_fault_windows': len(f),
                                             'heldout_fault_cases': f.case_uid.nunique(), 'heldout_fault_groups': f.association_group.nunique(),
                                             'heldout_cell_pass': descriptive_cell(n, f, plan)})
        support_frame = pd.DataFrame(support_rows); core_frame = pd.DataFrame(core_rows)
        write_csv(f'fold_populations_{k}.csv', pd.DataFrame(fold_rows))
        write_csv(f'fold_state_fault_domain_support_{k}.csv', support_frame)
        write_csv(f'fold_core_fault_domain_support_{k}.csv', core_frame)
        state_summary = support_frame.groupby('state').agg(train_reference_all_folds_pass=('train_reference_pass', 'all'),
                                                         heldout_all_fault_domain_cells_pass=('heldout_cell_pass', 'all'),
                                                         passing_heldout_cells=('heldout_cell_pass', 'sum'), total_heldout_cells=('heldout_cell_pass', 'size')).reset_index()
        write_csv(f'fold_state_summary_{k}.csv', state_summary)
        fold_summaries.append({'k': k, 'groups_per_fold': loads.tolist(), 'core_fault_domain_all_pass': bool(core_frame['pass'].all()),
                               'failed_core_cells': int((~core_frame['pass']).sum()), 'failed_state_cells': int((~support_frame.heldout_cell_pass).sum()),
                               'total_state_cells': len(support_frame), 'all_states_normal_supported': bool(support_frame.train_reference_pass.all()),
                               'state_summary': state_summary.to_dict(orient='records')})
        print('FOLD_SUPPORT_COMPLETE', k, flush=True)
    support_states = normal_table.loc[normal_table.reference_support_pass, 'state'].tolist()
    unsupported = frame[~frame.state.isin(support_states)]
    summary = {'status': 'COMPLETE_SOURCE_TRAIN_ONLY_NO_MODEL_FORWARD', 'population': counts(frame),
               'original_cases': len(cases_all), 'admitted_cases': len(cases), 'original_groups': cases_all.association_group.nunique(),
               'admitted_groups': cases.association_group.nunique(), 'all_excluded_cases': cases_all.loc[cases_all.admitted_windows.eq(0), 'case_uid'].tolist(),
               'pair_group_sizes_admitted': cases.groupby('association_group').size().value_counts().sort_index().to_dict(),
               'known_normal_support_states': support_states, 'unsupported_state_population': counts(unsupported),
               'pre_post_pairs': len(pairs), 'pre_post_groups': pairs.association_group.nunique(),
               'pre_post_changed': int(pairs.state_changed.sum()), 'pre_post_changed_fraction': float(pairs.state_changed.mean()),
               'normal_case_state_dynamics': {'cases': len(case_dynamics), 'multi_state_cases': sum(r['distinct_states'] > 1 for r in case_dynamics)},
               'fold_designs': fold_summaries, 'model_construction_forward_training_GPU_server': False,
               'Test_Real_data_read': False, 'encoder_latent_features_generated': False,
               'interpretation': 'Descriptive Train statistics and split-support feasibility only. No unseen-case performance, diagnostic efficacy or causal fault attribution.',
               'elapsed_s': time.perf_counter() - started}
    for relative, expected in identities.items():
        assert sha(DATA / relative) == expected, 'Train input changed during analysis: ' + relative
    summary['source_input_hashes_unchanged'] = True
    write_json('completion.json', summary)
    print(json.dumps(summary, ensure_ascii=True), flush=True)


if __name__ == '__main__':
    torch.set_num_threads(4)
    main()
