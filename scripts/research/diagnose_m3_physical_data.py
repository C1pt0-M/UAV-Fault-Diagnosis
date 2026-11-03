"""Local CPU-only statistics for frozen M3 physical candidates.

No model construction/forward/fitting, training, server, GPU or Test/Real.
Shared Train/Val windows are summarized once; raw provenance is bounded.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import sys
import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata
from scipy.spatial.distance import cdist
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'packages/uav_fault'), str(ROOT / 'scripts/research')]
from uav_fault.data.m3_features import old_candidate_features
from check_kinematic_feature_response import fixed_auc
from collect_m3_formal_results import observed_states
from continuous_kinematic_residuals import LocalSupportTrace
from validate_kinematic_residuals import rotation_wxyz, integrate_rotation
from scipy.spatial.transform import Rotation

IDS = [f'O{i}' for i in range(1, 10)] + [f'N{i}' for i in range(1, 5)]
FAULTS = ['GPS', 'accelerometer', 'barometer', 'gyroscope', 'magnetometer', 'motor']
FLIGHTS = ['hover', 'velocity', 'acce', 'dece', 'waypoint', 'circling']


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def save(p, value):
    Path(p).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def label_groups(frame):
    pure = frame.whole_window_label_valid.to_numpy(dtype=bool)
    y = frame.qualified_y.to_numpy()
    return np.where(pure & (y == 1) & frame.fault_fraction.eq(1), 'active',
                    np.where(pure & (y == 0) & frame.fault_fraction.eq(0),
                             np.where(frame.case_level2_label.eq('normal'), 'independent_normal', 'case_pre_normal'),
                             'mixed_or_unknown'))


def describe(values):
    x = np.asarray(values, dtype=float)
    if not len(x):
        return {}
    if not np.isfinite(x).all():
        raise ValueError('No deletion/filling of nonfinite summaries')
    return dict(n=len(x), mean=float(x.mean()), variance=float(x.var()),
                q10=float(np.quantile(x, .1)), median=float(np.median(x)),
                q90=float(np.quantile(x, .9)), minimum=float(x.min()), maximum=float(x.max()))


def paired_sign(positive, negative):
    delta = np.asarray(positive) - np.asarray(negative)
    if not len(delta):
        return None
    return float(np.mean((delta > 0) + .5 * (delta == 0)))


def native_residual_at(side, policy, index):
    """Original native support/formulas at one point, no new streaming replay."""
    t = int(side['timestamp_us'][index]); start = t - 200000
    traces = {}
    for group in ['baro', 'velocity', 'attitude', 'gyro']:
        prefix = group + '__native_'
        counters = {k[len(prefix):]: side[k] for k in side
                    if k.startswith(prefix) and k[len(prefix):] in ['baro_device_id', 'z_reset_counter', 'vz_reset_counter', 'quat_reset_counter']}
        traces[group] = LocalSupportTrace(side[prefix + 'observation_us'], side[prefix + 'available_us'],
                                         side[prefix + 'values'], int(policy['max_age_us'][group]),
                                         counters, side[prefix + 'valid'])
    info = {}
    for group, trace in traces.items():
        ix, dt, support, changed = trace.support(start, t)
        if changed:
            raise ValueError('Reset in selected admitted residual support')
        info[group] = (trace.point(start), trace.point(t), ix, dt)
    a, b, _, _ = info['baro']; _, _, ix, dt = info['velocity']
    n1 = traces['baro'].values[b, 0] - traces['baro'].values[a, 0] + np.sum(traces['velocity'].values[ix, 0] * dt)
    a, b, _, _ = info['attitude']; r0, r1 = rotation_wxyz(traces['attitude'].values[[a, b]]).as_matrix()
    _, _, ix, dt = info['gyro']; delta = integrate_rotation(traces['gyro'].values[ix], dt)
    return np.r_[n1, Rotation.from_matrix((r0 @ delta).T @ r1).as_rotvec()]


def extract(a, protocol, norm, inputs):
    columns = norm['columns']; names = IDS + columns
    mu = torch.tensor(norm['mean'], dtype=torch.float32)[None, :, None]
    sd = torch.tensor(norm['std'], dtype=torch.float32)[None, :, None]
    if not (sd > 0).all():
        raise ValueError('Frozen positive scales required')
    case_frames, summary = [], {}
    for split in ['train', 'val']:
        expected = protocol['data']['source_split_identities'][split]
        bp, rp = a.data / 'cache' / (split + '.pt'), a.features / 'cache' / (split + '.pt')
        for p, h in [(bp, expected['base_cache_sha256']), (rp, expected['residual_cache_sha256'])]:
            if sha(p) != h:
                raise ValueError('Frozen cache mismatch ' + str(p))
            inputs[str(p)] = h
        b = torch.load(bp, map_location='cpu', weights_only=True, mmap=True)
        r = torch.load(rp, map_location='cpu', weights_only=True, mmap=True)
        lp = a.features / 'ledgers' / (split + '_included_windows.csv')
        f = pd.read_csv(lp, keep_default_na=False, dtype={'association_group': str})
        if len(f) != expected['windows'] or not f.split.eq(split).all():
            raise ValueError('Wrong split population')
        ix = r['base_compact_index'].numpy()
        if not np.array_equal(f.original_cache_index, r['original_cache_index'].numpy()) or not np.array_equal(f.qualified_y, r['y'].numpy()):
            raise ValueError('Metadata/cache ID or label/order differs')
        if not torch.equal(b['original_cache_index'][ix], r['original_cache_index']) or not r['full_window_valid'].all():
            raise ValueError('Not the frozen joint150 domain')
        inputs[str(lp)] = sha(lp)
        frozen_source = read(ROOT / 'configs/experiments/causal_pure_mstca_m0_v1.json')['data_identities']
        base_ledger_path = a.data / 'ledgers' / (split + '_included_windows.csv')
        if sha(base_ledger_path) != frozen_source['ledgers/' + split + '_included_windows.csv']:
            raise ValueError('Frozen source ledger differs')
        base_ledger = pd.read_csv(base_ledger_path, keep_default_na=False, dtype={'association_group': str})
        linked = base_ledger.iloc[ix].reset_index(drop=True)
        if not linked.equals(f[linked.columns].reset_index(drop=True)):
            raise ValueError('Common metadata differs from frozen source rows')
        inputs[str(base_ledger_path)] = sha(base_ledger_path)
        f['split'] = split; f['flight'] = f.case_uid.str.split('/').str[1]
        f['fault'] = f.case_uid.str.split('/').str[2]; f['label_group'] = label_groups(f)
        n = len(f)
        arrays = {k: np.empty((n, 39), dtype=np.float32) for k in ['median', 'mean', 'variance']}
        arrays['drift'] = np.empty((n, 13), dtype=np.float32)
        o5_clamp = np.empty(n, dtype=np.float32)
        for start in range(0, n, 512):
            end = min(n, start + 512)
            physical = b['x'][ix[start:end]]
            z = (physical - mu) / sd
            old = old_candidate_features(z)
            new = r['residual'][start:end]
            values = torch.cat([old, new, physical], dim=1).numpy().astype(np.float64)
            if not np.isfinite(values).all():
                raise ValueError('Nonfinite joint window')
            arrays['median'][start:end] = np.median(values, axis=2)
            arrays['mean'][start:end] = values.mean(2)
            arrays['variance'][start:end] = values.var(2)
            arrays['drift'][start:end] = np.median(values[:, :13, 75:], axis=2) - np.median(values[:, :13, :75], axis=2)
            o5_clamp[start:end] = (z[:, columns.index('acc_norm')] <= .1).float().mean(1).numpy()
        f['o5_clamp_fraction'] = o5_clamp
        med = pd.DataFrame(arrays['median'], columns=names)
        keycols = ['split', 'case_uid', 'association_group', 'domain', 'flight', 'fault', 'label_group']
        case = pd.concat([f[keycols], med], axis=1).groupby(keycols, sort=True).median().reset_index()
        numbers = f.groupby(keycols, sort=True).size().reset_index(name='windows')
        case = case.merge(numbers, on=keycols, validate='one_to_one')
        case_frames.append(case)
        local = a.output / 'local_summaries' / (split + '.npz')
        local.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(local, original_ids=f.original_cache_index.to_numpy(), names=np.array(names),
                            o5_clamp=o5_clamp, **arrays)
        f.to_csv(a.output / 'local_summaries' / (split + '_metadata.csv'), index=False)
        summary[split] = dict(frame=f, arrays=arrays)
        print('STAT_SUMMARIES', split, n, 'cases', f.case_uid.nunique(), flush=True)
    case = pd.concat(case_frames, ignore_index=True)
    case.to_csv(a.output / 'case_feature_medians.csv', index=False)
    return case, summary, names


def statistical_tables(a, case, summaries, names):
    distributions, aucs, within, correlations, matching = [], [], [], [], []
    # Case-level AUC: independent normal pool, same split/domain/scenario.
    for (split, domain, flight), sub in case.groupby(['split', 'domain', 'flight'], sort=True):
        normals = sub[sub.label_group.eq('independent_normal')]
        for fault in FAULTS:
            active = sub[sub.fault.eq(fault) & sub.label_group.eq('active')]
            for name in IDS:
                if len(normals) and len(active):
                    aucs.append(dict(split=split, domain=domain, flight=flight, fault=fault, feature=name,
                                     normal_cases=len(normals), fault_cases=len(active),
                                     normal_association_groups=normals.association_group.nunique(),
                                     fault_association_groups=active.association_group.nunique(),
                                     signed_AUC=fixed_auc(normals[name], active[name]),
                                     small_sample=min(len(normals), len(active)) < 4))
        for group in ['independent_normal', 'active', 'case_pre_normal']:
            f = sub[sub.label_group.eq(group)]
            if len(f):
                for name in IDS:
                    distributions.append(dict(split=split, domain=domain, flight=flight, group=group,
                                              feature=name, cases=len(f), windows=int(f.windows.sum()), **describe(f[name])))
    # Same-case active versus pre-normal; condition on case, not only directory.
    for uid, sub in case.groupby('case_uid', sort=True):
        act = sub[sub.label_group.eq('active')]; pre = sub[sub.label_group.eq('case_pre_normal')]
        if len(act) == len(pre) == 1:
            ac, pr = act.iloc[0], pre.iloc[0]
            for name in IDS:
                within.append(dict(case_uid=uid, split=ac.split, domain=ac.domain, flight=ac.flight,
                                   fault=ac.fault, association_group=ac.association_group, feature=name,
                                   active_windows=int(ac.windows), pre_windows=int(pr.windows),
                                   active_minus_pre=float(ac[name] - pr[name])))
    # Descriptive centered ranks within condition strata; not fitted regressions.
    for split in ['train', 'val']:
        sub = case[case.split.eq(split) & case.label_group.ne('mixed_or_unknown')].copy()
        for scope in ['pooled_case_label', 'conditioned_domain_flight_fault_label']:
            values = sub[names].rank(method='average')
            used = sub
            if scope.startswith('conditioned'):
                keys = ['domain', 'flight', 'fault', 'label_group']
                size = sub.groupby(keys).case_uid.transform('size')
                used = sub[size >= 4]
                values = pd.DataFrame(index=used.index, columns=names, dtype=float)
                for _, block in used.groupby(keys, sort=True):
                    ranked = block[names].rank(method='average')
                    values.loc[block.index] = ranked - ranked.mean()
            corr = values.corr(min_periods=4)
            for feature in IDS:
                for base in names[13:]:
                    v = corr.loc[feature, base]
                    correlations.append(dict(split=split, scope=scope, feature=feature, base=base,
                                             rows=len(used), cases=used.case_uid.nunique(),
                                             rank_correlation=None if pd.isna(v) else float(v)))
        # Nearest independent-normal CASE matching on all26 median summaries.
        # Fixed source normalization; no fitting, prediction, or sign selection.
        norm = read(a.data / 'norm_stats.json'); mu = np.asarray(norm['mean']); sd = np.asarray(norm['std'])
        for (domain, flight), block in sub.groupby(['domain', 'flight'], sort=True):
            normal = block[block.label_group.eq('independent_normal')]
            active = block[block.label_group.eq('active')]
            if len(normal) < 2 or not len(active):
                continue
            zn = (normal[names[13:]].to_numpy() - mu) / sd
            za = (active[names[13:]].to_numpy() - mu) / sd
            nn = cdist(zn, zn) / np.sqrt(26); np.fill_diagonal(nn, np.inf)
            reference_p90 = float(np.quantile(nn.min(1), .9))
            distances = cdist(za, zn) / np.sqrt(26)
            nearest = distances.argmin(1)
            for i, (_, row) in enumerate(active.iterrows()):
                control = normal.iloc[nearest[i]]; distance = float(distances[i, nearest[i]])
                for feature in IDS:
                    matching.append(dict(split=split, domain=domain, flight=flight, fault=row.fault,
                                         feature=feature, case_uid=row.case_uid, normal_case_uid=control.case_uid,
                                         association_group=row.association_group,
                                         Base26_summary_RMS_distance=distance,
                                         normal_leave_case_out_nearest_P90=reference_p90,
                                         inside_normal_descriptive_support=distance <= reference_p90,
                                         active_minus_matched_normal=float(row[feature] - control[feature])))
    for filename, rows in [('condition_distributions.csv', distributions), ('signed_case_AUC.csv', aucs),
                           ('same_case_response.csv', within), ('Base26_candidate_correlations.csv', correlations),
                           ('Base26_summary_matched_case_differences.csv', matching)]:
        pd.DataFrame(rows).to_csv(a.output / filename, index=False)
    # Explicit means/variances: equal case averages of window time moments.
    moments = []
    for split, payload in summaries.items():
        f, arrays = payload['frame'], payload['arrays']
        for (domain, flight, label), ix in f.groupby(['domain', 'flight', 'label_group']).groups.items():
            part = f.loc[ix]
            for j, name in enumerate(IDS):
                means = pd.DataFrame({'case_uid': part.case_uid, 'value': arrays['mean'][ix, j]}).groupby('case_uid').value.mean()
                variances = pd.DataFrame({'case_uid': part.case_uid, 'value': arrays['variance'][ix, j]}).groupby('case_uid').value.mean()
                drifts = pd.DataFrame({'case_uid': part.case_uid, 'value': arrays['drift'][ix, j]}).groupby('case_uid').value.median()
                moments.append(dict(split=split, domain=domain, flight=flight, group=label, feature=name,
                                    windows=len(ix), cases=part.case_uid.nunique(),
                                    equal_case_mean_of_window_means=float(means.mean()),
                                    between_case_variance_of_window_means=float(means.var(ddof=0)),
                                    equal_case_mean_within_window_variance=float(variances.mean()),
                                    median_case_second_minus_first_half=float(drifts.median())))
    pd.DataFrame(moments).to_csv(a.output / 'condition_time_moments.csv', index=False)


def hover_diagnosis(a, summaries, names, inputs):
    f, arrays = summaries['val']['frame'].copy(), summaries['val']['arrays']
    p = a.predictions / 'window_comparison.csv'; inputs[str(p)] = sha(p)
    predictions = pd.read_csv(p, keep_default_na=False)
    if not np.array_equal(predictions.original_cache_index, f.original_cache_index) or not np.array_equal(predictions.label, f.qualified_y):
        raise ValueError('M3 prediction/metadata identity mismatch')
    for col in ['p_C0', 'p_CN', 'pred_C0', 'pred_CN']:
        f[col] = predictions[col]
    hover = f.flight.eq('hover'); fault = f.qualified_y.eq(1)
    lost = hover & fault & f.pred_C0.eq(1) & f.pred_CN.eq(0)
    if lost.sum() != 154:
        raise ValueError('Expected all154 original Hover new-FN windows')
    f['comparison_group'] = np.select([
        lost, hover & fault & f.pred_C0.eq(1) & f.pred_CN.eq(1),
        hover & ~fault & f.case_level2_label.eq('normal') & f.pred_C0.eq(1) & f.pred_CN.eq(0),
        hover & ~fault & f.case_level2_label.eq('normal') & f.pred_C0.eq(0) & f.pred_CN.eq(1),
        hover & ~fault & f.case_level2_label.eq('normal') & f.pred_C0.eq(1) & f.pred_CN.eq(1),
        hover & ~fault & f.case_level2_label.eq('normal') & f.pred_C0.eq(0) & f.pred_CN.eq(0)],
        ['CN_new_FN', 'both_TP', 'CN_corrected_normal_FP', 'CN_new_normal_FP', 'both_normal_FP', 'both_normal_TN'], default='other')
    out = pd.concat([f, pd.DataFrame(arrays['median'], columns=names)], axis=1)
    selected = out[hover & out.comparison_group.ne('other')]
    selected.to_csv(a.output / 'hover_window_medians.csv', index=False)
    distributions = []
    for (group, label, faultdir, domain), sub in selected.groupby(['comparison_group', 'label_group', 'fault', 'domain']):
        case = sub.groupby('case_uid')[names].median()
        for name in names:
            distributions.append(dict(group=group, label_group=label, fault=faultdir, domain=domain,
                                      feature=name, windows=len(sub), cases=len(case), **describe(case[name])))
    pd.DataFrame(distributions).to_csv(a.output / 'hover_error_distributions.csv', index=False)
    # Matched correct faults from the same CASE, nearest crop end; no sample deletion.
    paired = []
    for _, row in out[lost].iterrows():
        control = out[out.case_uid.eq(row.case_uid) & out.comparison_group.eq('both_TP')
                      & out.label_group.eq(row.label_group)]
        if not len(control):
            paired.append(dict(original_id=int(row.original_cache_index), case_uid=row.case_uid,
                               label_group=row.label_group, matched=False, reason='No same-case/same-purity both-TP control'))
            continue
        distance = (control.window_end_row - row.window_end_row).abs()
        normal = control.loc[distance.idxmin()]
        for feature in names:
            paired.append(dict(original_id=int(row.original_cache_index), case_uid=row.case_uid,
                               label_group=row.label_group, matched=True, feature=feature,
                               control_original_id=int(normal.original_cache_index),
                               end_time_distance_s=float(abs(normal.window_end_row - row.window_end_row) * .02),
                               lost_minus_correct=float(row[feature] - normal[feature])))
    pd.DataFrame(paired).to_csv(a.output / 'hover_same_case_correct_controls.csv', index=False)
    return out, lost


def provenance(a, out, lost, inputs):
    norm = read(a.data / 'norm_stats.json'); policy = read(a.data / 'input_contract.json')
    audits = []; series = []
    target_cases = sorted(out[lost].case_uid.unique())
    # All29 target cases use saved native provenance, not a new full-log audit.
    for uid in target_cases:
        key = uid.replace('/', '__')
        bp, sp = a.data / 'cases' / (key + '.npz'), a.features / 'cases' / (key + '.npz')
        record = read(a.data / 'case_records' / (key + '.json'))
        sr = read(a.features / 'case_records' / (key + '.json'))
        if record['split'] != 'val' or sr['split'] != 'val':
            raise ValueError('Only source-Val target cases')
        if sha(bp) != sr['source_case_sha256'] or sha(sp) != sr['sidecar_sha256']:
            raise ValueError('Changed original case/sidecar')
        inputs[str(bp)], inputs[str(sp)] = sha(bp), sha(sp)
        with np.load(bp, allow_pickle=False) as z: b = {k: z[k] for k in z.files}
        with np.load(sp, allow_pickle=False) as z: s = {k: z[k] for k in z.files}
        if not np.array_equal(b['timestamp_us'], s['timestamp_us']):
            raise ValueError('Native sidecar/grid mismatch')
        ix = out.index[out.case_uid.eq(uid) & lost]
        errors, ages, skew, exactgrid = [], [], [], []
        for position in ix:
            row = out.loc[position]; start, end = int(row.window_start_row), int(row.window_end_row)
            if not s['valid'][start:end].all() or s['reason'][start:end].any() or not b['row_input_valid'][start:end].all():
                raise ValueError('Invalid raw support in one of154 admitted windows')
            physical = b['canonical_rows'][start:end].copy()
            physical[:, norm['columns'].index('baro_alt_relative')] -= physical[0, norm['columns'].index('baro_alt_relative')]
            physical[:, norm['columns'].index('acc_norm')] = np.linalg.norm(physical[:, [norm['columns'].index('accel_' + x) for x in 'xyz']], axis=1)
            xn = (physical.T.astype(np.float32) - np.asarray(norm['mean'], dtype=np.float32)[:, None]) / np.asarray(norm['std'], dtype=np.float32)[:, None]
            old = old_candidate_features(torch.from_numpy(xn)[None])[0].numpy()
            allvalues = np.vstack([old, s['residual'][start:end].T, physical.T])
            error = np.max(np.abs(np.median(allvalues, axis=1) - row[IDS + norm['columns']].to_numpy(dtype=float)))
            if error > 2e-5:
                raise ValueError('Case/native/cache summary mismatch')
            errors.append(float(error))
            ages.append(int(s['endpoint_age_us'][start:end].max()))
            skew.append(int(np.max(np.abs(s['endpoint_age_us'][start:end, 2] - s['endpoint_age_us'][start:end, 3]))))
            endindex = end - 1
            native = native_residual_at(s, policy, endindex)
            if not np.allclose(native, s['residual'][endindex], rtol=0, atol=1e-10):
                raise ValueError('Native support residual formula mismatch')
            # Descriptive coarse-grid N1 approximation; never replaces the candidate.
            j = norm['columns'].index('baro_alt_relative'); vz = norm['columns'].index('vel_z')
            gridn1 = b['canonical_rows'][endindex, j] - b['canonical_rows'][endindex - 10, j] + b['canonical_rows'][endindex - 10:endindex, vz].sum() * .02
            exactgrid.append(float(gridn1 - native[0]))
        audits.append(dict(case_uid=uid, target_windows=len(ix), case_cache_summary_max_error=max(errors),
                           invalid_positions_in_target=0, residual_reason_bits_in_target=0,
                           max_endpoint_age_us=max(ages), max_attitude_gyro_observation_skew_us=max(skew),
                           grid_N1_minus_native_N1_median=float(np.median(exactgrid)),
                           grid_N1_minus_native_N1_max_abs=float(np.max(np.abs(exactgrid))),
                           declared_quaternion_resets_in_case=sr['native_counter_info']['attitude']['changes'].get('quat_reset_counter', 0)))
    pd.DataFrame(audits).to_csv(a.output / 'all154_native_provenance.csv', index=False)
    # Six raw ULogs: largest new-FN case for each priority sensor, two normal controls,
    # and the previously identified high-confidence barometer failure.
    priorities = []
    target = out[lost]
    for fault in ['GPS', 'gyroscope', 'magnetometer']:
        counts = target[target.fault.eq(fault)].groupby('case_uid').size().sort_values(ascending=False, kind='stable')
        priorities.append(counts.index[0])
    for group in ['CN_corrected_normal_FP', 'CN_new_normal_FP']:
        counts = out[out.comparison_group.eq(group)].groupby('case_uid').size().sort_values(ascending=False, kind='stable')
        priorities.append(counts.index[0])
    priorities.append('HIL/dece/barometer/TestCase_126_2504000125')
    from pyulog import ULog
    topics = {'baro': ('vehicle_air_data', ['baro_alt_meter']),
              'velocity': ('vehicle_local_position', ['vz']),
              'attitude': ('vehicle_attitude', [f'q[{i}]' for i in range(4)]),
              'gyro': ('sensor_combined', [f'gyro_rad[{i}]' for i in range(3)])}
    raw_checks = []
    for uid in priorities:
        key = uid.replace('/', '__'); record = read(a.data / 'case_records' / (key + '.json'))
        raw = Path(record['raw_ulog']); sp = a.features / 'cases' / (key + '.npz'); bp = a.data / 'cases' / (key + '.npz')
        if record['split'] != 'val' or sha(raw) != record['raw_ulog_sha256']:
            raise ValueError('Raw log source split/SHA mismatch')
        inputs[str(raw)] = sha(raw)
        log = ULog(str(raw), message_name_filter_list=[t for t, _ in topics.values()] + ['sensor_gps', 'sensor_mag'])
        with np.load(sp, allow_pickle=False) as z: s = {k: z[k] for k in z.files}
        with np.load(bp, allow_pickle=False) as z: b = {k: z[k] for k in z.files}
        for group, (topic, fields) in topics.items():
            found = [d.data for d in log.data_list if d.name == topic and d.multi_id == 0]
            if len(found) != 1:
                raise ValueError('Required native topic missing/ambiguous')
            data = found[0]; ix = s[group + '__native_source_index']
            vals = np.column_stack([data[k][ix] for k in fields])
            if not np.array_equal(vals, s[group + '__native_values']):
                raise ValueError('Original raw native values differ')
            if not np.array_equal(data['timestamp'][ix], s[group + '__native_available_us']):
                raise ValueError('Original raw native availability differs')
            raw_checks.append(dict(case_uid=uid, topic=topic, native_records_checked=len(ix),
                                   raw_values_and_available_clock_exact=True, raw_sha256=inputs[str(raw)]))
        for topic in ['sensor_gps', 'sensor_mag']:
            found = [d for d in log.data_list if d.name == topic and d.multi_id == 0]
            raw_checks.append(dict(case_uid=uid, topic=topic, native_records_checked=len(found[0].data['timestamp']) if len(found)==1 else 0,
                                   raw_values_and_available_clock_exact=None, raw_sha256=inputs[str(raw)],
                                   note='Presence recorded only; no calibrated GPS truth or decoded fault severity.'))
        # Full cropped physical traces (unique frames, not repeated windows).
        frame = pd.DataFrame(b['canonical_rows'], columns=norm['columns'])
        frame['case_uid'] = uid; frame['crop_seconds'] = np.arange(len(frame)) * .02
        frame['reference_state'] = b['reference_state']; frame['label_valid'] = b['row_label_valid']
        for j, name in enumerate(IDS[9:]): frame[name] = s['residual'][:, j]
        frame['residual_valid'] = s['valid'].all(1)
        frame['attitude_reason_bits'] = s['reason'][:, 1]
        frame['vertical_reason_bits'] = s['reason'][:, 0]
        frame['gyro_attitude_skew_us'] = np.max(np.abs(s['endpoint_age_us'][:, 2] - s['endpoint_age_us'][:, 3]), axis=1)
        # Compute old9 per original evaluation window; continuous Old8 is not substituted.
        representative = out[out.case_uid.eq(uid)].sort_values('window_end_row')
        for _, row in representative.iterrows():
            start, end = int(row.window_start_row), int(row.window_end_row)
            if end > len(frame): raise ValueError('Window exceeds saved source crop')
            physical = b['canonical_rows'][start:end].copy()
            physical[:, norm['columns'].index('baro_alt_relative')] -= physical[0, norm['columns'].index('baro_alt_relative')]
            physical[:, norm['columns'].index('acc_norm')] = np.linalg.norm(physical[:, [norm['columns'].index('accel_' + x) for x in 'xyz']], axis=1)
            z = (physical.T.astype(np.float32) - np.asarray(norm['mean'], dtype=np.float32)[:, None]) / np.asarray(norm['std'], dtype=np.float32)[:, None]
            old = old_candidate_features(torch.from_numpy(z)[None])[0].numpy()
            for j,name in enumerate(IDS[:9]):
                series.append(dict(case_uid=uid, original_id=int(row.original_cache_index), crop_end_seconds=float((end-1)*.02),
                                   feature=name, median=float(np.median(old[j])), variance=float(np.var(old[j]))))
        dest=a.output/'local_traces';dest.mkdir(exist_ok=True)
        frame.to_csv(dest/(key+'.csv'),index=False)
        print('RAW_LOG_VERIFIED',uid,flush=True)
    pd.DataFrame(raw_checks).to_csv(a.output/'six_raw_log_checks.csv',index=False)
    pd.DataFrame(series).to_csv(a.output/'six_case_old9_window_responses.csv',index=False)
    return priorities


def sensor_topic_statistics(a):
    """Bounded raw sensor-topic description, not decoded damage or new features."""
    from pyulog import ULog
    priorities=read(a.output/'completion.json')['six_raw_cases'] if (a.output/'completion.json').exists() else []
    if not priorities: raise ValueError('Existing bounded raw-case list required')
    rows=[]
    fields={'sensor_gps':['lat','lon','alt','eph','epv','vel_m_s','fix_type','satellites_used'],
            'sensor_gyro':['x','y','z','error_count'], 'sensor_mag':['x','y','z','error_count'],
            'vehicle_magnetometer':['magnetometer_ga[0]','magnetometer_ga[1]','magnetometer_ga[2]']}
    for uid in priorities:
        key=uid.replace('/','__');record=read(a.data/'case_records'/(key+'.json'))
        if record['split']!='val' or sha(record['raw_ulog'])!=record['raw_ulog_sha256']:raise ValueError('Raw sensor scope/SHA mismatch')
        with np.load(a.data/'cases'/(key+'.npz'),allow_pickle=False) as z:
            grid=z['timestamp_us'];state=z['reference_state'];valid=z['row_label_valid']
        log=ULog(record['raw_ulog'],message_name_filter_list=list(fields))
        for topic,columns in fields.items():
            found=[d.data for d in log.data_list if d.name==topic and d.multi_id==0]
            if len(found)!=1:
                rows.append(dict(case_uid=uid,topic=topic,phase='unavailable',field='',raw_records=0));continue
            data=found[0];t=data['timestamp']
            if np.any(np.diff(t.astype(np.int64))<0):raise ValueError('Original raw topic timestamp reversal')
            inside=(t>=grid[0])&(t<=grid[-1]);index=np.searchsorted(grid,t[inside],side='right')-1
            states=state[index];known=valid[index]
            for label in [0,1]:
                choose=inside.copy();choose[inside]=known&(states==label)
                times=t[choose];count=len(times)
                for field in columns:
                    if field not in data:continue
                    values=data[field][choose].astype(float);finite=np.isfinite(values)
                    row=dict(case_uid=uid,topic=topic,phase='known_normal' if label==0 else 'known_fault',field=field,
                             raw_records=count,finite_records=int(finite.sum()),nonfinite_records=int((~finite).sum()),
                             median_raw_interval_us=float(np.median(np.diff(times.astype(np.int64)))) if count>1 else None,
                             units='Original topic field; GNSS integer coordinate/scaling not reinterpreted as truth.')
                    if count and finite.all():row.update(describe(values))
                    rows.append(row)
    path=a.output/'raw_sensor_topic_phase_statistics.csv'
    if path.exists():raise FileExistsError(path)
    pd.DataFrame(rows).to_csv(path,index=False)


def motion_diagnostics(a):
    """Same previously frozen vel/gyro proxy; supplemental statistics, no fitting."""
    norm = read(a.data / 'norm_stats.json'); names = IDS + norm['columns']
    rows, coverage, case_parts = [], [], []
    for split in ['train', 'val']:
        frame = pd.read_csv(a.output / 'local_summaries' / (split + '_metadata.csv'), keep_default_na=False,
                            dtype={'association_group': str})
        with np.load(a.output / 'local_summaries' / (split + '.npz'), allow_pickle=False) as z:
            medians = z['median']
        b = torch.load(a.data / 'cache' / (split + '.pt'), map_location='cpu', weights_only=True, mmap=True)
        r = torch.load(a.features / 'cache' / (split + '.pt'), map_location='cpu', weights_only=True, mmap=True)
        if not np.array_equal(frame.original_cache_index, r['original_cache_index'].numpy()):
            raise ValueError('Motion metadata population differs')
        ix = r['base_compact_index'].numpy(); state = np.empty(len(frame), dtype=object)
        for start in range(0, len(frame), 512):
            end = min(len(frame), start + 512)
            state[start:end] = observed_states(b['x'][ix[start:end]].numpy(), norm['columns'])
        frame['observed_state'] = state
        frame[['split', 'original_cache_index', 'observed_state']].to_csv(a.output / 'local_summaries' / (split + '_motion.csv'), index=False)
        f = pd.concat([frame, pd.DataFrame(medians, columns=names)], axis=1)
        keys = ['split','case_uid','association_group','domain','flight','fault','label_group','observed_state']
        case = f[f.label_group.isin(['active','independent_normal','case_pre_normal'])].groupby(keys)[names].median().reset_index()
        case_parts.append(case)
        normal = case[case.label_group.eq('independent_normal')]
        for (domain, flight, fault), active in case[case.label_group.eq('active')].groupby(['domain','flight','fault']):
            controls = normal[normal.domain.eq(domain) & normal.flight.eq(flight)]
            for st, block in active.groupby('observed_state'):
                cn = controls[controls.observed_state.eq(st)]
                coverage.append(dict(split=split, domain=domain, flight=flight, fault=fault, observed_state=st,
                                     active_case_state_rows=len(block), control_cases=len(cn), matched=bool(len(cn))))
                if not len(cn): continue
                for feature in IDS:
                    rows.append(dict(split=split, domain=domain, flight=flight, fault=fault, observed_state=st,
                                     feature=feature, normal_cases=len(cn), fault_cases=len(block),
                                     signed_AUC=fixed_auc(cn[feature],block[feature]), small_sample=min(len(cn),len(block))<4))
    for name, data in [('motion_conditioned_case_medians.csv',pd.concat(case_parts)),
                       ('motion_conditioned_signed_AUC.csv',pd.DataFrame(rows)),
                       ('motion_control_coverage.csv',pd.DataFrame(coverage))]:
        path=a.output/name
        if path.exists(): raise FileExistsError('Preserve previous motion statistics')
        data.to_csv(path,index=False)
    print('COMPLETE_FIXED_MOTION_STATISTICS',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,default=Path('D:/UAV-Data/causal_input_v1'))
    p.add_argument('--features',type=Path,default=Path('D:/UAV-Data/causal_m1_continuous_v1'))
    p.add_argument('--predictions',type=Path,default=ROOT/'artifacts/M3_error_samples_v1_20261103')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError('Preserve any prior diagnosis')
    a.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2)
    protocol_path=ROOT/'configs/experiments/M3_preflight_v1.json'
    protocol=read(protocol_path);norm=read(a.data/'norm_stats.json')
    frozen_source=read(ROOT/'configs/experiments/causal_pure_mstca_m0_v1.json')['data_identities']
    if sha(a.data/'norm_stats.json')!=frozen_source['norm_stats.json']:raise ValueError('Frozen Base26 normalization differs')
    if sha(protocol_path)!=protocol_path.with_suffix('.json.sha256').read_text().strip():raise ValueError('Protocol SHA differs')
    inputs={str(protocol_path):sha(protocol_path),str(a.data/'norm_stats.json'):sha(a.data/'norm_stats.json')}
    plan={'scope':'CPU-only statistical summaries of existing shared Train151833/Val32381; no model fitting/inference or large new experiments.',
          'primary_summary':'Signed per-window temporal median, then one median per case/label. Within-window moments and half-window changes supplementary.',
          'AUC_direction':'Signed feature larger means fault; no flipping, threshold selection or feature deletion.',
          'population':'All shared windows retained; pure active and known normal distinguished from mixed/unknown.',
          'case_dependence':'Case counts and association groups; no window-level p-values, no statistical significance claim.',
          'Base26_control':'Descriptive rank centering within domain/flight/fault/label; nearest normal case on fixed-normalized26 median summaries, with unmatched/outside normal support reported. Not equality of full26 sequences or a proof of independent value.',
          'native_diagnosis':'All154 target windows in29 cases; six existing raw source-Val ULogs, no full-log rebuild/replay.',
          'changes_to_formulas_models_configs_data':False}
    save(a.output/'scope.json',plan)
    case,summaries,names=extract(a,protocol,norm,inputs)
    statistical_tables(a,case,summaries,names)
    motion_diagnostics(a)
    out,lost=hover_diagnosis(a,summaries,names,inputs)
    priorities=provenance(a,out,lost,inputs)
    files={p.relative_to(a.output).as_posix():dict(sha256=sha(p),bytes=p.stat().st_size) for p in a.output.rglob('*') if p.is_file()}
    save(a.output/'completion.json',{'status':'COMPLETE_CPU_ONLY_M3_PHYSICAL_DATA_DIAGNOSIS_STATISTICS',
         'Train_windows':151833,'Val_windows':32381,'target_hover_new_FN':154,
         'target_pure_active':int(out[lost].label_group.eq('active').sum()),'target_mixed_or_unknown':int(out[lost].label_group.eq('mixed_or_unknown').sum()),
         'target_cases':int(out[lost].case_uid.nunique()),'six_raw_cases':priorities,
         'input_sha256':inputs,'outputs':files,'server_GPU_model_training_forward_Test_Real':False,
         'new_feature_formulas_or_normalization':False,'generated_at':datetime.now(timezone.utc).isoformat()})
    sensor_topic_statistics(a)
    print('COMPLETE_CPU_PHYSICAL_STATISTICS',flush=True)


if __name__=='__main__':main()
