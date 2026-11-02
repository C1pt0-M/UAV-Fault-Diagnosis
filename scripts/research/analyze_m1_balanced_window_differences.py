"""Paired source-Val Balanced M0-matched/M1 differences; no retraining/tuning.

Use saved probabilities and source residual arrays. The frozen M1 adapter and
linear head allow exact decomposition of the residual logit-margin contribution,
without running the backbone or altering any model/checkpoint/input.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import sys
import numpy as np
import pandas as pd
import torch
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'packages/uav_fault'))
from uav_fault.models.cnn.mstca_kinematic_m1 import OnlineMSTCAKinematicM1

ARM = 'balanced_source_normal'
GROUPS = ['new_miss', 'recovered_fault', 'shared_miss', 'shared_TP',
          'resolved_FP', 'new_FP', 'shared_FP', 'shared_TN']
CHANGED = ['new_miss', 'recovered_fault', 'resolved_FP', 'new_FP']
COLUMNS = ['rh_signed_m', 'rR_x_rad', 'rR_y_rad', 'rR_z_rad']


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def align_prediction(prediction, expected_ids, labels):
    ids = np.asarray(prediction['original_indices'])
    expected_ids, labels = np.asarray(expected_ids), np.asarray(labels)
    if ids.dtype.kind not in 'iu' or len(np.unique(ids)) != len(ids) or len(np.unique(expected_ids)) != len(expected_ids):
        raise ValueError('Original window IDs must be unique integers')
    if len(ids) != len(expected_ids) or set(ids) != set(expected_ids):
        raise ValueError('Predictions do not have the identical original window population')
    positions = pd.Index(ids).get_indexer(expected_ids)
    if not np.array_equal(np.asarray(prediction['labels'])[positions], labels):
        raise ValueError('Paired original IDs have different labels')
    if not np.array_equal(np.asarray(prediction['compact_indices'])[positions], np.arange(len(labels))):
        raise ValueError('Compact positions disagree with original-ID join')
    q = np.asarray(prediction['probabilities'])[positions]
    if q.shape != (len(labels), 2) or not np.isfinite(q).all() or not ((q >= 0) & (q <= 1)).all() or not np.allclose(q.sum(1), 1, rtol=0, atol=1e-6):
        raise ValueError('Invalid saved probabilities')
    if np.any(q[:, 0] == q[:, 1]):
        raise ValueError('Probability ties require explicit historical argmax semantics')
    return q


def change_groups(y, predicted0, predicted1):
    y, a, b = map(np.asarray, [y, predicted0, predicted1])
    if y.shape != a.shape or a.shape != b.shape or not set(np.unique(np.r_[y, a, b])).issubset({0, 1}):
        raise ValueError('Aligned binary labels/predictions required')
    return np.select([(y == 1) & (a == 1) & (b == 0), (y == 1) & (a == 0) & (b == 1),
                      (y == 1) & (a == 0) & (b == 0), (y == 1) & (a == 1) & (b == 1),
                      (y == 0) & (a == 1) & (b == 0), (y == 0) & (a == 0) & (b == 1),
                      (y == 0) & (a == 1) & (b == 1)], GROUPS[:-1], default='shared_TN')


def residual_summaries(residual, mean, std):
    r = np.asarray(residual, dtype=np.float64)
    mu = np.asarray(mean, dtype=float)[None, :, None]
    scale = np.asarray(std, dtype=float)[None, :, None]
    if r.ndim != 3 or r.shape[1:] != (4, 150) or not np.isfinite(r).all() or scale.shape != (1, 4, 1) or not np.isfinite(scale).all() or not (scale > 0).all():
        raise ValueError('Finite physical4x150 residuals and frozen positive scales required')
    z = (r-mu)/scale
    out = {}
    for j in range(4):
        prefix = f'N{j+1}'
        out[prefix+'_signed_median'] = np.median(r[:, j], axis=1)
        out[prefix+'_abs_median'] = np.median(np.abs(r[:, j]), axis=1)
        out[prefix+'_abs_p95'] = np.quantile(np.abs(r[:, j]), .95, axis=1)
        out[prefix+'_std'] = np.std(r[:, j], axis=1)
        out[prefix+'_train_z_rms'] = np.sqrt(np.mean(z[:, j]**2, axis=1))
    norm = np.linalg.norm(r[:, 1:], axis=1)
    out['attitude_norm_median_rad'] = np.median(norm, axis=1)
    out['attitude_norm_p95_rad'] = np.quantile(norm, .95, axis=1)
    out['joint_train_z_rms'] = np.sqrt(np.mean(z**2, axis=(1, 2)))
    return pd.DataFrame(out)


def log_odds(q):
    q = np.asarray(q, dtype=np.float64)
    valid = (q > 0).all(1)
    out = np.full(len(q), np.nan)
    out[valid] = np.log(q[valid, 1])-np.log(q[valid, 0])
    return out


def decompose(q0, q1, residual_margin):
    total0, total1 = log_odds(q0), log_odds(q1)
    residual_margin = np.asarray(residual_margin)
    base1 = total1-residual_margin
    return pd.DataFrame({'M0_margin': total0, 'M1_margin': total1,
                         'M1_residual_margin': residual_margin,
                         'M1_common_path_margin': base1,
                         'common_path_change_vs_M0': base1-total0,
                         'finite_saved_log_odds': np.isfinite(total0) & np.isfinite(total1),
                         'residual_pivotal_to_normal': (base1 > 0) & (total1 < 0),
                         'residual_pivotal_to_fault': (base1 < 0) & (total1 > 0)})


def distribution(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return {'n': len(values), 'q10': float(np.quantile(values, .1)) if len(values) else None,
            'median': float(np.median(values)) if len(values) else None,
            'q90': float(np.quantile(values, .9)) if len(values) else None}


def controls(frame, target_name, control_name):
    records = []
    for _, row in frame[frame.group == target_name].iterrows():
        candidates = frame[(frame.group == control_name) & (frame.case_uid == row.case_uid) & (frame.phase == row.phase)]
        if target_name == 'resolved_FP':
            candidates = candidates[candidates.after_crop30s == row.after_crop30s]
        if not len(candidates):
            records.append({'target_original_id': int(row.original_cache_index), 'group': target_name,
                            'case_uid': row.case_uid, 'control_original_id': None})
            continue
        candidates = candidates.assign(probability_gap=np.abs(candidates.M0_fault_probability-row.M0_fault_probability),
                                       time_gap=np.abs(candidates.end_relative_crop_s-row.end_relative_crop_s))
        selected = candidates.sort_values(['probability_gap', 'time_gap', 'original_cache_index'], kind='stable').iloc[0]
        records.append({'target_original_id': int(row.original_cache_index), 'group': target_name,
                        'case_uid': row.case_uid, 'control_original_id': int(selected.original_cache_index),
                        'M0_probability_gap': float(selected.probability_gap), 'endpoint_gap_s': float(selected.time_gap)})
    return records


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ['protocol', 'data', 'features', 'results', 'output']:
        ap.add_argument('--'+name, type=Path, required=True)
    a = ap.parse_args()
    if a.output.exists():
        raise FileExistsError('Preserve independent window-difference output')
    protocol = read(a.protocol)
    if sha(a.protocol) != a.protocol.with_suffix('.json.sha256').read_text().strip():
        raise ValueError('Frozen M1 protocol changed')
    if sha(a.features/'build_manifest.json') != protocol['feature_manifest_sha256']:
        raise ValueError('Frozen feature manifest changed')
    manifest = read(a.features/'build_manifest.json')
    old = read(ROOT/protocol['M0_protocol_path'])
    inputs = {'protocol': {'path': str(a.protocol), 'sha256': sha(a.protocol)}}
    def check(path, expected):
        actual = sha(path)
        if actual != expected:
            raise ValueError('Historical input identity mismatch: '+str(path))
        inputs[str(path)] = {'sha256': actual, 'bytes': path.stat().st_size}
    for name in ['cache/val.pt', 'ledgers/val_included_windows.csv', 'norm_stats.json']:
        check(a.data/name, old['data_identities'][name])
    for split in ['train', 'val']:
        check(a.features/'cache'/(split+'.pt'), manifest['exports'][split]['sha256'])
    check(a.features/'residual_norm_stats.json', manifest['residual_norm_sha256'])
    norm = read(a.features/'residual_norm_stats.json')
    if norm['fit_split'] != 'source_train_normal_only' or norm['columns'] != COLUMNS:
        raise ValueError('Frozen source-train-only normalization required')
    source = pd.read_csv(a.data/'ledgers/val_included_windows.csv', keep_default_na=False, low_memory=False)
    export = torch.load(a.features/'cache/val.pt', weights_only=True, map_location='cpu', mmap=True)
    if export['split'] != 'val' or export['normalization_applied'] is not False or not export['full_window_valid'].all():
        raise ValueError('Full150 source-Val physical features required')
    f = source.iloc[export['base_compact_index'].numpy()].copy().reset_index(drop=True)
    if not np.array_equal(f.original_cache_index, export['original_cache_index']) or not np.array_equal(f.qualified_y, export['y']):
        raise ValueError('Source metadata/feature original-ID or label mismatch')
    if not f.split.eq('val').all() or not f.domain.isin(['SIL', 'HIL']).all() or len(f) != protocol['data_counts']['val']:
        raise ValueError('Only frozen source-Val population allowed')
    result_manifest = read(a.results/'manifest.json')
    probabilities = []
    for variant in ['M0_matched', 'M1']:
        relative = 'runs/'+variant+'/'+ARM+'/source_validation_predictions.npz'
        path = a.results/relative
        check(path, result_manifest['files'][relative]['sha256'])
        with np.load(path, allow_pickle=False) as z:
            probabilities.append(align_prediction(z, f.original_cache_index, f.qualified_y))
    q0, q1 = probabilities
    f['flight_status'] = f.case_uid.str.split('/').str[1]
    f['fault_directory'] = f.case_uid.str.split('/').str[2]
    f['normal_origin'] = np.where(f.qualified_y == 1, 'fault_endpoint', np.where(f.case_level2_label == 'normal', 'original_normal_case', 'normal_part_of_fault_case'))
    f['end_relative_crop_s'] = (f.window_end_row-1)*.02
    f['after_crop30s'] = f.end_relative_crop_s > 30
    f['end_relative_reference_onset_s'] = np.where(f.fault_onset_row >= 0, (f.window_end_row-1-f.fault_onset_row)*.02, np.nan)
    f['M0_fault_probability'], f['M1_fault_probability'] = q0[:, 1], q1[:, 1]
    f['probability_change'] = q1[:, 1]-q0[:, 1]
    f['M0_prediction'], f['M1_prediction'] = q0.argmax(1), q1.argmax(1)
    f['group'] = change_groups(f.qualified_y, f.M0_prediction, f.M1_prediction)
    feature_names = list(residual_summaries(export['residual'][:1], norm['mean'], norm['std']).columns)
    parts = []
    for start in range(0, len(f), 512):
        parts.append(residual_summaries(export['residual'][start:start+512], norm['mean'], norm['std']))
    f = pd.concat([f, pd.concat(parts, ignore_index=True)], axis=1)
    # Only existing source-Train features supply an empirical reference. These
    # are diagnostic percentiles, not a newly fitted model/scaler or threshold.
    check(a.data/'ledgers/train_included_windows.csv', old['data_identities']['ledgers/train_included_windows.csv'])
    train = torch.load(a.features/'cache/train.pt', weights_only=True, map_location='cpu', mmap=True)
    train_meta = pd.read_csv(a.data/'ledgers/train_included_windows.csv', keep_default_na=False, low_memory=False).iloc[train['base_compact_index'].numpy()].reset_index(drop=True)
    fit = ((train['y'].numpy() == 0) & train_meta.whole_window_label_valid.to_numpy(dtype=bool) & train_meta.fault_fraction.eq(0).to_numpy())
    if int(fit.sum()) != norm['fit_windows'] or not np.array_equal(train_meta.original_cache_index, train['original_cache_index']):
        raise ValueError('Exact source-Train normal reference population changed')
    normal_ref = train_meta[fit][['case_uid','domain']].reset_index(drop=True)
    normal_ref['flight_status'] = normal_ref.case_uid.str.split('/').str[1]
    fit_ids = np.flatnonzero(fit)
    normal_parts = []
    for start in range(0, len(fit_ids), 512):
        normal_parts.append(residual_summaries(train['residual'][fit_ids[start:start+512]], norm['mean'], norm['std']))
    normal_ref = pd.concat([normal_ref, pd.concat(normal_parts, ignore_index=True)], axis=1)
    percentile_names = [f'N{i}_abs_median' for i in range(1,5)]+['attitude_norm_median_rad','joint_train_z_rms']
    references = []
    for (domain, status), indices in f.groupby(['domain','flight_status']).groups.items():
        ref = normal_ref[(normal_ref.domain == domain) & (normal_ref.flight_status == status)]
        if not len(ref):raise ValueError('Missing source-Train normal comparison stratum')
        references.append({'domain':domain,'flight_status':status,'windows':len(ref),'cases':ref.case_uid.nunique(),
                           'feature_distributions':{name:distribution(ref[name]) for name in feature_names}})
        for name in percentile_names:
            sorted_values = np.sort(ref[name].to_numpy())
            values = f.loc[indices, name].to_numpy()
            rank = (np.searchsorted(sorted_values, values, side='left')+np.searchsorted(sorted_values, values, side='right'))/(2*len(ref))
            f.loc[indices, name+'_train_same_mode_percentile'] = rank
    # Physical baseline inputs are used for motion context, not new inference.
    base = torch.load(a.data/'cache/val.pt', weights_only=True, map_location='cpu', mmap=True)
    base_norm = read(a.data/'norm_stats.json')
    if base['x'].shape[1:] != (26,150):raise ValueError('Historical raw base26 tensor shape changed')
    names = base_norm['columns'];motion = []
    for start in range(0, len(f), 512):
        x = base['x'][export['base_compact_index'][start:start+512]].numpy()
        motion.append(pd.DataFrame({'vel_z_median_mps':np.median(x[:,names.index('vel_z')],axis=1),
                                    'horizontal_speed_median_mps':np.median(np.linalg.norm(x[:,[names.index('vel_x'),names.index('vel_y')]],axis=1),axis=1),
                                    'throttle_mean':np.mean(x[:,names.index('actuator_ctrl_4')],axis=1)}))
    f = pd.concat([f,pd.concat(motion,ignore_index=True)],axis=1)
    checkpoint = a.results/'runs/M1'/ARM/'best.pt'
    check(checkpoint,result_manifest['files']['runs/M1/'+ARM+'/best.pt']['sha256'])
    model = OnlineMSTCAKinematicM1();model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True));model.eval();torch.set_num_threads(4)
    if not isinstance(model.head[1],torch.nn.Linear) or len(model.head)!=2 or model.training:
        raise ValueError('Exact eval linear head required for additive margin decomposition')
    margin_weight = model.head[1].weight[1]-model.head[1].weight[0]
    mu=torch.tensor(norm['mean'],dtype=torch.float32).view(4,1);std=torch.tensor(norm['std'],dtype=torch.float32).view(4,1)
    contributions=[]
    with torch.inference_mode():
        for start in range(0,len(f),512):
            r=(export['residual'][start:start+512]-mu)/std
            contributions.append((model.residual_adapter(r)@margin_weight).numpy())
    f=pd.concat([f,decompose(q0,q1,np.concatenate(contributions))],axis=1)
    if not f[f.group.isin(CHANGED)].finite_saved_log_odds.all():raise ValueError('Changed window has saturated saved probabilities')
    counts=[];decompositions=[]
    for group in GROUPS:
        g=f[f.group==group]
        counts.append({'group':group,'windows':len(g),'cases':g.case_uid.nunique(),'association_groups':g.association_group.nunique()})
        finite=g[g.finite_saved_log_odds]
        decompositions.append({'group':group,'windows':len(g),'finite_log_odds':len(finite),
                               'residual_negative':int((g.M1_residual_margin<0).sum()),
                               'residual_pivotal_to_normal':int(g.residual_pivotal_to_normal.sum()),
                               'residual_pivotal_to_fault':int(g.residual_pivotal_to_fault.sum()),
                               'common_path_already_normal':int((finite.M1_common_path_margin<0).sum()),
                               'common_path_lower_than_M0':int((finite.common_path_change_vs_M0<0).sum()),
                               **{k:distribution(g[k]) for k in ['M0_fault_probability','M1_fault_probability','probability_change','M1_residual_margin','common_path_change_vs_M0']}})
    c={x['group']:x['windows'] for x in counts}
    if (c['new_miss']+c['shared_miss'],c['recovered_fault']+c['shared_miss'],c['resolved_FP']+c['shared_FP'],c['new_FP']+c['shared_FP']) != (526,402,472,335):
        raise ValueError('Paired gross transitions do not reproduce frozen Balanced results')
    breakdowns={}
    for field in ['fault_directory','domain','flight_status','phase','normal_origin','after_crop30s']:
        values=[]
        for value,part in f.groupby(field,sort=True):
            values.append({field:str(value),'fault_windows':int(part.qualified_y.eq(1).sum()),'normal_windows':int(part.qualified_y.eq(0).sum()),
                           **{group:int(part.group.eq(group).sum()) for group in GROUPS},
                           'new_miss_cases':part[part.group=='new_miss'].case_uid.nunique(),'resolved_FP_cases':part[part.group=='resolved_FP'].case_uid.nunique()})
        breakdowns[field]=values
    cross=f.groupby(['fault_directory','domain','flight_status','group']).size().unstack('group',fill_value=0).reset_index()
    feature_rows=[]
    for group in GROUPS:
        g=f[f.group==group]
        for name in feature_names+[n+'_train_same_mode_percentile' for n in percentile_names]+['vel_z_median_mps','horizontal_speed_median_mps','throttle_mean']:
            feature_rows.append({'group':group,'feature':name,'windows':len(g),**distribution(g[name]),
                                 'case_equal_median':float(g.groupby('case_uid')[name].median().median()) if len(g) else None})
    control_records=controls(f,'new_miss','shared_TP')+controls(f,'resolved_FP','shared_FP')
    by_id=f.set_index('original_cache_index');control_summary=[]
    for group in ['new_miss','resolved_FP']:
        matched=[r for r in control_records if r['group']==group and r['control_original_id'] is not None]
        target_ids=[r['target_original_id'] for r in matched];control_ids=[r['control_original_id'] for r in matched]
        control_summary.append({'group':group,'target_windows':int(f.group.eq(group).sum()),'matched_windows':len(matched),'unique_control_windows':len(set(control_ids)),
                                'M0_probability_gap':distribution([r['M0_probability_gap'] for r in matched]),'endpoint_gap_s':distribution([r['endpoint_gap_s'] for r in matched]),
                                'feature_comparison':{name:{'target':distribution(by_id.loc[target_ids,name]),'control':distribution(by_id.loc[control_ids,name]),'paired_difference':distribution(by_id.loc[target_ids,name].to_numpy()-by_id.loc[control_ids,name].to_numpy())} for name in feature_names},
                                'rule':'Same case, phase, and for normal fixes same crop30s bin; nearest saved M0 probability, then time distance then original ID. Controls reusable; missing pairs explicit, no inference of independent significance.'})
    # Deterministic descriptive examples, not best-looking/strongest changes.
    example_ids=[]
    def example(part,rule):
        if not len(part):return
        center=part.probability_change.median();row=part.assign(distance=(part.probability_change-center).abs()).sort_values(['distance','original_cache_index'],kind='stable').iloc[0]
        example_ids.append({'original_cache_index':int(row.original_cache_index),'selection_rule':rule})
    for subtype in ['GPS','accelerometer','gyroscope','barometer','magnetometer','motor']:
        example(f[(f.group=='new_miss')&(f.fault_directory==subtype)],'new_miss '+subtype+': closest to within-directory median probability change, tie original ID')
    for domain,status in [('SIL','hover'),('HIL','hover'),('HIL','velocity'),('SIL','dece')]:
        part=f[(f.group=='resolved_FP')&(f.domain==domain)&(f.flight_status==status)&(f.normal_origin=='original_normal_case')]
        if len(part):
            dominant=part.groupby('case_uid').size().sort_values(ascending=False,kind='stable').index[0]
            example(part[part.case_uid==dominant],f'resolved_FP {domain}/{status}: largest-count case, median probability change')
    example(f[(f.group=='new_miss')&(f.M1_residual_margin>0)],'counterexample: new_miss with positive direct residual contribution, median probability change')
    example(f[(f.group=='recovered_fault')&(f.fault_directory=='gyroscope')],'recovered_fault gyroscope: median probability change')
    examples=[]
    for chosen in example_ids:
        row=by_id.loc[chosen['original_cache_index']];uid=row.case_uid;key=uid.replace('/','__')+'.npz'
        sidepath=a.features/'cases'/key
        if 'cases/'+key not in manifest['files']:raise ValueError('Missing case-sidecar identity')
        check(sidepath,manifest['files']['cases/'+key]['sha256'])
        with np.load(sidepath,allow_pickle=False) as z:
            s,end=int(row.window_start_row),int(row.window_end_row)
            if not z['valid'][s:end].all():raise ValueError('Example residual window invalid')
            compact=f.index[f.original_cache_index==chosen['original_cache_index']][0]
            np.testing.assert_array_equal(z['residual'][s:end].astype(np.float32).T,export['residual'][compact].numpy())
            start_s,end_s=float(z['timestamp_us'][s]/1e6),float(z['timestamp_us'][end-1]/1e6)
        record=row.to_dict();record.update(chosen);record.update(window_first_publication_s=start_s,window_last_publication_s=end_s)
        examples.append(record)
    # Source tables only; no raw logs, final test/Real, no full-model inference.
    a.output.mkdir(parents=True,exist_ok=False)
    minimal=['original_cache_index','case_uid','association_group','domain','flight_status','fault_directory','qualified_y','normal_origin','window_start_row','window_end_row','phase','fault_fraction','end_relative_crop_s','end_relative_reference_onset_s','M0_fault_probability','M1_fault_probability','M0_prediction','M1_prediction','group']
    f[minimal].to_csv(a.output/'all_paired_predictions.csv',index=False)
    f[f.group.isin(CHANGED)].to_csv(a.output/'changed_windows.csv',index=False)
    pd.DataFrame(counts).to_csv(a.output/'group_counts.csv',index=False)
    cross.to_csv(a.output/'fault_domain_mode_transitions.csv',index=False)
    pd.DataFrame(feature_rows).to_csv(a.output/'feature_summaries.csv',index=False)
    pd.DataFrame(control_records).to_csv(a.output/'within_case_controls.csv',index=False)
    pd.DataFrame(examples).to_csv(a.output/'typical_windows.csv',index=False)
    case_counts=f.groupby(['case_uid','group']).size().unstack('group',fill_value=0).reset_index();case_counts.to_csv(a.output/'per_case_transitions.csv',index=False)
    result={'status':'PASS_PAIRED_BALANCED_SOURCE_VAL_WINDOW_DIFFERENCE_ANALYSIS','created_at':datetime.now(timezone.utc).isoformat(),'repository_commit':__import__('subprocess').check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip(),
            'inputs':inputs,'windows':len(f),'threshold':.5,'groups':counts,'breakdowns':breakdowns,'direct_residual_decomposition':decompositions,
            'feature_distributions':feature_rows,'source_train_normal_reference':references,'within_case_comparisons':control_summary,'typical_original_ids':example_ids,
            'near_boundary':{group:{'windows':int(f.group.eq(group).sum()),'M0_p_fault_below_0_6':int(((f.group==group)&(f.M0_fault_probability<.6)).sum())} for group in ['new_miss','resolved_FP']},
            'branch_interpretation':'At eval, dropout is identity and head linear: M1 margin=(w1-w0)h_base+(b1-b0)+(w1-w0)h_residual. Infer common-path margin by subtracting actual residual contribution from saved log probability ratio. This is frozen-model algebra, not a trained ablation, not independent feature importance, and not proof of physical causation. Shared backbone/head also changed during separate fine-tuning.',
            'analysis_source_sha256':sha(Path(__file__)),'no_training_or_threshold_change':True,'backbone_forward_calls':0,'final_test_or_Real_read':False,'historical_assets_modified':False,
            'limitations':['One seed, exposed source-Val, overlapping windows and paired SIL/HIL not independent.', 'Directory flight status is recorded scenario, crop30s is descriptive not certified phase.', 'Train-normal percentiles compare residuals within recorded domain/mode; not probability of health, and not motion/strength-matched experimental controls.', 'Time-asynchronous publication consistency is not physical health truth.','No per-axis causal attribution or claims that residual features caused learned backbone changes.']}
    # NaN in scalar tables denotes missing log-odds for saturated saved scores;
    # no model feature is filled, no saturated score is clipped for statistics.
    (a.output/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    artifacts={p.name:{'sha256':sha(p),'bytes':p.stat().st_size} for p in a.output.iterdir() if p.is_file()}
    (a.output/'manifest.json').write_text(json.dumps({'status':result['status'],'files':artifacts},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'groups':counts,'direct_decomposition':decompositions,'controls':[{k:v for k,v in x.items() if k!='feature_comparison'} for x in control_summary]},ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
