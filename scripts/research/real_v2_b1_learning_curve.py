"""Frozen nested-CV Real B1. CPU only; no sealed/test input or outer selection.

Models, frame features/scores and window OOF arrays stay in a fresh local
runtime directory. Every fit and local artifact is journalled before proceeding.
A failed B0 reproduction gate stops before learning-curve/event metrics.
"""
from __future__ import annotations
import csv
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'scripts/research'), str(ROOT / 'packages/uav_fault')]
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score
from threadpoolctl import threadpool_limits
import real_v2_b0_diagnostics as b0

DATA = b0.DATA
CFG = ROOT / 'configs/experiments/real_v2_b1_learning_curve.json'
OUT = ROOT / 'artifacts/real_v2_b1_learning_curve'
LOCAL = Path('D:/UAV-Data/real_v2_b1_learning_curve_runtime')
FRACTIONS = (.10, .25, .50, 1.)
GRID = (.45, .50, .55, .60, .65, .70, .75, .80, .85, .90, .95)
POLICY = dict(inference_every=1, clear_th=.4, k_on=3, k_off=3, min_event_s=.5, merge_gap_s=.5)
ARMS = ('R', 'S_plus_R')


def boundary(eligible, data=DATA, local=LOCAL, source=b0.SOURCE):
    data, local, source = Path(data).resolve(), Path(local).resolve(), Path(source).resolve()
    allowed = {data / 'cache/train.pt', data / 'ledgers/train_included_windows.csv', data / 'ledgers/real_included_windows.csv'}
    kinds = {}
    for uid in eligible.case_uid:
        for kind in ('windows', 'cases'):
            p = data / kind / (uid.replace('/', '__') + '.npz')
            allowed.add(p); kinds[p] = kind
    log = {'windows': [], 'cases': [], 'source_inputs': [], 'blocked': []}
    def guard(event, args):
        if event != 'open' or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        p = Path(os.fsdecode(args[0])).resolve()
        write = any(c in (args[1] or '') for c in 'wax+') or args[2] & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND)
        bad = ((p.is_relative_to(data.parent) and not p.is_relative_to(local) and (p not in allowed or write))
               or (p.is_relative_to(source.parent) and (p != source or write))
               or p.is_relative_to((ROOT / 'artifacts/internal_test_final_v1').resolve()))
        if bad:
            log['blocked'].append(str(p)); raise PermissionError('B1 input boundary: ' + str(p))
        if p in kinds:
            log[kinds[p]].append(str(p))
        elif p in allowed:
            log['source_inputs'].append(str(p))
    sys.addaudithook(guard)
    return log


def subset_groups(eligible, outer_fold, fraction, rep):
    if fraction not in FRACTIONS or rep not in range(5):
        raise ValueError('Unregistered fraction/repeat')
    train = eligible[eligible.dev_fold.ne(outer_fold)]
    groups = train[['association_group', 'stratum_family']].drop_duplicates()
    if groups.association_group.duplicated().any():
        raise ValueError('A group cannot have multiple family strata')
    selected = []
    for family, part in groups.groupby('stratum_family', sort=True):
        if family not in ('motor', 'sensor', 'normal_full', 'normal_segment'):
            raise ValueError('Unknown subset stratum')
        order = sorted(part.association_group, key=lambda g: hashlib.sha256(f'real_v2_b1|{rep}|{g}'.encode('utf-8')).hexdigest())
        selected.extend(order[:math.ceil(fraction * len(order))])
    assert not set(selected) & set(eligible.loc[eligible.dev_fold.eq(outer_fold), 'association_group'])
    return sorted(selected)


def domain_weights(source_weights, real_weights):
    if (not len(source_weights) or not len(real_weights) or not np.isfinite(source_weights).all()
            or not np.isfinite(real_weights).all() or (source_weights <= 0).any() or (real_weights <= 0).any()):
        raise ValueError('Positive finite domain weights required')
    scaled = real_weights * (source_weights.sum() / real_weights.sum())
    return np.r_[source_weights, scaled], float(source_weights.sum() / real_weights.sum())


def ap_plan(p, y):
    return b0.auc_plan(-np.asarray(p), y, np.ones(len(y), bool))


def weighted_ap(plan, weights):
    idx, positive, starts = plan
    if not len(idx):
        return None
    w = weights[idx]
    pos = np.add.reduceat(w * positive, starts)
    neg = np.add.reduceat(w * ~positive, starts)
    n = float(pos.sum())
    if not n:
        return None
    tp, total = np.cumsum(pos), np.cumsum(pos + neg)
    precision = np.divide(tp, total, out=np.zeros_like(tp, dtype=float), where=total > 0)
    return float(np.dot(pos, precision) / n)


def h1_bootstrap(probabilities, frame, eligible, reps=2000, seed=20261108):
    y = frame.qualified_y.to_numpy()
    plans = {key: ap_plan(p, y) for key, p in probabilities.items()}
    groups = np.unique(eligible.association_group.to_numpy())
    inverse = pd.Index(groups).get_indexer(frame.association_group)
    assert (inverse >= 0).all()
    samples = {f: [] for f in FRACTIONS}
    rng = np.random.default_rng(seed)
    for _ in range(reps):
        w = np.bincount(rng.integers(0, len(groups), len(groups)), minlength=len(groups))[inverse]
        aps = {key: weighted_ap(plan, w) for key, plan in plans.items()}
        for frac in FRACTIONS:
            nrep = 1 if frac == 1 else 5
            delta = [(aps[('S_plus_R', frac, rep)] - aps[('R', frac, rep)])
                     if aps[('S_plus_R', frac, rep)] is not None and aps[('R', frac, rep)] is not None else None
                     for rep in range(nrep)]
            samples[frac].append(float(np.mean(delta)) if all(x is not None for x in delta) else None)
    ones = np.ones(len(y), int)
    point = {key: weighted_ap(plan, ones) for key, plan in plans.items()}
    result = {}
    for frac, values in samples.items():
        nrep = 1 if frac == 1 else 5
        mean = float(np.mean([point[('S_plus_R', frac, rep)] - point[('R', frac, rep)] for rep in range(nrep)]))
        undefined = sum(x is None for x in values)
        ci = np.percentile(values, [2.5, 97.5]).tolist() if not undefined else None
        result[str(frac)] = {'repeat_mean_delta_AP': mean, 'ci95': ci, 'undefined_replicates': undefined,
                             'primary': frac == .25, 'holds': bool(ci is not None and ci[0] > 0) if frac == .25 else None}
    return {'fractions': result, 'replicates': reps, 'rng_seed': seed, 'groups': len(groups),
            'statistic': 'paired group resampling, AP per arm/repeat recomputed; mean delta over repetitions',
            'no_drop_redraw_pseudocount': True}


def event_record(meta, arrays, p, onset_th, aggregate=None):
    if aggregate is None:
        from replay_m1_continuous import aggregate_scores
        aggregate = aggregate_scores
    from uav_fault.online.detection_policy import OnlineDetectionPolicy
    times = arrays['timestamp_us'] / 1e6
    ref, valid = arrays['reference_state'], arrays['row_label_valid']
    if p.shape != times.shape or np.isinf(p).any() or ((p[np.isfinite(p)] < 0) | (p[np.isfinite(p)] > 1)).any():
        raise ValueError('Invalid frame scores')
    scored = np.isfinite(p)
    normal = valid & (ref == 0) & scored  # Frozen B1 explicitly includes scored.
    on = np.flatnonzero(valid & (ref == 1))
    onset = float(times[on[0]]) if len(on) else None
    fault_case = meta['role_v1'] == 'test_fault'
    full = meta['role_v1'] in b0.FULL_ROLES
    pre = normal & (times < onset) if fault_case and onset is not None else np.zeros(len(times), bool)
    post = normal & (times >= onset) if fault_case and onset is not None else np.zeros(len(times), bool)
    ev = aggregate(times, p, policy=OnlineDetectionPolicy(onset_th=onset_th, **POLICY))
    false = []
    for e in ev['finalized_events']:
        mask = (times >= e['start_t']) & (times <= e['end_t'])
        if mask.any() and normal[mask].all():
            false.append(e)
    conf = [c for c in ev['confirmations'] if fault_case and onset is not None and c['candidate_start_t'] >= onset]
    n2 = sum(e['end_t'] < onset for e in false) if fault_case and onset is not None else 0
    post_events = sum(e['start_t'] >= onset for e in false) if fault_case and onset is not None else 0
    return {**meta, 'onset_th': onset_th, 'onset_s': onset, 'N1_frames': int(normal.sum()) if full else 0,
        'N2_frames': int(pre.sum()), 'post_frames': int(post.sum()),
        'supplement_frames': int(normal.sum()) if meta['role_v1'] == 'supplemental_normal_test' else 0,
        'N1_events': len(false) if full else 0, 'N2_events': int(n2), 'post_events': int(post_events),
        'supplement_events': len(false) if meta['role_v1'] == 'supplemental_normal_test' else 0,
        'post_onset_scored': bool(fault_case and onset is not None and (scored & (times >= onset)).any()),
        'detected': bool(conf), 'latency_s': float(conf[0]['confirmed_t'] - onset) if conf else None,
        'spans_onset': bool(onset is not None and any(e['start_t'] < onset <= e['end_t'] for e in ev['finalized_events']))}


def event_summary(rows, weights=None):
    d = pd.DataFrame(rows)
    w = np.ones(len(d), int) if weights is None else np.asarray(weights)
    result = {}
    for name in ('N1', 'N2', 'post', 'supplement'):
        events = int(np.dot(w, d[name + '_events']))
        frames = int(np.dot(w, d[name + '_frames']))
        result[name] = {'events': events, 'normal_frames': frames, 'hours': frames * .02 / 3600,
                        'rate': events * 180000 / frames if frames else None}
    for family in ('motor', 'sensor'):
        mask = d.family.eq(family).to_numpy() & d.post_onset_scored.to_numpy(dtype=bool)
        detected = mask & d.detected.to_numpy(dtype=bool)
        n, tp = int(w[mask].sum()), int(w[detected].sum())
        lat = np.repeat(d.loc[detected, 'latency_s'].to_numpy(dtype=float), w[detected].astype(int))
        result[family] = {'cases': n, 'detected': tp, 'detection': tp / n if n else None,
                          'latency_median_s': float(np.median(lat)) if len(lat) else None,
                          'latency_p90_s': float(np.quantile(lat, .9)) if len(lat) else None}
    return result


def select_onset(grid_rows):
    """Exact integer/Fraction ties, including PI supplement 1 fallback A."""
    def rate(row, family):
        v = row[family]
        if not v['cases']:
            raise ValueError('Inner detection denominator unavailable')
        return Fraction(v['detected'], v['cases'])
    def objective(row):
        n = row['N1']['normal_frames'] + row['N2']['normal_frames']
        if not n:
            raise ValueError('No inner normal exposure')
        return Fraction(row['N1']['events'] + row['N2']['events'], n)
    def total(row):return row['motor']['detected'] + row['sensor']['detected']
    feasible = [row for row in grid_rows if min(rate(row, 'motor'), rate(row, 'sensor')) >= Fraction(4, 5)]
    if feasible:
        best = min(objective(row) for row in feasible)
        tied = [row for row in feasible if objective(row) == best]
    else:
        best = max(min(rate(row, 'motor'), rate(row, 'sensor')) for row in grid_rows)
        tied = [row for row in grid_rows if min(rate(row, 'motor'), rate(row, 'sensor')) == best]
    most = max(total(row) for row in tied)
    final = [row for row in tied if total(row) == most]
    chosen = min(final, key=lambda row: row['onset_th'])
    return {'onset_th': chosen['onset_th'], 'no_feasible_onset': not bool(feasible),
            'tied_onsets_primary': sorted(row['onset_th'] for row in tied),
            'tied_onsets_after_detection': sorted(row['onset_th'] for row in final),
            'inner': chosen, 'supplement1_fallback_A': True}


def h2(summary):
    checks = {'N1': summary['N1']['rate'] is not None and summary['N1']['rate'] <= 5.5,
              'N2': summary['N2']['rate'] is not None and summary['N2']['rate'] <= 5.5,
              'motor': summary['motor']['detection'] is not None and summary['motor']['detection'] >= .8,
              'sensor': summary['sensor']['detection'] is not None and summary['sensor']['detection'] >= .8}
    return {'holds': all(checks.values()), 'checks': checks, 'failed': [k for k,v in checks.items() if not v]}


def event_bootstrap(rows_by_arm, eligible, reps=2000, seed=20261108):
    groups = np.unique(eligible.association_group.to_numpy())
    metrics = [('N1', 'rate'), ('N2', 'rate'), ('motor', 'detection'), ('sensor', 'detection'),
               ('motor', 'latency_median_s'), ('motor', 'latency_p90_s'), ('sensor', 'latency_median_s'), ('sensor', 'latency_p90_s')]
    samples = {arm: {a + '/' + b: [] for a,b in metrics} for arm in rows_by_arm}
    frames = {arm: pd.DataFrame(rows) for arm,rows in rows_by_arm.items()}
    inverse = {arm: pd.Index(groups).get_indexer(d.association_group) for arm,d in frames.items()}
    assert all((v >= 0).all() for v in inverse.values())
    rng = np.random.default_rng(seed)
    for _ in range(reps):
        count = np.bincount(rng.integers(0,len(groups),len(groups)),minlength=len(groups))
        for arm, d in frames.items():
            s = event_summary(d, count[inverse[arm]])
            for a,b in metrics:samples[arm][a+'/'+b].append(s[a][b])
    return {'replicates': reps, 'rng_seed': seed, 'groups': len(groups), 'fixed_selected_onsets': True,
        'no_drop_redraw_pseudocount': True,
        'arms': {arm: {name: {'ci95': np.percentile(v,[2.5,97.5]).tolist() if all(x is not None for x in v) else None,
                             'undefined_replicates': sum(x is None for x in v)} for name,v in tab.items()} for arm,tab in samples.items()}}


def consequences(h1_holds, h2_holds):
    if not h2_holds:return {'key':'H2_fails','conclusion':'No B2; back to data/labels with user','automatic_next_step':False}
    if h1_holds:return {'key':'H1_and_H2','conclusion':'Propose B2 to user; no automatic GPU/T1 launch','automatic_next_step':False}
    return {'key':'H2_only','conclusion':'Real supervision usable; no measurable source benefit at 25%; user decides whether T1 is worth GPU','automatic_next_step':False}


def run():
    if OUT.exists() or LOCAL.exists():raise FileExistsError('Fresh B1 output AND runtime required')
    eligible, startup = b0.eligible_split(b0.SPLIT, b0.SPLIT_SHA)
    if len(eligible) != 315 or b0.sha(b0.SOURCE) != b0.SOURCE_SHA:raise ValueError('Startup identity/population mismatch')
    cfg = json.loads(CFG.read_text(encoding='utf-8'))
    if cfg['status'] != 'FROZEN_BEFORE_B1_RUN' or tuple(cfg['learning_curve']['fractions']) != FRACTIONS:raise ValueError('Protocol not frozen')
    OUT.mkdir(parents=True,exist_ok=False);LOCAL.mkdir(parents=True,exist_ok=False)
    for sub in ('models','oof','features','scores','inner_events'):(LOCAL/sub).mkdir()
    accesses = boundary(eligible)
    manifest = {};fits=[];subsets=[];window_rows=[];selections=[];scans=[];probabilities={};full_models={}
    started=time.perf_counter();stage='startup'
    with (OUT/'run.log').open('x',encoding='utf-8',newline='\n') as journal, (OUT/'fit_receipts.jsonl').open('x',encoding='utf-8') as fitlog, (OUT/'local_artifacts.jsonl').open('x',encoding='utf-8') as artifactlog:
        def note(*args):print(*args,flush=True);print(*args,file=journal,flush=True)
        def record(path):
            rel=path.relative_to(LOCAL).as_posix();manifest[rel]=b0.sha(path)
            artifactlog.write(json.dumps({'file':rel,'sha256':manifest[rel]})+'\n');artifactlog.flush()
            return path
        def npz(path,**values):
            path.parent.mkdir(parents=True,exist_ok=True)
            with path.open('xb') as f:np.savez_compressed(f,**values)
            return record(path)
        def access_summary():
            allowed={str((DATA/k/(uid.replace('/','__')+'.npz')).resolve()) for k in ('windows','cases') for uid in eligible.case_uid}
            assert set(accesses['windows'])|set(accesses['cases'])<=allowed
            return {**accesses,'unique_windows':len(set(accesses['windows'])),'unique_cases':len(set(accesses['cases'])), 'dev_only':True}
        try:
            note('STARTUP_PASS',json.dumps(startup))
            frame=b0.read_ledger(DATA/'ledgers/real_included_windows.csv',eligible)
            features=[];ids=[]
            for uid in eligible.case_uid:
                part=frame[frame.case_uid.eq(uid)]
                features.append(b0.window_features(DATA/'windows'/(uid.replace('/','__')+'.npz'),part));ids.extend(part.original_cache_index.tolist())
            x=np.concatenate(features);del features
            frame=frame.set_index('original_cache_index').loc[ids].reset_index();y=frame.qualified_y.to_numpy();masks=b0.populations(frame)
            source_data=None
            def fit(arm,train_mask,name,excluded):
                assert not frame.loc[train_mask,'dev_fold'].isin(excluded).any()
                real_x,real_y=x[train_mask],y[train_mask]
                real_w=b0.real_weights(real_y,frame.loc[train_mask,'normal_origin'])
                scale=1.
                if arm=='S_plus_R':
                    sx,sy,sw=source_data
                    weights,scale=domain_weights(sw,real_w)
                    xx,yy=np.concatenate([sx,real_x]),np.r_[sy,real_y]
                else:xx,yy,weights=real_x,real_y,real_w
                note('FIT_START',len(fits)+1,name,'real',len(real_y),'source',len(xx)-len(real_y))
                start=time.perf_counter();model=HistGradientBoostingClassifier(**b0.RECIPE)
                model.fit(xx,yy,sample_weight=weights)
                path=LOCAL/'models'/(name+'.pkl')
                with path.open('xb') as f:pickle.dump(model,f,protocol=pickle.HIGHEST_PROTOCOL)
                record(path)
                row={'name':name,'arm':arm,'real_windows':len(real_y),'source_windows':len(xx)-len(real_y),
                    'real_groups':int(frame.loc[train_mask,'association_group'].nunique()),'training_folds':sorted(map(int,frame.loc[train_mask,'dev_fold'].unique())),
                    'excluded_folds':list(excluded),'n_iter':int(model.n_iter_),'fit_time_s':time.perf_counter()-start,
                    'real_weight_scale':scale,'real_weight_total':float(real_w.sum()*scale),
                    'source_weight_total':float(weights[:len(xx)-len(real_y)].sum()) if arm=='S_plus_R' else 0.,
                    'local_model':str(path),'sha256':manifest[path.relative_to(LOCAL).as_posix()]}
                fits.append(row);fitlog.write(json.dumps(row)+'\n');fitlog.flush();note('FIT_COMPLETE',len(fits),name,round(row['fit_time_s'],3))
                return model,path
            def predict(model,xx):
                if not len(xx):return np.empty(0,float)
                p=np.concatenate([model.predict_proba(xx[i:i+4096])[:,1] for i in range(0,len(xx),4096)])
                if not np.isfinite(p).all() or ((p<0)|(p>1)).any():raise ValueError('Invalid GBT probability')
                return p
            with threadpool_limits(limits=16):
                stage='B0_reproduction'
                gatep=np.full(len(y),np.nan)
                for k in range(5):
                    tr=frame.dev_fold.ne(k).to_numpy();held=~tr
                    model,path=fit('R',tr,f'outer_R_f1p00_rep0_k{k}',[k]);gatep[held]=predict(model,x[held]);full_models[('R',k)]=path;del model
                old=pd.read_csv(ROOT/'artifacts/real_v2_b0_diagnostics/oof_predictions.csv')
                if not old.original_cache_index.is_unique or set(old.original_cache_index)!=set(ids):raise ValueError('B0 gate index coverage differs')
                old=old.set_index('original_cache_index').loc[ids]
                same_labels=bool(np.array_equal(old.qualified_y.to_numpy(),y))
                difference=float(np.abs(gatep-old.p_R.to_numpy()).max())
                gate={'windows':len(y),'max_abs_difference':difference,'limit':1e-6,'labels_equal':same_labels,'pass':same_labels and difference<=1e-6}
                b0.save(OUT/'reproduction_gate.json',gate)
                if not gate['pass']:
                    b0.save(OUT/'results.json',{'status':'STOPPED_REPRODUCTION','gate':gate,'access':access_summary(),'metrics_computed':False})
                    return 2
                note('B0_REPRODUCTION_PASS',difference)
                # The five gate fits ARE the registered full-R outer models.
                probabilities[('R',1.,0)]=gatep
                stage='source_train_features'
                import torch
                source_frame=pd.read_csv(DATA/'ledgers/train_included_windows.csv',keep_default_na=False)
                cache=torch.load(DATA/'cache/train.pt',map_location='cpu',weights_only=True,mmap=True)
                if (cache['split']!='train' or cache['normalization_applied'] is not False or cache['x'].dtype!=torch.float32
                    or tuple(cache['x'].shape)!=(162069,26,150) or not np.array_equal(cache['original_cache_index'].numpy(),source_frame.original_cache_index)
                    or not np.array_equal(cache['y'].numpy(),source_frame.qualified_y)):
                    raise ValueError('Source train cache/ledger mismatch')
                sx=np.concatenate([b0.stats(cache['x'][i:i+4096].numpy()) for i in range(0,len(cache['x']),4096)])
                if not np.isfinite(sx).all():raise ValueError('Nonfinite raw source features; no imputation')
                sy=cache['y'].numpy().copy();sw=b0.real_weights(sy,source_frame.normal_origin);source_data=(sx,sy,sw);del cache
                b0.save(OUT/'source_train_receipt.json',{'windows':len(sy),'raw_float32':True,'normalization_applied':False,
                    'cache_sha256':b0.sha(DATA/'cache/train.pt'),'ledger_sha256':b0.sha(DATA/'ledgers/train_included_windows.csv'),'weight_total':float(sw.sum())})
                with b0.SOURCE.open('rb') as f:source_model=pickle.load(f)
                def window_metric(arm,frac,rep,p):
                    pre=masks['pre_fault'];plan=b0.auc_plan(p,y,(y==1)|pre)
                    fpr=b0.fpr_at_recall(p,y,masks,.8)
                    row={'arm':arm,'fraction':frac,'repeat':rep,'AP':float(average_precision_score(y,p)),
                        'AUC_b':b0.weighted_auc(plan,np.ones(len(y),int)),'threshold_recall08':fpr['threshold'],'actual_recall':fpr['actual_recall'],
                        **{'FPR_'+name:v for name,v in fpr['normal_FPR'].items()}}
                    npz(LOCAL/'oof'/f'{arm}_f{frac:.2f}_rep{rep}.npz',probabilities=p,original_cache_index=np.asarray(ids,np.int64))
                    return row
                stage='learning_curve'
                for frac in FRACTIONS:
                    for rep in range(1 if frac==1 else 5):
                        for arm in ARMS:
                            key=(arm,frac,rep)
                            if key not in probabilities:
                                p=np.full(len(y),np.nan)
                                for k in range(5):
                                    groups=subset_groups(eligible,k,frac,rep)
                                    if arm=='R':subsets.append({'outer_fold':k,'fraction':frac,'repeat':rep,'groups':groups})
                                    tr=frame.association_group.isin(groups).to_numpy();held=frame.dev_fold.eq(k).to_numpy()
                                    assert not set(frame.loc[tr,'association_group'])&set(frame.loc[held,'association_group'])
                                    name=f'outer_{arm}_f{frac:.2f}_rep{rep}_k{k}'
                                    model,path=fit(arm,tr,name,[k]);p[held]=predict(model,x[held]);del model
                                    if frac==1:full_models[(arm,k)]=path
                                if not np.isfinite(p).all():raise ValueError('Incomplete outer OOF')
                                probabilities[key]=p
                            window_rows.append(window_metric(arm,frac,rep,probabilities[key]))
                            pd.DataFrame(window_rows).to_csv(OUT/'window_metrics.csv',index=False)
                source_window=window_metric('S',1.,0,predict(source_model,x))
                b0.save(OUT/'window_summary.json',{'rows':window_rows,'S_reference':source_window})
                b0.save(OUT/'subsets.json',subsets)
                stage='H1_bootstrap'
                h1=h1_bootstrap(probabilities,frame,eligible);b0.save(OUT/'H1.json',h1)
                note('WINDOWS_AND_H1_COMPLETE')
                stage='frame_features'
                from replay_ens_events_val import case_windows
                cache_paths={}
                for i,row in enumerate(eligible.itertuples(index=False),1):
                    uid=row.case_uid;key=uid.replace('/','__')+'.npz'
                    with np.load(DATA/'cases'/key,allow_pickle=False) as z:
                        times,raw,valid=z['timestamp_us'],z['canonical_rows'],z['row_input_valid'];ref,lv=z['reference_state'],z['row_label_valid']
                    if (times.dtype!=np.int64 or times.ndim!=1 or not len(times) or (np.diff(times)<=0).any() or raw.shape!=(len(times),26)
                        or valid.dtype!=bool or valid.shape!=times.shape or lv.dtype!=bool or lv.shape!=times.shape or ref.shape!=times.shape
                        or not np.isin(ref[lv],[0,1]).all()):raise ValueError('Invalid Real frame layout')
                    ends,xr=case_windows(raw,valid) if len(raw)>=150 else (np.empty(0,np.int64),np.empty((0,26,150),np.float32))
                    ff=b0.stats(xr) if len(xr) else np.empty((0,130),np.float32)
                    if not np.isfinite(ff).all():raise ValueError('Nonfinite valid frame features')
                    path=npz(LOCAL/'features'/key,timestamp_us=times,reference_state=ref,row_label_valid=lv,ends=ends,features=ff)
                    cache_paths[uid]=path
                    if i%50==0:note('FRAME_FEATURES',i)
                meta=eligible.set_index('case_uid')
                def score_events(model,name,uids,thresholds):
                    rows=[]
                    for uid in uids:
                        with np.load(cache_paths[uid],allow_pickle=False) as z:arrays={k:z[k] for k in z.files}
                        p=np.full(len(arrays['timestamp_us']),np.nan,float);p[arrays['ends']]=predict(model,arrays['features'])
                        npz(LOCAL/'scores'/name/(uid.replace('/','__')+'.npz'),probabilities=p)
                        record_meta={'case_uid':uid,**meta.loc[uid].to_dict()}
                        for th in thresholds:rows.append(event_record(record_meta,arrays,p,th))
                    return rows
                rows_by_arm={a:[] for a in (*ARMS,'S')}
                stage='nested_events'
                for k in range(5):
                    for arm in ARMS:
                        inner=[]
                        for j in range(5):
                            if j==k:continue
                            tr=~frame.dev_fold.isin([k,j]).to_numpy()
                            model,path=fit(arm,tr,f'inner_{arm}_outer{k}_held{j}',[k,j])
                            inner.extend(score_events(model,path.stem,eligible.loc[eligible.dev_fold.eq(j),'case_uid'],GRID));del model
                        # Inner observations alone determine this outer onset.
                        pd.DataFrame(inner).to_csv(LOCAL/'inner_events'/f'{arm}_outer{k}.csv',index=False);record(LOCAL/'inner_events'/f'{arm}_outer{k}.csv')
                        grid_rows=[{'onset_th':th,**event_summary([row for row in inner if row['onset_th']==th])} for th in GRID]
                        choice=select_onset(grid_rows);choice.update(arm=arm,outer_fold=k)
                        selections.append(choice);scans.append({'arm':arm,'outer_fold':k,'grid':grid_rows})
                        selection_rows=[{'arm':v['arm'],'outer_fold':v['outer_fold'],'onset_th':v['onset_th'],
                            'no_feasible_onset':v['no_feasible_onset'],'tied_onsets_primary':json.dumps(v['tied_onsets_primary']),
                            'tied_onsets_after_detection':json.dumps(v['tied_onsets_after_detection']),
                            'inner_N1_rate':v['inner']['N1']['rate'],'inner_N2_rate':v['inner']['N2']['rate'],
                            'inner_motor':v['inner']['motor']['detection'],'inner_sensor':v['inner']['sensor']['detection']} for v in selections]
                        pd.DataFrame(selection_rows).to_csv(OUT/'onset_selection.csv',index=False)
                        path=full_models[(arm,k)]
                        if b0.sha(path)!=manifest[path.relative_to(LOCAL).as_posix()]:raise ValueError('Outer model identity changed')
                        with path.open('rb') as f:model=pickle.load(f)
                        rows_by_arm[arm].extend(score_events(model,path.stem,eligible.loc[eligible.dev_fold.eq(k),'case_uid'],[choice['onset_th']]))
                        del model;note('OUTER_EVENT_COMPLETE',arm,k,choice['onset_th'],choice['no_feasible_onset'])
                rows_by_arm['S']=score_events(source_model,'frozen_S',eligible.case_uid,[.95])
                for arm,rows in rows_by_arm.items():
                    for row in rows:row['arm']=arm
                pd.DataFrame([row for rows in rows_by_arm.values() for row in rows]).to_csv(OUT/'events_per_case.csv',index=False)
                b0.save(OUT/'inner_selection.json',{'selections':selections,'scans':scans})
                event_point={arm:event_summary(rows) for arm,rows in rows_by_arm.items()}
                b0.save(OUT/'event_points.json',event_point)
                stage='event_bootstrap'
                event_ci=event_bootstrap(rows_by_arm,eligible);b0.save(OUT/'event_CI.json',event_ci)
                stage='paired_case_AUC'
                paired=[]
                for arm in ARMS:
                    p=probabilities[(arm,1.,0)]
                    for uid,part in frame[frame.family.isin(['motor','sensor'])].groupby('case_uid',sort=True):
                        mask=frame.case_uid.eq(uid).to_numpy() & ((y==1)|masks['pre_fault'])
                        a=b0.weighted_auc(b0.auc_plan(p,y,mask),np.ones(len(y),int))
                        paired.append({'arm':arm,'case_uid':uid,'family':part.family.iloc[0],'AUC':a,
                            'fault_windows':int((frame.case_uid.eq(uid).to_numpy()&(y==1)).sum()),
                            'pre_fault_windows':int((frame.case_uid.eq(uid).to_numpy()&masks['pre_fault']).sum())})
                pd.DataFrame(paired).to_csv(OUT/'paired_case_AUC.csv',index=False)
                paired_summary={}
                for arm in ARMS:
                    paired_summary[arm]={}
                    for family in ('motor','sensor'):
                        values=[v['AUC'] for v in paired if v['arm']==arm and v['family']==family and v['AUC'] is not None]
                        n=sum(v['arm']==arm and v['family']==family for v in paired)
                        paired_summary[arm][family]={'cases':n,'defined':len(values),'undefined':n-len(values),
                            'median':float(np.median(values)) if values else None,'q25':float(np.quantile(values,.25)) if values else None,
                            'q75':float(np.quantile(values,.75)) if values else None,'min':float(min(values)) if values else None,'max':float(max(values)) if values else None}
                by_flight={arm:{f:event_summary([row for row in rows if row['flight']==f]) for f in sorted(eligible.flight.unique())} for arm,rows in rows_by_arm.items()}
                verdicts={arm:h2(event_point[arm]) for arm in ARMS};h2_any=any(v['holds'] for v in verdicts.values());h1_holds=h1['fractions']['0.25']['holds']
                if b0.sha(b0.SOURCE)!=b0.SOURCE_SHA or b0.sha(b0.SPLIT)!=b0.SPLIT_SHA:raise ValueError('Frozen inputs changed')
                assert len(fits)==200 and len(selections)==10
                result={'status':'COMPLETE','protocol':str(CFG.relative_to(ROOT)),'config_sha256':b0.sha(CFG),'startup':startup,
                    'gate':gate,'recipe':b0.RECIPE,'fractions':FRACTIONS,'window_metrics':window_rows,'S_window_reference':source_window,
                    'H1':h1,'onset_selection':selections,'event':event_point,'event_CI':event_ci,'H2':{'arms':verdicts,'line_holds':h2_any},
                    'descriptive':{'by_flight':by_flight,'paired_case_AUC':paired_summary},'consequences':consequences(h1_holds,h2_any),
                    'fit_count':len(fits),'runtime_root':str(LOCAL),'access':access_summary(),'wall_time_s':time.perf_counter()-started,
                    'no_sealed_internal_test_or_GPU':True,'no_automatic_B2':True}
                b0.save(OUT/'results.json',result)
                b0.save(OUT/'receipt.json',{'status':'COMPLETE','config_sha256':b0.sha(CFG),'split_sha256':b0.SPLIT_SHA,
                    'S_model_sha256':b0.SOURCE_SHA,'runtime_root':str(LOCAL),'local_artifacts_sha256':manifest,
                    'fit_count':len(fits),'fits':fits,'selections':selections,'access':access_summary(),
                    'results_sha256':b0.sha(OUT/'results.json'),'wall_time_s':time.perf_counter()-started})
                note('COMPLETE',json.dumps(result['H2']),result['consequences']['key'])
                return 0
        except Exception as exc:
            note('FAILED_STOP',stage,repr(exc))
            b0.save(OUT/'failure.json',{'stage':stage,'error':repr(exc),'completed_fits':len(fits),'local_artifacts_sha256':manifest,
                'access':access_summary(),'no_retry':True})
            raise


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8');sys.stderr.reconfigure(encoding='utf-8')
    raise SystemExit(run())
