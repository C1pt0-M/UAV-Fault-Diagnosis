"""Verify the full causal publication-replay dataset without training or tuning.

All persisted case frames and provenance are checked; raw source replay parity
uses fixed representative train/val cases only. No relabeling or threshold edits.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
import heapq
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from pyulog import ULog

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/data'));sys.path.insert(0,str(ROOT/'packages/uav_fault'))
from build_full_causal_input import OUT,ART,OLD,read_json,check,case_key,extract_group,transform_raw,records_for,now
from audit_rflymad_raw_sources import digest
from uav_fault.causal_pipeline import GROUPS,CausalInputStream
from uav_fault.causal_input import canonical_window


def save(path,d):
    with path.open('x',encoding='utf-8') as f:json.dump(d,f,ensure_ascii=False,indent=2,allow_nan=False)


def verify_case(task):
    record=read_json(OUT/'case_records'/(case_key(task['case_uid'])+'.json'));policy=read_json(OUT/'input_contract.json')
    for p,h in record['output_identities'].items():check(Path(p),h)
    check(Path(task['label_file']),task['label_sha256']);check(Path(task['original_csv']),task['original_csv_sha256'])
    with np.load(record['frame_file'],allow_pickle=False) as saved:
        d={key:saved[key] for key in saved.files}
        q=d['timestamp_us'];matrix=d['canonical_rows'];n=len(q)
        if matrix.shape!=(n,26) or not np.all(np.diff(q)==20000):raise ValueError('Grid/shape mismatch')
        with np.load(task['label_file'],allow_pickle=False) as lab:
            if not np.array_equal(q,lab['timestamp_us']):raise ValueError('Original crop changed')
            ref=lab['ros_state'] if task['domain']=='REAL' else lab['gtd_state_reference']
            historical=lab['historical_state'] if task['domain']=='REAL' else lab['existing_state']
            valid_label=lab['row_label_valid'] if task['domain']=='REAL' else lab['source_references_agree']
            if not np.array_equal(ref,d['reference_state']) or not np.array_equal(historical,d['historical_state']) or not np.array_equal(valid_label,d['row_label_valid']):
                raise ValueError('Revised labels changed during input repair')
        all_good=np.isfinite(matrix).all(axis=1)
        for group,(_,_,fields) in GROUPS.items():
            selected=d[group+'__source_index'];obs=d[group+'__observation_us'];avail=d[group+'__available_us'];age=d[group+'__age_us'];reason=d[group+'__reason'];valid=d[group+'__valid']
            found=selected>=0
            if np.any(obs[found]>avail[found]) or np.any(avail[found]>q[found]):raise ValueError('Future source provenance')
            if not np.array_equal(age[found],q[found]-obs[found]):raise ValueError('Age mismatch')
            if np.any(age[valid]>policy['max_age_us'][group]):raise ValueError('Stale source marked good')
            cols=[policy['columns'].index(name) for name,_ in fields]
            if not np.isfinite(matrix[valid][:,cols]).all():raise ValueError('Valid topic nonfinite')
            if not np.isnan(matrix[~valid][:,cols]).all():raise ValueError('Invalid topic silently filled')
            if group not in record['group_errors']:
                if not np.array_equal(valid,reason==0):raise ValueError('Topic mask/reason mismatch')
            else:
                if valid.any() or not np.all(reason==8):raise ValueError('Clock failure not quarantined')
            all_good &= valid
        if task['clock_review']:all_good[:]=False
        if not np.array_equal(all_good,d['row_input_valid']):raise ValueError('Combined input mask mismatch')
        starts=np.asarray(task['selected']['start'],dtype=np.int64);ends=starts+150;indices=np.asarray(task['selected']['index'],dtype=np.int64)
        if not np.array_equal(starts,d['window_start_row']) or not np.array_equal(indices,d['original_cache_index']):raise ValueError('Selected indices changed')
        input_mask=np.asarray([all_good[s:e].all() for s,e in zip(starts,ends)])
        mask=input_mask & valid_label[ends-1] & np.isin(ref[ends-1],[0,1])
        if not np.array_equal(mask,d['supervised_valid']) or not np.array_equal(input_mask,d['input_window_valid']):raise ValueError('Window eligibility mismatch')
        with np.load(record['window_file'],allow_pickle=False) as w:
            if not np.array_equal(w['original_cache_index'],indices[mask]) or not np.array_equal(w['y'],ref[ends[mask]-1]):raise ValueError('Compact window identities mismatch')
            # Independently rebuild derivative/relative fields and compare all selected tensors.
            for j,start in enumerate(starts[mask]):
                raw=matrix[start:start+150].copy();columns=policy['columns']
                accel=raw[:,[columns.index('accel_'+a) for a in 'xyz']]
                raw[:,columns.index('acc_norm')]=np.sqrt((accel*accel).sum(axis=1))
                b=columns.index('baro_alt_relative');raw[:,b]-=raw[0,b]
                if not np.array_equal(raw.T.astype(np.float32),w['x'][j]):raise ValueError('Saved causal window transform mismatch')
        return {'case_uid':task['case_uid'],'split':task['split'],'windows':len(starts),'eligible':int(mask.sum()),'rows':n}


def verify_split(split):
    tasks=[t for t in read_json(OUT/'tasks.json') if t['split']==split]
    results=[]
    with ThreadPoolExecutor(max_workers=3) as pool:
        for i,r in enumerate(pool.map(verify_case,tasks),1):
            results.append(r)
            if i%250==0:print('FULL_CAUSAL_VERIFIED_CASES',split,i,'/',len(tasks),flush=True)
    torch.set_num_threads(2)
    info=read_json(OUT/(split+'_cache_complete.json'));check(Path(info['cache_file']),info['cache_sha256'])
    stats=read_json(OUT/'norm_stats.json');policy=read_json(OUT/'input_contract.json')
    cache=torch.load(info['cache_file'],weights_only=True,map_location='cpu',mmap=True)
    if cache['data_version']!=read_json(OUT/'RUNNING.json')['version'] or cache['norm_sha256']!=digest(OUT/'norm_stats.json') or cache['input_contract_sha256']!=digest(OUT/'input_contract.json'):
        raise ValueError('Cache version/normalization mismatch')
    ledger=pd.read_csv(OUT/'ledgers'/(split+'_all_windows.csv'),keep_default_na=False)
    if not np.array_equal(ledger.original_cache_index,np.arange(len(ledger))):raise ValueError('Original full denominator lost')
    excluded=pd.read_csv(OUT/'ledgers'/(split+'_excluded_windows.csv'),keep_default_na=False)
    included=pd.read_csv(OUT/'ledgers'/(split+'_included_windows.csv'),keep_default_na=False)
    if set(excluded.original_cache_index)&set(included.original_cache_index) or len(excluded)+len(included)!=len(ledger):raise ValueError('Ledgers fail partition')
    oldname={'train':'train.pt','val':'val.pt','test':'test.pt','real':'real_holdout_test.pt'}[split]
    historical=torch.load(OLD/oldname,weights_only=True,map_location='cpu',mmap=True)
    if not np.array_equal(historical['y'].numpy(),ledger.historical_y):raise ValueError('Original cache order or y changed')
    compact=split in ['train','val']
    if compact:
        if not np.array_equal(cache['original_cache_index'].numpy(),included.original_cache_index):raise ValueError('Compact index order mismatch')
        if not np.array_equal(cache['y'].numpy(),included.qualified_y):raise ValueError('Compact labels mismatch')
    else:
        if 'y' in cache:raise ValueError('Evidence must not masquerade as generic binary y')
        if not np.array_equal(cache['supervised_valid'].numpy(),ledger.supervised_valid):raise ValueError('Evidence mask mismatch')
        if not np.array_equal(cache['qualified_reference_y'].numpy(),ledger.qualified_y):raise ValueError('Evidence target mismatch')
        if split=='real':
            roles={'adapt_normal':0,'calibrate_normal':1,'test_normal':2,'test_fault':3,'supplemental_normal_test':4}
            if not np.array_equal(cache['role_id'].numpy(),ledger.role.map(roles)):raise ValueError('Real role mismatch')
    # Every eligible cache row compared to its per-case selected raw source tensor.
    for r in records_for(split):
        with np.load(r['window_file'],allow_pickle=False) as d:
            indices=d['original_cache_index'];rows=ledger.loc[indices,'compact_index'].to_numpy(dtype=np.int64) if compact else indices
            if not np.array_equal(cache['x'][rows].numpy(),d['x']):raise ValueError('Cache X not same causal source inputs')
    mu=torch.tensor(stats['mean'],dtype=torch.float32).view(1,-1,1);std=torch.tensor(stats['std'],dtype=torch.float32).view(1,-1,1)
    for start in range(0,len(cache['x']),256):
        x=cache['x'][start:start+256];mask=np.ones(len(x),dtype=bool) if compact else cache['supervised_valid'][start:start+256].numpy()
        if not torch.isfinite((x[mask]-mu)/std).all():raise ValueError('Normalized cache nonfinite')
        if not compact and not torch.isnan(x[~mask]).all():raise ValueError('Invalid evidence X not explicitly unavailable')
    result={'status':'PASS_FULL_CAUSAL_SPLIT','split':split,'cases':len(tasks),'rows':sum(r['rows'] for r in results),
        'original_windows':len(ledger),'eligible_windows':len(included),'excluded_windows':len(excluded),
        'all_saved_frames_provenance_masks_checked':True,'all_selected_causal_X_exactly_verified':True,
        'original_denominator_and_label_identity_preserved':True,'normalization_and_role_checks_passed':True,
        'model_or_training_used':False,'verifier_sha256':digest(__file__),'completed_at':now()}
    save(ART/('verification_'+split+'.json'),result);print(json.dumps(result,ensure_ascii=False),flush=True)


def verify_norm():
    records=records_for('train');n=0;sums=np.zeros(26);squares=np.zeros(26);windows=0
    for r in records:
        with np.load(r['window_file'],allow_pickle=False) as d:
            x=d['x'].transpose(0,2,1).reshape(-1,26).astype(np.float64);n+=len(x);windows+=len(d['y'])
            sums+=x.sum(axis=0);squares+=(x*x).sum(axis=0)
    stats=read_json(OUT/'norm_stats.json');mean=sums/n;std=np.sqrt(np.maximum(squares/n-mean*mean,0));std[std==0]=1
    if stats['fit_windows']!=windows or stats['fit_frame_contributions']!=n:raise ValueError('Norm fit denominator wrong')
    # Independent sum/square check vs stable central moments: roundoff allowance,
    # not tensor parity allowance. Relative tol needed for 1e3-scale PWM means.
    if not np.allclose(mean,stats['mean'],rtol=1e-10,atol=1e-9) or not np.allclose(std,stats['std'],rtol=1e-7,atol=1e-7):raise ValueError('Independent source-train moments differ')
    result={'status':'PASS_SOURCE_TRAIN_ONLY_NORM','fit_windows':windows,'fit_frame_contributions':n,
        'mean_max_abs_diff':float(np.max(np.abs(mean-np.asarray(stats['mean'])))),
        'std_max_abs_diff':float(np.max(np.abs(std-np.asarray(stats['std'])))),
        'fit_scope':'Only eligible original cap64 source train windows, no suffix/val/test/Real',
        'norm_sha256':digest(OUT/'norm_stats.json'),'verifier_sha256':digest(__file__)}
    save(ART/'verification_norm.json',result);print(json.dumps(result),flush=True)


def verify_parity():
    # Fixed previous pilot representatives: source normal train per domain,
    # normal val per domain and both pre-existing train/val long-gap regressions.
    selected=read_json(ROOT/'artifacts/causal_input_parity_v1_20261031/selection.json')
    uids=[selected['fit'][0],selected['fit'][6],*selected['validation']]
    all_tasks={t['case_uid']:t for t in read_json(OUT/'tasks.json')};policy=read_json(OUT/'input_contract.json');norm=read_json(OUT/'norm_stats.json')
    compared=0;results=[]
    for uid in uids:
        task=all_tasks[uid];record=read_json(OUT/'case_records'/(case_key(uid)+'.json'))
        log=ULog(record['raw_ulog'],message_name_filter_list=sorted({g[0] for g in GROUPS.values()}))
        with np.load(record['frame_file'],allow_pickle=False) as saved:
            d={key:saved[key] for key in saved.files}
            grid=d['timestamp_us'];traces={};events=[]
            for order,(g,(topic,_,_)) in enumerate(GROUPS.items()):
                data=next(item.data for item in log.data_list if item.name==topic and item.multi_id==0)
                traces[g],offset=extract_group(data,g,grid)
                for j,t in enumerate(traces[g].available_us):events.append((int(t),order,j,g))
            events.sort();cursor=0;runner=CausalInputStream(policy['columns'],policy['max_age_us'],origin_us=int(grid[0]),mean=norm['mean'],std=norm['std'])
            selected_starts=set(int(v) for v in d['window_start_row']);matched=0;invalid=0
            for i,tick in enumerate(grid):
                while cursor<len(events) and events[cursor][0]<=tick:
                    available,_,j,g=events[cursor];tr=traces[g];runner.push(g,int(tr.observation_us[j]),available,tr.values[j]);cursor+=1
                frame=runner.emit(int(tick))
                if frame['input_valid']!=bool(d['row_input_valid'][i]) or not np.array_equal(frame['values'],d['canonical_rows'][i],equal_nan=True):raise ValueError('Full version streaming rows differ')
                for g,meta in frame['provenance'].items():
                    for k,v in meta.items():
                        if v!=d[g+'__'+k][i]:raise ValueError('Stream provenance mismatch')
                if frame['window_due'] and frame['window_start_row'] in selected_starts:
                    start=frame['window_start_row'];j=np.flatnonzero(d['window_start_row']==start)[0]
                    if frame['window_input_valid']!=bool(d['input_window_valid'][j]):raise ValueError('Stream selected mask mismatch')
                    if frame['window_input_valid']:
                        expected=canonical_window(d['canonical_rows'][start:start+150],policy['columns'],d['row_input_valid'][start:start+150],mean=norm['mean'],std=norm['std'])
                        if not np.array_equal(expected,frame['window']):raise ValueError('Full normalized stream mismatch')
                        matched+=1
                    else:invalid+=1
            compared+=matched;results.append({'case_uid':uid,'normalized_input_valid_windows_equal':matched,'selected_unavailable_windows_retained':invalid})
    result={'status':'PASS_FULL_VERSION_SOURCE_STREAM_PARITY','cases':results,'exact_equal_normalized_windows':compared,
        'tensor_tolerance':0,'norm_sha256':digest(OUT/'norm_stats.json'),'actual_live_transport_tested':False,'verifier_sha256':digest(__file__)}
    save(ART/'verification_parity.json',result);print(json.dumps(result),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',choices=['split','norm','parity'],required=True);p.add_argument('--split',choices=['train','val','test','real'],default='train');a=p.parse_args()
    if a.stage=='split':verify_split(a.split)
    elif a.stage=='norm':verify_norm()
    else:verify_parity()


if __name__=='__main__':main()
