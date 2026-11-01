"""Full fixed-rule 26-channel publication-replay input build. No model work.

Keep original crops/splits/cap64 identities; separate input from label masks.
Original data and label repairs are read-only. Completed case shards are verified
on resume. Identity failures are fatal; raw clock/topic failures are explicit
input-unavailable evidence, not repaired by sorting, interpolation or zero fill.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd
from pyulog import ULog

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'packages/uav_fault'))
sys.path.insert(0, str(ROOT / 'scripts/research'))
from uav_fault.causal_pipeline import GROUPS, TopicTrace, batch_frames
from uav_fault.causal_input import canonical_window, window_input_valid
from audit_rflymad_raw_sources import digest

OUT = Path('D:/UAV-Data/causal_input_v1')
ART = ROOT / 'artifacts/causal_input_full_v1_20261031'
OLD = Path('D:/UAV-Data/online_cache_v2')
SOURCE = Path('D:/UAV-Data/online_cache_v2_labeltime_v1')
TEST = Path('D:/UAV-Data/internal_test_labeltime_v1')
REAL = Path('D:/UAV-Data/real_labeltime_v1')
PRIOR = ROOT / 'artifacts/fixed_crop_label_validation_20261031'
VERSION = 'rflymad_causal_publication_replay_v1'
CLOCK_ERROR = 8


def now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def jsonl(path):
    return [json.loads(s) for s in Path(path).read_text(encoding='utf-8').splitlines()]


def save(path, data):
    path = Path(path)
    if path.exists(): raise FileExistsError('Refuse overwrite: ' + str(path))
    temporary = path.with_name(path.name + '.partial')
    with temporary.open('x', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2, allow_nan=False)
    temporary.replace(path)


def check(path, expected):
    if digest(path) != expected: raise ValueError('IDENTITY_MISMATCH: ' + str(path))


def case_key(uid):
    return uid.replace('/', '__')


def initialize():
    if OUT.exists(): raise FileExistsError('Independent output required')
    if shutil.disk_usage(OUT.parent).free < 30_000_000_000: raise ValueError('Not enough free disk')
    policy = read_json(ROOT / 'artifacts/causal_input_parity_v1_20261031/input_policy.json')
    src = {r['case_uid']: r for r in jsonl(PRIOR / 'case_results.jsonl')}
    prior_real = {r['case_uid']: r for r in jsonl(ROOT / 'artifacts/real_labeltime_audit_20261031/case_results.jsonl')}
    tasks = []
    for r in jsonl(SOURCE / 'case_repair_results.jsonl'):
        p = src[r['case_uid']]; ref = PRIOR / 'label_references' / (case_key(r['case_uid']) + '.npz')
        tasks.append({'case_uid': r['case_uid'], 'domain': r['domain'], 'split': r['split'],
            'role': r['split'], 'family': r['level2_label'], 'group': r.get('simulation_pair_key', ''),
            'raw_dir': r['source_path'], 'raw_ulog_sha256': p['original_ulog_sha256'],
            'original_csv': r['historical_final_output_path'], 'original_csv_sha256': p['processed_csv_sha256'],
            'label_file': str(ref), 'label_sha256': p['sidecar_sha256'], 'clock_review': p['input_clock_review_required']})
    for part in range(1,6):
        for r in jsonl(TEST / 'parts' / ('part' + str(part) + '.jsonl')):
            logs = [(p,h) for p,h in r['original_identities'].items() if p.endswith('.ulg')]
            if len(logs) != 1: raise ValueError('Missing original test ULog identity')
            tasks.append({'case_uid': r['case_uid'], 'domain': r['domain'], 'split': 'test', 'role': 'test',
                'family': r['level2_label'], 'group': r.get('simulation_pair_key',''), 'raw_dir': r['source_path'],
                'raw_ulog_sha256': logs[0][1], 'original_csv': r['historical_final_output_path'],
                'original_csv_sha256': r['output_sha256'], 'label_file': r['reference_file'],
                'label_sha256': r['sidecar_sha256'], 'clock_review': r['original_inertial_clock_review']})
    for r in jsonl(REAL / 'case_repair_results.jsonl'):
        p = prior_real[r['case_uid']]
        raw = Path('D:/RflyMAD/Real').joinpath(*r['case_uid'].split('/')[1:])
        tasks.append({'case_uid': r['case_uid'], 'domain': 'REAL', 'split': 'real', 'role': r['role'],
            'family': r['level2_label'], 'group': r['association_group'], 'raw_dir': str(raw),
            'raw_ulog_sha256': p['raw_ulog_sha256'], 'original_csv': r['historical_final_output_path'],
            'original_csv_sha256': r['output_sha256'], 'label_file': r['reference_file'],
            'label_sha256': r['reference_sha256'], 'clock_review': False, 'label_quarantine': r['quarantined']})
    expected = {'train': 2580, 'val': 555, 'test': 553, 'real': 465}
    if Counter(t['split'] for t in tasks) != expected or len({t['case_uid'] for t in tasks}) != 4153:
        raise ValueError('Frozen scope mismatch')
    links = {}
    for split in ['train','val']:
        f = pd.read_csv(ROOT / 'artifacts/source_training_coverage_audit_20261030_r2' / (split+'_source_window_links.csv'))
        for uid,g in f.groupby('case_uid',sort=False):
            links[uid] = {'index': g.cache_index.tolist(), 'start': g.window_start_row.tolist(), 'end': g.window_end_row.tolist(), 'old': g.label.tolist()}
    for split,path,old_col in [('test',TEST/'test_window_links.csv','existing_label'),('real',REAL/'real_cap64_window_links.csv','historical_label')]:
        f = pd.read_csv(path)
        for uid,g in f.groupby('case_uid',sort=False):
            links[uid] = {'index': g.source_cache_index.tolist(), 'start': g.window_start_row.tolist(), 'end': g.window_end_row.tolist(), 'old': g[old_col].tolist()}
    for t in tasks:
        t['selected'] = links[t['case_uid']]
        if not 1 <= len(t['selected']['start']) <= 64: raise ValueError('Invalid cap64 original selection')
    paths = [ART/'plan.json', ROOT/'configs/experiments/causal_input_contract_v1.json',
        ROOT/'artifacts/causal_input_parity_v1_20261031/input_policy.json',
        SOURCE/'repair_manifest.json', TEST/'repair_manifest.json', REAL/'repair_manifest.json',
        Path('D:/UAV-Data/splits/rflymad_case_split_v2/rflymad_case_level_split_manifest.csv'),
        ROOT/'packages/uav_fault/uav_fault/causal_input.py', ROOT/'packages/uav_fault/uav_fault/causal_pipeline.py']
    paths += [OLD/name for name in ['train.pt','val.pt','test.pt','real_holdout_test.pt','norm_stats.json']]
    identities = {str(p): digest(p) for p in paths}
    OUT.mkdir(parents=True,exist_ok=False)
    for folder in ['cases','case_records','windows','cache','ledgers']: (OUT/folder).mkdir()
    shutil.copyfile(ART/'plan.json',OUT/'plan.json')
    save(OUT/'input_contract.json', {'version':VERSION,'scope':'Full original-ULog publication replay; no live deployment clearance',
        'columns':policy['columns'],'max_age_us':policy['max_age_us'],'original_sampling':'original cap64, no refill',
        'training_enabled':False,'deployment_enabled':False,'clock':'Original ULog publication proxy; sample time where supplied'})
    save(OUT/'tasks.json',tasks)
    save(OUT/'RUNNING.json', {'started_at':now(),'version':VERSION,'script_sha256':digest(__file__),
        'tasks_sha256':digest(OUT/'tasks.json'),'identities':identities,'expected_cases':expected})
    print('INITIALIZED_FULL_CAUSAL',expected,flush=True)


def extract_group(data, group, grid):
    _, clock, fields = GROUPS[group]
    pub = np.asarray(data['timestamp'])
    if pub.ndim != 1 or not np.issubdtype(pub.dtype,np.integer): raise ValueError('noninteger_publication_clock')
    pub = pub.astype(np.int64)
    if (pub < 0).any() or (pub[1:] < pub[:-1]).any(): raise ValueError('publication_clock_reversal')
    start = max(int(np.searchsorted(pub,grid[0],side='right'))-1,0)
    end = int(np.searchsorted(pub,grid[-1],side='right'))
    available = pub[start:end]
    if clock == 'timestamp': obs = available.copy()
    else:
        if clock not in data: raise ValueError('missing_observation_clock:' + clock)
        raw = np.asarray(data[clock])
        if raw.ndim != 1 or not np.issubdtype(raw.dtype,np.integer): raise ValueError('noninteger_sample_clock')
        raw = raw[start:end].astype(np.int64)
        if clock == 'accelerometer_timestamp_relative':
            if (raw==2147483647).any(): raise ValueError('invalid_accel_clock_sentinel')
            obs = available + raw
        else: obs = raw
    if (obs<=0).any() or (obs>available).any() or (obs[1:]<obs[:-1]).any(): raise ValueError('sample_clock_invalid_or_reversed')
    values = np.column_stack([data[field][start:end] for _,field in fields]).astype(np.float64)
    return TopicTrace(obs,available,values), start


def transform_raw(matrix, starts, eligible_mask, columns):
    # Use the already tested shared window transformation; zero/one are an
    # explicit identity transform here, not production normalization defaults.
    out = np.empty((int(eligible_mask.sum()),26,150),dtype=np.float32)
    for i,start in enumerate(starts[eligible_mask]):
        out[i] = canonical_window(matrix[start:start+150],columns,np.ones(150,dtype=bool),mean=np.zeros(26),std=np.ones(26))
    return out


def process_case(task):
    policy = read_json(OUT/'input_contract.json');columns=policy['columns'];limits=policy['max_age_us']
    uid=task['case_uid'];key=case_key(uid);marker=OUT/'case_records'/(key+'.json')
    if marker.exists():
        r=read_json(marker)
        for p,h in r['output_identities'].items(): check(Path(p),h)
        if r['task_sha256'] != hashlib.sha256(json.dumps(task,sort_keys=True).encode()).hexdigest(): raise ValueError('Resume task changed')
        return {'case_uid':uid,'resumed':True,**r['summary']}
    check(Path(task['label_file']),task['label_sha256']);check(Path(task['original_csv']),task['original_csv_sha256'])
    with np.load(task['label_file'],allow_pickle=False) as d:
        grid=d['timestamp_us'].copy().astype(np.int64)
        if task['domain']=='REAL':
            historical=d['historical_state'].copy();reference=d['ros_state'].copy();label_valid=d['row_label_valid'].copy()
        else:
            historical=d['existing_state'].copy();reference=d['gtd_state_reference'].copy();label_valid=d['source_references_agree'].copy()
    if len(grid)<150 or not np.all(np.diff(grid)==20000): raise ValueError('Saved original crop grid invalid')
    original_grid=pd.read_csv(task['original_csv'],usecols=['Timestamp']).Timestamp.to_numpy(dtype=np.float64)
    if not np.array_equal(original_grid,grid): raise ValueError('Saved reference crop not original CSV')
    raw=Path(task['raw_dir']);logs=list(raw.rglob('*.ulg')) if task['domain']=='REAL' else list((raw/'Log').glob('*.ulg'))
    if len(logs)!=1: raise ValueError('Original ULog identity ambiguous')
    check(logs[0],task['raw_ulog_sha256']);stat=(logs[0].stat().st_size,logs[0].stat().st_mtime_ns)
    traces={};group_error={};offsets={}
    try:
        log=ULog(str(logs[0]),message_name_filter_list=sorted({g[0] for g in GROUPS.values()}))
    except Exception as exc:
        log=None;group_error={g:'ULog_decode:'+type(exc).__name__+':'+str(exc) for g in GROUPS}
    for group,(topic,_,fields) in GROUPS.items():
        try:
            if log is None: raise ValueError(group_error[group])
            found=[d.data for d in log.data_list if d.name==topic and d.multi_id==0]
            if len(found)!=1: raise ValueError('missing_or_ambiguous_topic:'+topic)
            traces[group],offsets[group]=extract_group(found[0],group,grid)
        except (ValueError,KeyError,IndexError) as exc:
            group_error[group]=str(exc)
            traces[group]=TopicTrace(np.empty(0,dtype=np.int64),np.empty(0,dtype=np.int64),np.empty((0,len(fields))))
            offsets[group]=0
    matrix,valid,provenance=batch_frames(traces,grid,columns,limits)
    for group in group_error: provenance[group]['reason'][:]=CLOCK_ERROR
    if task['clock_review']: valid[:]=False
    starts=np.asarray(task['selected']['start'],dtype=np.int64);ends=np.asarray(task['selected']['end'],dtype=np.int64)
    old=np.asarray(task['selected']['old'],dtype=np.int64);original_indices=np.asarray(task['selected']['index'],dtype=np.int64)
    if not np.all(ends-starts==150) or (starts<0).any() or (ends>len(grid)).any() or not np.array_equal(old,historical[ends-1]):
        raise ValueError('Original selected window identity/old label differs')
    all_starts=np.arange(0,len(grid)-149,5,dtype=np.int64)
    picked=all_starts[np.unique(np.linspace(0,len(all_starts)-1,64).round().astype(int))] if len(all_starts)>64 else all_starts
    if not np.array_equal(starts,picked): raise ValueError('Changed cap64 original selection')
    input_valid=window_input_valid(valid,starts,window_frames=150)
    supervised=input_valid & label_valid[ends-1] & np.isin(reference[ends-1],[0,1])
    raw_x=transform_raw(matrix,starts,supervised,columns)
    whole_labels=window_input_valid(label_valid,starts,window_frames=150)
    arrays={'timestamp_us':grid,'canonical_rows':matrix,'row_input_valid':valid,
        'historical_state':historical,'reference_state':reference,'row_label_valid':label_valid,
        'window_start_row':starts,'window_end_row':ends,'original_cache_index':original_indices,
        'historical_y':old,'reference_y':reference[ends-1],'label_endpoint_valid':label_valid[ends-1],
        'input_window_valid':input_valid,'supervised_valid':supervised,'whole_window_label_valid':whole_labels}
    for group,p in provenance.items():
        for name,value in p.items(): arrays[group+'__'+name]=value
        arrays[group+'__source_index_original']=np.where(p['source_index']>=0,p['source_index']+offsets[group],-1)
    framepath=OUT/'cases'/(key+'.npz');windowpath=OUT/'windows'/(key+'.npz')
    for path in [framepath,windowpath]:
        if path.exists():
            interrupted=OUT/'interrupted_files'/path.parent.name/path.name;interrupted.parent.mkdir(parents=True,exist_ok=True)
            if interrupted.exists(): raise FileExistsError('Unregistered output snapshot exists')
            path.replace(interrupted)
    np.savez_compressed(framepath,**arrays)
    np.savez_compressed(windowpath,x=raw_x,y=reference[ends[supervised]-1].astype(np.int64),
        original_cache_index=original_indices[supervised],window_start_row=starts[supervised])
    reason_counts=Counter();window_reasons=[]
    for i,(s,e) in enumerate(zip(starts,ends)):
        reasons=[]
        for g in GROUPS:
            flags=provenance[g]['reason'][s:e]
            for flag,label in [(1,'missing'),(2,'stale'),(4,'nonfinite'),(CLOCK_ERROR,'clock_or_topic_review')]:
                if (flags&flag).any(): reasons.append(g+':'+label)
        if task['clock_review']: reasons.append('known_original_inertial_clock_review')
        if not label_valid[e-1] or reference[e-1]<0: reasons.append('endpoint_label_uncertain')
        if not input_valid[i] and not any(':'+x in '|'.join(reasons) for x in ['missing','stale','nonfinite','clock_or_topic_review']) and not task['clock_review']:
            reasons.append('derived_input_nonfinite')
        window_reasons.append('|'.join(reasons))
        if not supervised[i]: reason_counts.update(reasons)
    if (logs[0].stat().st_size,logs[0].stat().st_mtime_ns)!=stat: raise ValueError('Raw input changed during rebuild')
    summary={'split':task['split'],'rows':len(grid),'original_windows':len(starts),'input_eligible_windows':int(input_valid.sum()),
        'supervised_eligible_windows':int(supervised.sum()),'excluded_windows':int((~supervised).sum()),
        'clock_or_topic_review_groups':len(group_error),'eligible_normal':int((reference[ends[supervised]-1]==0).sum()),
        'eligible_fault':int((reference[ends[supervised]-1]==1).sum())}
    record={'case_uid':uid,'task_sha256':hashlib.sha256(json.dumps(task,sort_keys=True).encode()).hexdigest(),
        'domain':task['domain'],'split':task['split'],'role':task['role'],'family':task['family'],'group':task['group'],
        'frame_file':str(framepath),'window_file':str(windowpath),'summary':summary,'group_errors':group_error,
        'window_review_reasons':window_reasons,'exclusion_reasons':dict(reason_counts),
        'raw_ulog':str(logs[0]),'raw_ulog_sha256':task['raw_ulog_sha256'],'label_file':task['label_file'],
        'label_sha256':task['label_sha256'],'original_csv':task['original_csv'],'original_csv_sha256':task['original_csv_sha256'],
        'output_identities':{str(framepath):digest(framepath),str(windowpath):digest(windowpath)}}
    save(marker,record)
    return {'case_uid':uid,'resumed':False,**summary}


def process(split,workers):
    run=read_json(OUT/'RUNNING.json');execution=read_json(OUT/'EXECUTION_REVISION.json') if (OUT/'EXECUTION_REVISION.json').exists() else run
    check(Path(__file__),execution['script_sha256']);check(OUT/'tasks.json',run['tasks_sha256'])
    tasks=[t for t in read_json(OUT/'tasks.json') if t['split']==split]
    results=[]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i,r in enumerate(pool.map(process_case,tasks,chunksize=1),1):
            results.append(r)
            if i%100==0: print('FULL_CAUSAL_CASES',split,i,'/',len(tasks),'kept',sum(r['supervised_eligible_windows'] for r in results),flush=True)
    total={k:sum(r[k] for r in results) for k in ['rows','original_windows','input_eligible_windows','supervised_eligible_windows','excluded_windows','clock_or_topic_review_groups']}
    total.update(cases=len(tasks),split=split,completed_at=now())
    target=OUT/(split+'_frames_complete.json')
    if not target.exists(): save(target,total)
    print('FRAMES_COMPLETE',json.dumps(total),flush=True)


def records_for(split):
    tasks=[t for t in read_json(OUT/'tasks.json') if t['split']==split]
    records=[read_json(OUT/'case_records'/(case_key(t['case_uid'])+'.json')) for t in tasks]
    tasks_by_uid={t['case_uid']:t for t in tasks}
    return sorted(records,key=lambda r: tasks_by_uid[r['case_uid']]['selected']['index'][0])


def make_ledgers(split,records):
    rows=[]
    for r in records:
        with np.load(r['frame_file'],allow_pickle=False) as saved:
            keys=['original_cache_index','window_start_row','window_end_row','reference_state','reference_y',
                  'supervised_valid','historical_y','input_window_valid','label_endpoint_valid','whole_window_label_valid']
            d={k:saved[k] for k in keys}
            onsets=np.flatnonzero(d['reference_state']==1);onset=int(onsets[0]) if len(onsets) else -1
            for j,index in enumerate(d['original_cache_index']):
                s,e=int(d['window_start_row'][j]),int(d['window_end_row'][j]);ref=d['reference_state'][s:e]
                certain=np.all(ref>=0)
                phase='unknown_reference' if not certain else ('normal_flight' if onset<0 else ('pre_fault' if e<=onset else ('stable_fault' if np.all(ref==1) else ('post_fault_normal' if np.all(ref==0) else 'transition'))))
                label=int(d['reference_y'][j]) if d['supervised_valid'][j] else -1
                rows.append({'original_cache_index':int(index),'case_uid':r['case_uid'],'domain':r['domain'],'split':split,
                    'role':r['role'],'case_level2_label':r['family'],'association_group':r['group'],
                    'window_start_row':s,'window_end_row':e,'historical_y':int(d['historical_y'][j]),
                    'reference_y':int(d['reference_y'][j]),'qualified_y':label,'input_valid':bool(d['input_window_valid'][j]),
                    'label_valid':bool(d['label_endpoint_valid'][j]),'supervised_valid':bool(d['supervised_valid'][j]),
                    'whole_window_label_valid':bool(d['whole_window_label_valid'][j]),
                    'normal_origin':'normal_flight' if r['family']=='normal' else ('normal_part_of_fault_case' if label==0 else 'fault_or_unknown'),
                    'level2_label':'uncertain' if label<0 else ('normal' if label==0 else r['family']),
                    'phase':phase,'fault_fraction':float((ref==1).mean()) if certain else None,'fault_onset_row':onset,
                    'review_reason':r['window_review_reasons'][j]})
    frame=pd.DataFrame(rows).sort_values('original_cache_index').reset_index(drop=True)
    frame['compact_index']=-1
    frame.loc[frame.supervised_valid,'compact_index']=np.arange(int(frame.supervised_valid.sum()))
    if split=='train':
        suffix=pd.DataFrame({'original_cache_index':np.arange(165120,168800),'case_uid':'historical_rflysim_suffix',
            'domain':'RflySim','split':'train','role':'historical_train_suffix','historical_y':0,
            'reference_y':-1,'qualified_y':-1,'input_valid':False,'label_valid':False,'supervised_valid':False,
            'review_reason':'legacy_suffix_per_topic_availability_provenance_not_verified','compact_index':-1})
        frame=pd.concat([frame,suffix],ignore_index=True)
    expected={'train':168800,'val':35520,'test':35392,'real':29319}[split]
    if not np.array_equal(frame.original_cache_index.to_numpy(),np.arange(expected)):raise ValueError('Old cache index coverage differs')
    frame.to_csv(OUT/'ledgers'/(split+'_all_windows.csv'),index=False)
    frame[~frame.supervised_valid].to_csv(OUT/'ledgers'/(split+'_excluded_windows.csv'),index=False)
    frame[frame.supervised_valid].to_csv(OUT/'ledgers'/(split+'_included_windows.csv'),index=False)
    return frame


def fit_norm(records):
    count=0;mean=np.zeros(26);m2=np.zeros(26);windows=0
    for r in records:
        with np.load(r['window_file'],allow_pickle=False) as d:
            x=d['x'].transpose(0,2,1).reshape(-1,26).astype(np.float64)
            if not len(x):continue
            if not np.isfinite(x).all():raise ValueError('Nonfinite qualified train input')
            n=len(x);mu=x.mean(axis=0);delta=mu-mean
            centered=x-mu;local=(centered*centered).sum(axis=0)
            m2+=local+delta*delta*count*n/(count+n);mean+=delta*n/(count+n);count+=n;windows+=len(d['y'])
    if not count:raise ValueError('No eligible train inputs, do not widen thresholds')
    std=np.sqrt(m2/count);zero=std==0;std[zero]=1
    return {'version':VERSION,'data_version':VERSION,'input_contract_sha256':digest(OUT/'input_contract.json'),
        'columns':read_json(OUT/'input_contract.json')['columns'],'mean':mean.tolist(),'std':std.tolist(),
        'window_size':150,'target_hz':50.0,'stride':5,'feature_schema':'rfly_seq_columns_v2',
        'protocol_name':VERSION,'label_space':'normal_fault','source_fault_classes':['motor','sensor'],
        'fit_split':'source_train_only','fit_windows':windows,'fit_frame_contributions':count,
        'zero_variance_unit_scale_columns':[read_json(OUT/'input_contract.json')['columns'][i] for i in np.flatnonzero(zero)],
        'weighting':'All 150 frames in each eligible original-cap64 source train window, overlap retained',
        'method':'float64 parallel central moments; population standard deviation','deployment_approved':False}


def build_cache(split):
    import torch
    torch.set_num_threads(2)
    execution=read_json(OUT/'EXECUTION_REVISION.json') if (OUT/'EXECUTION_REVISION.json').exists() else read_json(OUT/'RUNNING.json')
    check(Path(__file__),execution['script_sha256'])
    if (OUT/(split+'_cache_complete.json')).exists():raise FileExistsError('Cache already completed')
    records=records_for(split);frame=make_ledgers(split,records)
    if split=='train':
        if not (OUT/'norm_stats.json').exists():save(OUT/'norm_stats.json',fit_norm(records))
    stats=read_json(OUT/'norm_stats.json');columns=stats['columns'];mu=torch.tensor(stats['mean'],dtype=torch.float32).view(1,-1,1);sd=torch.tensor(stats['std'],dtype=torch.float32).view(1,-1,1)
    old_name={'train':'train.pt','val':'val.pt','test':'test.pt','real':'real_holdout_test.pt'}[split]
    historical=torch.load(OLD/old_name,weights_only=True,map_location='cpu',mmap=True)
    if not np.array_equal(historical['y'].numpy(),frame.historical_y.to_numpy()):raise ValueError('Old cache y/index identity differs')
    compact=split in ['train','val'];n=int(frame.supervised_valid.sum()) if compact else len(frame)
    x=torch.empty((n,26,150),dtype=torch.float32) if compact else torch.full((n,26,150),float('nan'),dtype=torch.float32)
    labels=torch.empty(n,dtype=torch.int64) if compact else torch.full((n,),-1,dtype=torch.int64)
    indices=[];cursor=0
    for i,r in enumerate(records,1):
        for p,h in r['output_identities'].items():check(Path(p),h)
        with np.load(r['window_file'],allow_pickle=False) as d:
            raw=torch.from_numpy(d['x']);idx=d['original_cache_index'];y=torch.from_numpy(d['y'].copy()).long()
            normalized=(raw-mu)/sd
            if not torch.isfinite(normalized).all():raise ValueError('Bad normalized input')
            if compact:
                x[cursor:cursor+len(y)].copy_(raw);labels[cursor:cursor+len(y)].copy_(y);indices.extend(int(v) for v in idx);cursor+=len(y)
            else:
                x[idx]=raw;labels[idx]=y
        if i%500==0:print('CACHE_ASSEMBLED',split,i,'/',len(records),flush=True)
    if compact and cursor!=n:raise ValueError('Compact population mismatch')
    if compact:
        if not np.array_equal(indices,frame.loc[frame.supervised_valid,'original_cache_index']):raise ValueError('Compact index order mismatch')
        payload={'x':x,'y':labels,'original_cache_index':torch.tensor(indices,dtype=torch.int64),'data_version':VERSION,
            'input_contract_sha256':digest(OUT/'input_contract.json'),'norm_sha256':digest(OUT/'norm_stats.json'),
            'split':split,'normalization_applied':False}
        name=split+'.pt'
    else:
        payload={'x':x,'qualified_reference_y':labels,'historical_y':historical['y'].clone(),
            'input_valid':torch.tensor(frame.input_valid.to_numpy(dtype=bool)),'label_valid':torch.tensor(frame.label_valid.to_numpy(dtype=bool)),
            'supervised_valid':torch.tensor(frame.supervised_valid.to_numpy(dtype=bool)),
            'original_cache_index':torch.arange(n),'data_version':VERSION,'split':split,
            'input_contract_sha256':digest(OUT/'input_contract.json'),'norm_sha256':digest(OUT/'norm_stats.json'),'normalization_applied':False}
        if split=='real':
            mapping={'adapt_normal':0,'calibrate_normal':1,'test_normal':2,'test_fault':3,'supplemental_normal_test':4}
            payload['role_id']=torch.tensor(frame.role.map(mapping).to_numpy(dtype=np.int64))
        name=split+'_evidence.pt'
    path=OUT/'cache'/name;temp=path.with_name(path.name+'.partial')
    if path.exists() or temp.exists():raise FileExistsError('Existing incomplete/final cache requires review')
    torch.save(payload,temp);del payload,x;temp.replace(path)
    loaded=torch.load(path,weights_only=True,map_location='cpu',mmap=True)
    targetmask=np.ones(n,dtype=bool) if compact else frame.supervised_valid.to_numpy(dtype=bool)
    target_y=loaded['y'] if compact else loaded['qualified_reference_y']
    if not torch.isin(target_y[targetmask],torch.tensor([0,1])).all():raise ValueError('Invalid qualified binary labels')
    for start in range(0,n,256):
        chunk=loaded['x'][start:start+256];mask=targetmask[start:start+256]
        if not torch.isfinite(chunk[mask]).all():raise ValueError('Persisted qualified X invalid')
        if not compact and not torch.isnan(chunk[~mask]).all():raise ValueError('Unqualified evidence X must not masquerade as usable input')
    result={'split':split,'original_windows':len(frame),'supervised_windows':int(frame.supervised_valid.sum()),
        'input_eligible_windows':int(frame.input_valid.sum()),'excluded_windows':int((~frame.supervised_valid).sum()),
        'cases':len(records),'entirely_excluded_cases':sum(r['summary']['supervised_eligible_windows']==0 for r in records),
        'input_issues':dict(Counter(reason for r in records for reason in r['group_errors'].values())),
        'class_counts':{'normal':int((frame.loc[frame.supervised_valid,'qualified_y']==0).sum()),'fault':int((frame.loc[frame.supervised_valid,'qualified_y']==1).sum())},
        'roles':{str(role):{'original':len(g),'eligible':int(g.supervised_valid.sum())} for role,g in frame.groupby('role')},
        'cache_file':str(path),'cache_sha256':digest(path),'norm_sha256':digest(OUT/'norm_stats.json'),
        'training_run':False,'deployment_approved':False}
    save(OUT/(split+'_cache_complete.json'),result);print('CACHE_COMPLETE',json.dumps(result),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',choices=['init','cases','cache'],required=True)
    p.add_argument('--split',choices=['train','val','test','real'],default='train');p.add_argument('--workers',type=int,choices=range(1,7),default=4);args=p.parse_args()
    if args.stage=='init':initialize()
    elif args.stage=='cases':process(args.split,args.workers)
    else:build_cache(args.split)


if __name__=='__main__':main()
