"""Reproducible offline normal-only UAV baselines; never reads REAL or open-set CSVs.

Processed signals are NOT causal acquisition. Keep timing metrics exploratory.
Source files are read-only; artifacts go into a separate research run directory.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import average_precision_score, roc_auc_score
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
AUDIT = ROOT/'docs/research/2026-10-08/data_feasibility'
OUT = ROOT/'docs/research/2026-10-08/uav_baselines_v1'
MANIFEST = Path('D:/UAV-Data/splits/rflymad_case_split_v2/rflymad_case_level_split_manifest.csv')
CONTRACT = Path('D:/UAV-Data/feature_contract_v1.json')
MODELS = ('pca','persistence','gru')
CONFIG = dict(seed=42, context_frames_50hz=150, input_stride_frames=5,
    decision_stride_frames=5, train_windows_per_case=96, val_loss_windows_per_case=96,
    pca_explained_variance=.95, gru_hidden=32, batch_size=256, max_epochs=12,
    patience=3, learning_rate=.001, normal_validation_quantile=.99,
    consecutive_decisions=3, torch_threads=4, evaluation='offline_processed_simulation_only',
    feature_count=26, label_use='train normal case selection; validation normal threshold; test evaluation only',
    detection_rule='new alarm rising edge within fault interval; continuing prefault alarm is not a hit',
    smoothing='three consecutive scores strictly above threshold; clears immediately below threshold',
    timing_caveat='processed linear interpolation uses future raw values; delay is not a causal online claim')


def write_json(path, obj):
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def contexts(x, ends):
    """30 historical observations from t-3.0 to t-0.1 s; target at t."""
    indices=ends[:,None]-150+np.arange(0,150,5)[None,:]
    past=x[indices].copy()
    target=x[ends].copy()
    # Contract channel 14 is baro_alt_relative. Use the SAME historical origin
    # for context and target, without reading a future sample at extraction.
    origin=past[:,0,14].copy()
    past[:,:,14]-=origin[:,None]
    target[:,14]-=origin
    return past,target


def select_ends(n, cap=0):
    ends=np.arange(150,n,5,dtype=np.int64)
    if cap and len(ends)>cap:
        ends=ends[np.linspace(0,len(ends)-1,cap,dtype=int)]
    return ends


def active_intervals(mask, t):
    a=np.flatnonzero(mask & ~np.r_[False,mask[:-1]])
    b=np.flatnonzero(mask & ~np.r_[mask[1:],False])+1
    return [(float(t[i]),float(t[j-1]+.02)) for i,j in zip(a,b)]


def load_case(row, features):
    assert row['domain'] in ('SIL','HIL') and row['split'] in ('train','val','test')
    assert row['split']!='train' or row['level1_label']=='normal'
    payload=Path(row['final_output_path']).read_bytes()
    actual=hashlib.sha256(payload).hexdigest()
    if actual!=row['current_sha256']:
        raise ValueError(f"Changed source file since audit: {row['case_uid']}")
    df=pd.read_csv(io.BytesIO(payload))
    x=np.empty((len(df),26),dtype=np.float32)
    names=[f['canonical_name'] for f in features]
    assert names[14]=='baro_alt_relative'
    for i,f in enumerate(features):
        col=f['source_columns'][row['domain']]
        if col!='derived': x[:,i]=df[col].to_numpy(dtype=np.float32)
    x[:,names.index('acc_norm')]=np.linalg.norm(x[:,[names.index('accel_x'),names.index('accel_y'),names.index('accel_z')]],axis=1)
    if not np.isfinite(x).all(): raise ValueError('Nonfinite input; no silent imputation')
    ts=df.Timestamp.to_numpy(dtype=np.float64)/1e6
    ts-=ts[0]
    if not np.allclose(np.diff(ts),.02,atol=1e-5,rtol=0): raise ValueError('Non-50-Hz timestamps')
    flags=df[row['window_label_source']].to_numpy()>0
    if (row['level1_label']=='normal')!=bool(not flags.any()): raise ValueError('Label activity inconsistency')
    meta={k:row[k] for k in ('case_uid','domain','flight_status','level3_label','split','simulation_pair_key')}
    meta.update(duration_s=float(ts[-1]),intervals=active_intervals(flags,ts), current_sha256=actual)
    return dict(meta=meta,x=x,t=ts,flags=flags)


def alarm_states(scores, threshold, consecutive=3):
    alarm=np.zeros(len(scores),dtype=bool)
    run=0
    for i,s in enumerate(scores):
        run=run+1 if s>threshold else 0
        alarm[i]=run>=consecutive
    edges=alarm & ~np.r_[False,alarm[:-1]]
    return alarm,edges


def evaluate_alarm(times, truth, intervals, scores, threshold):
    alarm,edges=alarm_states(scores,threshold,CONFIG['consecutive_decisions'])
    starts=times[edges]
    delays=[]
    for begin,end in intervals:
        valid=starts[(starts>=begin-1e-8)&(starts<end-1e-8)]
        if len(valid): delays.append(float(valid[0]-begin))
    return dict(fault_events=len(intervals),detected_events=len(delays),
        fault_case=int(bool(intervals)),detected_case=int(bool(delays)),
        false_alarm_events=int((edges & ~truth).sum()),
        prefault_alarm_case=int(bool(intervals) and np.any(starts<intervals[0][0])),
        normal_decisions=int((~truth).sum()),normal_exposure_s=float((~truth).sum()*.1),
        normal_alarm_decisions=int((alarm & ~truth).sum()),
        point_tp=int((alarm & truth).sum()),point_fp=int((alarm & ~truth).sum()),
        point_fn=int((~alarm & truth).sum()),point_tn=int((~alarm & ~truth).sum()),
        detected_delays_s=delays)


def aggregate(rows):
    keys=('fault_events','detected_events','fault_case','detected_case','false_alarm_events',
          'prefault_alarm_case','normal_decisions','normal_exposure_s','normal_alarm_decisions',
          'point_tp','point_fp','point_fn','point_tn')
    r={k:sum(x[k] for x in rows) for k in keys}
    r['cases']=len(rows)
    r['event_recall']=r['detected_events']/r['fault_events'] if r['fault_events'] else None
    r['false_alarms_per_normal_hour']=r['false_alarm_events']/(r['normal_exposure_s']/3600) if r['normal_exposure_s'] else None
    r['normal_time_alarm_fraction']=r['normal_alarm_decisions']/r['normal_decisions'] if r['normal_decisions'] else None
    delays=[v for x in rows for v in x['detected_delays_s']]
    r['detected_only_delay_median_s']=float(np.median(delays)) if delays else None
    r['detected_only_delay_p90_s']=float(np.quantile(delays,.9)) if delays else None
    den=2*r['point_tp']+r['point_fp']+r['point_fn']
    r['point_f1_no_adjustment']=2*r['point_tp']/den if den else 0.
    return r


class GRUPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru=nn.GRU(26,CONFIG['gru_hidden'],batch_first=True)
        self.head=nn.Linear(CONFIG['gru_hidden'],26)

    def forward(self,x):
        _,h=self.gru(x)
        # Residual prediction retains persistence as a natural reference.
        return x[:,-1,:]+self.head(h[-1])


def fit_gru(train_x, train_y, val_x, val_y):
    model=GRUPredictor()
    optimizer=torch.optim.Adam(model.parameters(),lr=CONFIG['learning_rate'])
    tx=torch.from_numpy(train_x); ty=torch.from_numpy(train_y)
    vx=torch.from_numpy(val_x); vy=torch.from_numpy(val_y)
    gen=torch.Generator().manual_seed(CONFIG['seed'])
    best=float('inf'); best_state=None; stale=0; history=[]
    for epoch in range(1,CONFIG['max_epochs']+1):
        started=time.monotonic(); model.train(); total=0.
        order=torch.randperm(len(tx),generator=gen)
        for idx in order.split(CONFIG['batch_size']):
            optimizer.zero_grad(set_to_none=True)
            loss=((model(tx[idx])-ty[idx])**2).mean()
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.)
            optimizer.step(); total+=loss.item()*len(idx)
        model.eval(); validation=0.
        with torch.no_grad():
            for start in range(0,len(vx),512):
                end=start+512
                validation+=((model(vx[start:end])-vy[start:end])**2).sum().item()
        validation/=len(vx)*26
        history.append(dict(epoch=epoch,train_mse=total/len(tx),val_normal_mse=validation,elapsed_s=time.monotonic()-started))
        print(json.dumps(history[-1]),flush=True)
        if validation<best-1e-7:
            best=validation;best_state={k:v.detach().clone() for k,v in model.state_dict().items()};stale=0
        else: stale+=1
        if stale>=CONFIG['patience']: break
    model.load_state_dict(best_state);model.eval()
    torch.save(best_state,OUT/'gru_state.pt')
    write_json(OUT/'training_history.json',history)
    return model


def score_cases(cases, mean, scale, pca, gru):
    for case in cases:
        ends=select_ends(len(case['x']))
        xs,ys=contexts(case['x'],ends)
        xs=(xs-mean)/scale;ys=(ys-mean)/scale
        errors={}
        errors['pca']=(ys-pca.inverse_transform(pca.transform(ys)))**2
        errors['persistence']=(ys-xs[:,-1,:])**2
        chunks=[]
        with torch.no_grad():
            for start in range(0,len(xs),512):
                pred=gru(torch.from_numpy(xs[start:start+512])).numpy()
                chunks.append((ys[start:start+512]-pred)**2)
        errors['gru']=np.concatenate(chunks)
        case['scores']={k:v.mean(axis=1) for k,v in errors.items()}
        case['ends']=ends
        case['decision_t']=case['t'][ends]
        case['truth']=case['flags'][ends]
        case['fault_contribution']={k:v[case['truth']].mean(axis=0).tolist() if case['truth'].any() else None for k,v in errors.items()}
        # Keep only evaluation series after scoring to reduce resident memory.
        del case['x']


def bootstrap_ci(rows):
    # Paired SIL/HIL scenarios form the bootstrap unit; never resample windows.
    groups={}
    for r in rows: groups.setdefault(r['simulation_pair_key'],[]).append(r)
    values=list(groups.values());rng=np.random.default_rng(CONFIG['seed'])
    counts=np.array([[sum(x['fault_events'] for x in v),sum(x['detected_events'] for x in v),
        sum(x['false_alarm_events'] for x in v),sum(x['normal_exposure_s'] for x in v)] for v in values])
    totals=counts[rng.integers(len(values),size=(500,len(values)))].sum(axis=1)
    recall=totals[:,1]/totals[:,0]
    far=totals[:,2]/(totals[:,3]/3600)
    return dict(bootstrap_unit='simulation_pair_key',replicates=500,
        event_recall_95ci=np.quantile(recall,[.025,.975]).tolist(),
        false_alarms_per_normal_hour_95ci=np.quantile(far,[.025,.975]).tolist())


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--smoke',action='store_true');args=parser.parse_args()
    global OUT
    if args.smoke: OUT=OUT.with_name(OUT.name+'_smoke');CONFIG.update(max_epochs=2,train_windows_per_case=12,val_loss_windows_per_case=12)
    OUT.mkdir(parents=True,exist_ok=True)
    random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    write_json(OUT/'protocol.json',CONFIG)
    features=json.loads(CONTRACT.read_text(encoding='utf-8'))['features']
    names=[f['canonical_name'] for f in features]
    manifest=pd.read_csv(MANIFEST,keep_default_na=False)
    audit=pd.read_csv(AUDIT/'full_case_audit.csv',keep_default_na=False)
    selected=manifest[((manifest.split=='train')&(manifest.level1_label=='normal'))|manifest.split.isin(['val','test'])].copy()
    assert set(selected.domain)<= {'SIL','HIL'}
    assert selected.groupby('simulation_pair_key').split.nunique().max()==1
    selected=selected.merge(audit[['case_uid','sha256']],on='case_uid',validate='one_to_one').rename(columns={'sha256':'current_sha256'})
    if args.smoke:
        selected=selected.groupby(['split','domain','level3_label'],sort=True).head(1)
    selected['original_manifest_sha256']=selected['output_sha256']
    selected['output_sha256']=selected['current_sha256']
    selected.to_csv(OUT/'frozen_case_index.csv',index=False,encoding='utf-8-sig')
    write_json(OUT/'provenance.json',dict(manifest_sha256=hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
        contract_sha256=hashlib.sha256(CONTRACT.read_bytes()).hexdigest(),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        features=names,numpy=np.__version__,torch=torch.__version__,
        source_update='current hashes from full audit; split membership unchanged; HIL original hashes retained',
        source_files_read='train normal + val + internal test only; no REAL/open-set input files'))
    rows=selected.to_dict('records')
    # Test signals are loaded only after model and thresholds have been frozen.
    devrows=[r for r in rows if r['split']!='test']
    print(f'Reading {len(devrows)} train/validation cases',flush=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        cases=list(pool.map(lambda r:load_case(r,features),devrows))
    train=[c for c in cases if c['meta']['split']=='train']
    val=[c for c in cases if c['meta']['split']=='val']
    vt=[c for c in val if c['meta']['level3_label']=='normal']
    def materialize(items,cap):
        pairs=[contexts(c['x'],select_ends(len(c['x']),cap)) for c in items]
        return np.concatenate([v[0] for v in pairs]),np.concatenate([v[1] for v in pairs])
    tx,ty=materialize(train,CONFIG['train_windows_per_case'])
    vx,vy=materialize(vt,CONFIG['val_loss_windows_per_case'])
    mean=ty.mean(axis=0);scale=np.maximum(ty.std(axis=0),1e-6)
    tx=(tx-mean)/scale;ty=(ty-mean)/scale;vx=(vx-mean)/scale;vy=(vy-mean)/scale
    pca=PCA(n_components=CONFIG['pca_explained_variance'],svd_solver='full').fit(ty)
    print(f'Normal training cases={len(train)}, windows={len(ty)}; PCA components={pca.n_components_}',flush=True)
    np.savez_compressed(OUT/'pca_and_scaler.npz',mean=mean,scale=scale,pca_mean=pca.mean_,components=pca.components_,explained_variance=pca.explained_variance_ratio_)
    gru=fit_gru(tx,ty,vx,vy)
    write_json(OUT/'fit_summary.json',dict(train_cases=len(train),train_windows=len(ty),val_normal_cases=len(vt),val_loss_windows=len(vy),pca_components=int(pca.n_components_)))
    del tx,ty,vx,vy,train
    print('Scoring validation and freezing thresholds',flush=True)
    score_cases(val,mean,scale,pca,gru)
    thresholds={model:float(np.quantile(np.concatenate([c['scores'][model] for c in vt]),CONFIG['normal_validation_quantile'])) for model in MODELS}
    write_json(OUT/'thresholds.json',thresholds)
    testrows=[r for r in rows if r['split']=='test']
    print(f'Thresholds frozen: {thresholds}; reading {len(testrows)} internal test cases',flush=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        test=list(pool.map(lambda r:load_case(r,features),testrows))
    score_cases(test,mean,scale,pca,gru)
    metrics=[];events=[];score_dir=OUT/'scores';score_dir.mkdir(exist_ok=True)
    for case_index,case in enumerate(val+test):
        meta=case['meta']
        score_name=f'case_{case_index:04d}.npz'
        np.savez_compressed(score_dir/score_name,t=case['decision_t'],truth=case['truth'],**case['scores'])
        for model in MODELS:
            r=evaluate_alarm(case['decision_t'],case['truth'],meta['intervals'],case['scores'][model],thresholds[model])
            r.update({k:v for k,v in meta.items() if k not in ('intervals','current_sha256')})
            r.update(model=model,score_file=f'scores/{score_name}')
            contribution=case['fault_contribution'][model]
            r['top_fault_residual_features']=[names[i] for i in np.argsort(contribution)[-3:][::-1]] if contribution is not None else []
            metrics.append(r)
        events.append(dict(**meta,score_file=f'scores/{score_name}'))
    write_json(OUT/'case_metrics.json',metrics);write_json(OUT/'case_metadata.json',events)
    pd.DataFrame(metrics).to_csv(OUT/'case_metrics.csv',index=False,encoding='utf-8-sig')
    summaries=[];breakdowns=[]
    for split in ('val','test'):
        sc=[c for c in val+test if c['meta']['split']==split]
        truth=np.concatenate([c['truth'] for c in sc])
        for model in MODELS:
            mr=[r for r in metrics if r['split']==split and r['model']==model]
            result=dict(split=split,model=model,threshold=thresholds[model],**aggregate(mr))
            score=np.concatenate([c['scores'][model] for c in sc])
            result.update(point_auprc=float(average_precision_score(truth,score)),point_auroc=float(roc_auc_score(truth,score)))
            result.update(bootstrap_ci(mr))
            summaries.append(result)
            for group in ('domain','flight_status','level3_label'):
                for value in sorted({r[group] for r in mr}):
                    subset=[r for r in mr if r[group]==value]
                    breakdowns.append(dict(split=split,model=model,group=group,value=value,**aggregate(subset)))
    write_json(OUT/'summary.json',summaries)
    pd.DataFrame(summaries).to_csv(OUT/'summary.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(breakdowns).to_csv(OUT/'breakdown.csv',index=False,encoding='utf-8-sig')
    print(pd.DataFrame(summaries)[['split','model','event_recall','false_alarms_per_normal_hour','normal_time_alarm_fraction','detected_only_delay_median_s','point_auprc']].to_string(index=False),flush=True)
    print(f'Artifacts written: {OUT}',flush=True)


if __name__=='__main__': main()
