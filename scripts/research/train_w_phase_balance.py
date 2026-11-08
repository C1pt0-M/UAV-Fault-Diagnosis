"""W r1.1 CV: only sampler weights differ from frozen T. No source val."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/research'))
import train_t_selection as t
from train_t_selection import s
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from smoke_normal_source_sampling import sampling_weights, ORIGINAL, PREFAULT, FAULT
OUTPUT=ROOT/'artifacts/w_phase_balance_mstca_v1'
PROTOCOL=ROOT/'configs/experiments/w_phase_balance_mstca_v1.json'
FREEZE_COMMIT='4d3b5538597469cba7b707eb7850c49f8ee91e25'
FLIGHTS=('acce','circling','dece','hover','velocity','waypoint')
ARMS=('W1','W2')
SEEDS=t.SEEDS
DATA=Path('/root/autodl-tmp/causal_source_v1_20261031/unpacked/data')

def train_only_guard(root):
    root=Path(root).resolve()
    allowed={'build_manifest.json','input_contract.json','norm_stats.json','train_cache_complete.json','cache/train.pt','ledgers/train_included_windows.csv'}
    def guard(event,args):
        if event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)):
            return
        path=Path(os.fsdecode(args[0])).resolve()
        if path.is_relative_to(root):
            relative=path.relative_to(root).as_posix()
            if relative not in allowed or any(c in (args[1] or '') for c in 'wax+') or args[2]&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND):
                raise PermissionError('W forbids non-train or writable source access: '+relative)
    sys.addaudithook(guard)

def onset_references(fit):
    flight=fit.case_uid.str.split('/').str[1]
    if not flight.isin(FLIGHTS).all():
        raise ValueError('Unknown flight type')
    onset=pd.to_numeric(fit.fault_onset_row.replace('',np.nan),errors='raise')
    valid=fit.loc[onset.ge(0),['case_uid']].copy()
    valid['flight']=flight.loc[valid.index]
    valid['onset_s']=onset.loc[valid.index]*.02
    if valid.groupby('case_uid').onset_s.nunique().gt(1).any():
        raise ValueError('Inconsistent onset within case')
    cases=valid.drop_duplicates('case_uid')
    refs=cases.groupby('flight').onset_s.median().to_dict()
    if set(refs)!=set(FLIGHTS) or not all(np.isfinite(v) and v>=0 for v in refs.values()):
        raise ValueError('Every flight needs a valid FIT-only onset case')
    return {f:float(refs[f]) for f in FLIGHTS}

def cells(frame,refs):
    flight=frame.case_uid.str.split('/').str[1].to_numpy()
    if not set(flight).issubset(refs):
        raise ValueError('Missing flight reference')
    end=frame.window_end_row.to_numpy(dtype=np.float64)*.02
    if not np.isfinite(end).all() or (end<0).any():
        raise ValueError('Invalid window end time')
    post=end>np.array([refs[f] for f in flight])
    return flight,np.where(post,'post','pre')

def phase_weights(fit,arm,*,force_one=False):
    if arm not in ARMS:
        raise ValueError('Only W1/W2 have new training weights')
    y=fit.qualified_y.to_numpy()
    origins=np.where(y==1,FAULT,np.where(fit.normal_origin.eq('normal_flight'),ORIGINAL,PREFAULT))
    base=sampling_weights(y,origins,True)
    refs=onset_references(fit)
    flight,phase=cells(fit,refs)
    result=base.clone()
    records={}
    for f in FLIGHTS:
        for p in ('pre','post'):
            mask=(flight==f)&(phase==p)
            normal=torch.from_numpy(mask&(y==0));fault=torch.from_numpy(mask&(y==1))
            n=float(base[normal].sum());q=float(base[fault].sum());ratio=q/n if n else 1.
            multiplier=1. if force_one else max(1.,ratio if arm=='W1' else np.sqrt(ratio))
            result[normal]*=multiplier
            records[f+'/'+p]={'N_c':n,'F_c':q,'r_c':ratio,'multiplier':float(multiplier),
                'windows':int(mask.sum()),'normal_windows':int(normal.sum()),'fault_windows':int(fault.sum())}
    if not torch.isfinite(result).all() or (result<=0).any():
        raise ValueError('Invalid sampling weights')
    total=float(result.sum())
    for name,row in records.items():
        f,p=name.split('/');mask=torch.from_numpy((flight==f)&(phase==p))
        row['sampling_probability']=float(result[mask].sum())/total
        row['normal_sampling_probability']=float(result[mask&torch.from_numpy(y==0)].sum())/total
    return result,{'onset_reference_s':refs,'cells':records,'effective_sample_size':total**2/float(result.square().sum()),
        'weight_sum':total,'onset_rule':'FIT unique case_uid, fault_onset_row>=0; median onset seconds',
        'renormalized':False,'draws_per_epoch':len(fit)}

def run(args):
    output=Path(args.output)
    if output.exists():
        raise FileExistsError('Fresh W output required; never overwrite a failed run')
    if args.seed not in SEEDS or args.fold not in range(3) or args.arm not in ARMS:
        raise ValueError('Unregistered W slot')
    if output.resolve()!=(OUTPUT/'cv'/f'{args.arm}_seed{args.seed}_fold{args.fold}').resolve():
        raise ValueError('Use canonical W run directory')
    t.require_formal_environment()
    identities=t.verify_data(args.data,local=True)  # Six TRAIN identities; never hash val.
    frame=s.read_ledger(args.data)
    sizes=s.fold_sizes(frame,s.read_json(s.DEFAULT_PROTOCOL))
    positions,_=s.split_ledger(frame,args.fold,'cv')
    fit=s.CacheWindows(args.data,frame,positions['fit'])
    inner=s.CacheWindows(args.data,frame,positions['inner'])
    held=s.CacheWindows(args.data,frame,positions['heldout'])
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.allow_tf32=True
    torch.backends.cuda.matmul.allow_tf32=False
    np.random.seed(args.seed);torch.manual_seed(args.seed);torch.cuda.manual_seed_all(args.seed)
    before=torch.random.get_rng_state().clone()
    model=s.build_variant('mstca',26,2,.2)
    receipt=s.initialize_variant(model,'mstca',args.seed,policy='module_seed_v1')
    if not torch.equal(before,torch.random.get_rng_state()) or receipt['full_state_digest']!=t.golden_digests()[('mstca',args.seed)]:
        raise RuntimeError('Golden mismatch: STOP, never regenerate')
    weights,statistics=phase_weights(frame.iloc[positions['fit']],args.arm)
    sampler=WeightedRandomSampler(weights,len(fit),replacement=True,generator=torch.Generator().manual_seed(args.seed+11))
    loader=DataLoader(fit,batch_size=512,sampler=sampler,num_workers=0,generator=torch.Generator().manual_seed(args.seed+101))
    output.mkdir(parents=True,exist_ok=False)
    s.write_json(output/'factory_receipt.json',receipt)
    report={'status':'RUNNING','phase':'cv','arm':args.arm,'seed':args.seed,'fold':args.fold,'protocol_commit':FREEZE_COMMIT,
        'protocol_name':'w_phase_balance_mstca_v1','environment':t.environment(),'data_identities':identities,
        'fold_sizes':sizes,'used_window_counts':{k:len(v) for k,v in positions.items()},'initial_digest':receipt['full_state_digest'],
        'sampler':statistics,'epochs':0,'history':[],'optimizer_steps':0,'checkpoint_policy':'ALL_100_RETAINED_DISK_SUFFICIENT',
        'model_inputs':'unchanged 26 cached channels, no time/phase/position added','started_at':datetime.now(timezone.utc).isoformat()}
    s.write_json(output/'report.json',report)
    start=time.perf_counter()
    try:
        model.to('cuda:0')
        def progress(history):
            report.update(history=history,epochs=len(history),optimizer_steps=sum(row['gradient_steps'] for row in history))
            s.write_json(output/'report.json',report,update=True)
        history,ip,hp=t.trajectory(model,loader,t.inference_loader(inner,args.seed),t.inference_loader(inner,args.seed),
            t.inference_loader(held,args.seed),output,100,'cuda:0',progress=progress)
        selection={'T0':t.replay_t0([row['inner_class_balanced_ce'] for row in history]),'epoch100':{'epoch':100}}
        s.write_json(output/'selections.json',selection)
        t.save_predictions(output/'inner_predictions.npz',ip,inner)
        t.save_predictions(output/'heldout_predictions.npz',hp,held)
        report.update(status='COMPLETE',wall_time_s=time.perf_counter()-start,
            checkpoint_state_digests={str(row['epoch']):row['state_digest'] for row in history},finished_at=datetime.now(timezone.utc).isoformat())
        s.write_json(output/'report.json',report,update=True)
        return report
    except Exception as exc:
        report.update(status='FAILED',error=repr(exc),wall_time_s=time.perf_counter()-start)
        s.write_json(output/'report.json',report,update=True)
        raise

def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,default=DATA)
    p.add_argument('--arm',choices=ARMS,required=True)
    p.add_argument('--seed',type=int,choices=SEEDS,required=True)
    p.add_argument('--fold',type=int,choices=range(3),required=True)
    p.add_argument('--output',type=Path,required=True)
    return p
if __name__=='__main__':
    args=parser().parse_args();train_only_guard(args.data)
    r=run(args);print({k:r[k] for k in ('status','arm','seed','fold','epochs')})
