"""Independent M1/M0-matched source development, maximum four formal jobs.

Reuses frozen M0 data checks, loss, schedule, selection, sampling and reporting.
No original trainer/backbone/cache is overwritten. GPU requires complete gates.
"""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shutil
import time
import train_causal_source_v1 as source
import train_causal_mstca_m0 as m0
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from uav_fault.data.m1_continuous import M1ContinuousDataset, M1IndexedDataset
from uav_fault.models.cnn.mstca_kinematic_m1 import OnlineMSTCAKinematicM1
from train_source_normal_sampling import state_digest, selection_step

ROOT=Path(__file__).resolve().parents[2]


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def load_protocol(path):
    if source.sha(path)!=path.with_suffix('.json.sha256').read_text().strip():raise ValueError('M1 frozen protocol lock mismatch')
    p=read(path)
    if p['protocol_name']!='causal_m1_continuous150_v1' or p['seeds']!=[42] or p['arms']!=['uniform_replacement','balanced_source_normal']:
        raise ValueError('Unregistered M1 experiment')
    old_path=ROOT/p['M0_protocol_path']
    if source.sha(old_path)!=p['M0_protocol_sha256']:raise ValueError('Original M0 protocol changed')
    old=m0.load_protocol(old_path)
    for k in ['loss','formal_environment']:
        if p[k]!=old[k]:raise ValueError('Unregistered original training semantic change:'+k)
    if {k:v for k,v in p['validation'].items() if k!='population'}!={k:v for k,v in old['validation'].items() if k!='population'}:
        raise ValueError('Original validation selection/settings changed')
    for k,v in old['training'].items():
        if k not in ['parameters','initialization','draws_per_epoch'] and p['training'][k]!=v:
            raise ValueError('Unregistered M0 training hyperparameter change:'+k)
    if p['window_frames']!=150 or p['horizon_us']!=200000:raise ValueError('Wrong M1 temporal contract')
    if len(p['formal_jobs'])>4 or any(j['variant'] not in ['M1','M0_matched'] for j in p['formal_jobs']):raise ValueError('Too many/unregistered formal models')
    if len({(j['variant'],j['arm']) for j in p['formal_jobs']})!=len(p['formal_jobs']):raise ValueError('Duplicate formal job')
    for name,h in p['code_identities'].items():
        if source.sha(ROOT/name)!=h:raise ValueError('Frozen code changed:'+name)
    return p


def prepare(base_root,feature_root,p):
    manifest=read(feature_root/'build_manifest.json')
    if source.sha(feature_root/'build_manifest.json')!=p['feature_manifest_sha256']:raise ValueError('M1 full manifest changed')
    old=read(ROOT/p['M0_protocol_path'])
    for name,h in old['data_identities'].items():
        if source.sha(base_root/name)!=h:raise ValueError('M0 base data changed:'+name)
    ds={s:M1ContinuousDataset(base_root,feature_root,split=s) for s in ['train','val']}
    frame={}
    for s in ds:
        full=source.source_metadata(base_root,s,ds[s].base)
        selected=full.iloc[ds[s].base_indices.numpy()].copy().reset_index(drop=True)
        selected['cache_index']=np.arange(len(selected))
        if not np.array_equal(selected.original_cache_index,ds[s].original_cache_indices):raise ValueError('M1 metadata original IDs mismatch')
        if len(selected)!=p['data_counts'][s]:raise ValueError('M1 common population mismatch')
        frame[s]=selected
    if set(frame['train'].case_uid)&set(frame['val'].case_uid) or set(frame['train'].association_group)&set(frame['val'].association_group):raise ValueError('Source pair/case leakage')
    weights={arm:source.sampling_weights(np.asarray(ds['train'].labels),frame['train'].normal_origin.to_numpy(),arm=='balanced_source_normal') for arm in p['arms']}
    tables={arm:source.probability_table(w,np.asarray(ds['train'].labels),frame['train'].normal_origin.to_numpy()) for arm,w in weights.items()}
    for key in ['total_weight','normal_probability','fault_probability']:
        if not math.isclose(tables[p['arms'][0]][key],tables[p['arms'][1]][key],abs_tol=1e-10,rel_tol=0):raise ValueError('Sampler class mass changed')
    return ds,frame,weights,tables


def build(p,variant,anchor,arm):
    old=m0.load_protocol(ROOT/p['M0_protocol_path'])
    base,receipt=m0.build_model(old,42,anchor)
    common=p['common_M0_pretrained'][arm]
    path=ROOT/common['path']
    if source.sha(path)!=common['sha256']:raise ValueError('Verified M0 pretrained common checkpoint changed')
    state=torch.load(path,map_location='cpu',weights_only=True)
    base.load_state_dict(state,strict=True)
    if state_digest(base.state_dict())!=common['state_digest']:raise ValueError('Pretrained common tensor digest mismatch')
    receipt=dict(receipt,pretrained_common_source=common,common_parameters_from_M0_best=True,trained_checkpoint_used=True,current_common_state_digest=common['state_digest'],comparison='M1 and M0-matched both start from this same arm-specific pretrained state; not a from-scratch feature-only comparison')
    if variant=='M0_matched':return base,receipt
    if variant!='M1':raise ValueError('Unknown M1 experiment variant')
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42);model=OnlineMSTCAKinematicM1(hidden_size=64,dropout=.2)
    model.load_common_initial(base.state_dict())
    if sum(v.numel() for v in model.parameters())!=109106:raise ValueError('M1 adapter parameter count mismatch')
    return model,receipt


def evaluate(model,variant,loader,device,metadata):
    model.eval();prob=[];actual=[];indices=[]
    with torch.inference_mode():
        for x,r,y,idx in loader:
            x,r=x.to(device),r.to(device)
            z=model(x,r) if variant=='M1' else model(x)
            q=z.softmax(1).cpu().numpy()
            if not np.isfinite(q).all():raise FloatingPointError('Nonfinite validation probabilities')
            prob.append(q);actual.extend(y.tolist());indices.extend(idx.tolist())
    if indices!=metadata.cache_index.tolist():raise ValueError('Validation ordering differs')
    probabilities=np.concatenate(prob);labels=np.array(actual,dtype=np.int64)
    return source.stratified_metrics(labels,probabilities.argmax(1),metadata),probabilities,labels


def run(args,p):
    if args.output.exists():raise FileExistsError('Preserve all M1 runs; no overwrite')
    if args.arm not in p['arms'] or args.seed!=42:raise ValueError('Unregistered M1 arm/seed')
    if not args.cpu_check:
        if args.gate is None:raise RuntimeError('Formal M1 requires completed pretraining gate receipt')
        gate=read(args.gate)
        if gate['status']!='PASS_ALL_M1_PRETRAINING_GATES' or gate['protocol_sha256']!=source.sha(args.protocol):raise RuntimeError('Formal M1 blocked by pretraining gates')
        if not torch.cuda.is_available():raise RuntimeError('Formal M1 BLOCKED_NO_CUDA; no CPU fallback')
        for k,v in p['formal_environment'].items():
            if source.environment()[k]!=v:raise RuntimeError('Formal environment mismatch:'+k)
        if {'variant':args.variant,'arm':args.arm} not in p['formal_jobs']:raise ValueError('Formal job not registered')
    ds,frame,weights,tables=prepare(args.data,args.features,p)
    cfg=p['training'];torch.set_num_threads(cfg['cpu_threads']);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cudnn.allow_tf32=True;torch.backends.cuda.matmul.allow_tf32=False
    np.random.seed(42);torch.manual_seed(42)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(42)
    model,init=build(p,args.variant,args.initial_anchor,args.arm);initial={k:v.detach().clone() for k,v in model.state_dict().items()}
    metadata=frame['val'];indices=None;draws=len(ds['train']);epochs=100;batch=512;vbatch=512
    if args.cpu_check:
        draws=128;epochs=2;batch=32;vbatch=128
        picked=[]
        for _,g in metadata.groupby('case_uid',sort=False):picked.extend(g.iloc[np.unique(np.linspace(0,len(g)-1,min(2,len(g))).round().astype(int))].cache_index.tolist())
        metadata=metadata.iloc[picked].copy().reset_index(drop=True)
    device=torch.device('cpu' if args.cpu_check else 'cuda:0');model.to(device)
    sampler=WeightedRandomSampler(weights[args.arm],draws,replacement=True,generator=torch.Generator().manual_seed(53))
    trainloader=DataLoader(M1IndexedDataset(ds['train']),batch_size=batch,sampler=sampler,num_workers=0,generator=torch.Generator().manual_seed(143))
    valloader=DataLoader(M1IndexedDataset(ds['val'],metadata.cache_index.to_numpy()),batch_size=vbatch,shuffle=False,num_workers=0,generator=torch.Generator().manual_seed(143))
    criterion=source.FocalLoss(gamma=cfg['loss_gamma']);optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay'])
    args.output.mkdir(parents=True,exist_ok=False);out=args.output;started=time.perf_counter()
    report={'status':'RUNNING_CPU_CHECK' if args.cpu_check else 'RUNNING_FORMAL_M1_COMPARISON','variant':args.variant,'arm':args.arm,'seed':42,'protocol_sha256':source.sha(args.protocol),'cpu_check':args.cpu_check,'environment':source.environment(),'model_parameters':sum(v.numel() for v in model.parameters()),'source_train_windows':len(ds['train']),'source_val_windows':len(ds['val']),'input_frames':150,'initial_state_digest':state_digest(initial),'common_initialization':init,'sampling':tables[args.arm],'test_or_Real_read':False,'started_at':datetime.now(timezone.utc).isoformat()}
    source.write(out/'report.json',report);source.save_state(out/'initial.pt',initial);shutil.copyfile(args.protocol,out/'protocol.json')
    selection={'best_score':-float('inf'),'best_epoch':None,'significant_reference':-float('inf'),'bad_epochs':0};history=[]
    try:
        for epoch in range(1,epochs+1):
            lr=source.get_warmup_cosine_lr(epoch-1,5,100,cfg['lr'],cfg['min_lr'])
            for group in optimizer.param_groups:group['lr']=lr
            model.train();losses=[];realized=Counter()
            for x,r,y,idx in trainloader:
                if not torch.isfinite(x).all() or not torch.isfinite(r).all() or not torch.isin(y,torch.tensor([0,1])).all():raise ValueError('Unavailable sample before forward/BN')
                x,r,y=x.to(device),r.to(device),y.to(device);optimizer.zero_grad(set_to_none=True)
                z=model(x,r) if args.variant=='M1' else model(x);loss=criterion(z,y)
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite training loss')
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1,error_if_nonfinite=True);optimizer.step()
                losses.append(float(loss.detach().cpu()));realized.update(frame['train'].normal_origin.iloc[idx.numpy()].tolist())
            metrics,_,_=evaluate(model,args.variant,valloader,device,metadata)
            selection,best,stop=selection_step(selection,float(metrics['overall']['macro_f1']),epoch,p['validation']['early_stop'])
            if best:source.save_state(out/'best.pt',model.state_dict())
            source.save_state(out/'last.pt',model.state_dict())
            history.append({'epoch':epoch,'lr':lr,'train_loss':float(np.mean(losses)),'gradient_steps':len(losses),'realized_training_origins':dict(realized),'source_validation':metrics,'selection':dict(selection)})
            source.write(out/'history.json',history);print('M1_SOURCE_JOB',args.variant,args.arm,'CPU' if args.cpu_check else 'FORMAL',epoch,'macro_f1',metrics['overall']['macro_f1'],flush=True)
            if stop:break
        model.load_state_dict(torch.load(out/'best.pt',map_location=device,weights_only=True))
        metrics,probs,labels=evaluate(model,args.variant,valloader,device,metadata)
        if metrics!=history[selection['best_epoch']-1]['source_validation']:raise RuntimeError('Best reload validation mismatch')
        if not any(not torch.equal(initial[k],v.detach().cpu()) for k,v in model.named_parameters()):raise RuntimeError('No parameter updates')
        if source.sha(args.protocol)!=report['protocol_sha256']:raise ValueError('Frozen protocol changed during training')
        if source.sha(args.features/'build_manifest.json')!=p['feature_manifest_sha256']:raise ValueError('Feature manifest changed during training')
        for name,expected in p['code_identities'].items():
            if source.sha(ROOT/name)!=expected:raise ValueError('Frozen code changed during training:'+name)
        old=read(ROOT/p['M0_protocol_path'])
        for name,expected in old['data_identities'].items():
            if source.sha(args.data/name)!=expected:raise ValueError('Original base data changed during training:'+name)
        np.savez_compressed(out/'source_validation_predictions.npz',probabilities=probs,labels=labels,compact_indices=metadata.cache_index.to_numpy(dtype=np.int64),original_indices=metadata.original_cache_index.to_numpy(dtype=np.int64))
        report.update(status='PASS_CPU_SMALL_TRAINING_NOT_PERFORMANCE' if args.cpu_check else 'COMPLETE_FORMAL_SOURCE_M1_COMPARISON',epochs=len(history),best_epoch=selection['best_epoch'],gradient_steps=sum(h['gradient_steps'] for h in history),validation_windows=len(metadata),source_validation=metrics,elapsed_s=time.perf_counter()-started,finished_at=datetime.now(timezone.utc).isoformat(),best_checkpoint_sha256=source.sha(out/'best.pt'),checkpoint_reload_matches=True)
        source.write(out/'report.json',report);return report
    except Exception as exc:
        report.update(status='FAILED',error=repr(exc));source.write(out/'report.json',report);raise


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--protocol',type=Path,required=True);p.add_argument('--data',type=Path,required=True);p.add_argument('--features',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--initial-anchor',type=Path,required=True);p.add_argument('--gate',type=Path);p.add_argument('--variant',choices=['M1','M0_matched'],required=True);p.add_argument('--arm',choices=['uniform_replacement','balanced_source_normal'],required=True);p.add_argument('--seed',type=int,default=42);p.add_argument('--cpu-check',action='store_true');a=p.parse_args();run(a,load_protocol(a.protocol))


if __name__=='__main__':main()
