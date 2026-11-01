"""Frozen finite causal source-development jobs, no test/Real reads.

Paired sampling arms, one registered seed, current versioned data and masks.
CPU check performs tiny actual updates but is never a formal retraining result.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'packages/uav_fault'))
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from uav_fault.data.causal_cache import CausalCacheDataset, VERSION
from smoke_normal_source_sampling import (ORIGINAL, PREFAULT, FAULT, FocalLoss, OnlineKGMSTCA,
    sampling_weights, probability_table, stratified_metrics, save_state)
from train_source_normal_sampling import selection_step, state_digest
from uav_fault.models.train_online_detector import get_warmup_cosine_lr


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b''): h.update(b)
    return h.hexdigest()


def write(path, value):
    partial = path.with_name(path.name + '.partial')
    with partial.open('w', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
    partial.replace(path)


def load_protocol(path):
    lock = path.with_suffix(path.suffix + '.sha256')
    if sha(path) != lock.read_text(encoding='utf-8').strip():
        raise ValueError('Frozen source protocol identity mismatch')
    p = json.loads(path.read_text(encoding='utf-8'))
    if p['protocol_name'] != 'causal_source_development_v1' or p['data_version'] != VERSION:
        raise ValueError('Unsupported registered source protocol')
    return p


def source_metadata(root, split, dataset):
    frame = pd.read_csv(root / 'ledgers' / (split + '_included_windows.csv'),
        dtype={'case_uid': str, 'association_group': str, 'domain': str, 'split': str},
        keep_default_na=False, low_memory=False)
    if len(frame) != len(dataset) or not np.array_equal(frame.compact_index, np.arange(len(frame))):
        raise ValueError('Current compact metadata population/order mismatch')
    if not frame.domain.isin(['SIL','HIL']).all() or not frame.split.eq(split).all() or not frame.role.eq(split).all():
        raise ValueError('Non-source split/role in source development')
    for key in ['input_valid','label_valid','supervised_valid']:
        if frame[key].dtype != bool or not frame[key].all():
            raise ValueError('Invalid sample reached source forward population')
    if not np.array_equal(frame.original_cache_index, dataset.original_cache_indices.numpy()):
        raise ValueError('Original cache IDs differ from admitted reader')
    labels = np.asarray(dataset.labels, dtype=np.int64)
    if not np.array_equal(labels, frame.qualified_y) or not np.isin(labels,[0,1]).all():
        raise ValueError('Current labels/metadata differ')
    origin = np.where(labels == 1, FAULT, np.where(frame.case_level2_label == 'normal', ORIGINAL, PREFAULT))
    expected = np.where(labels == 1, 'fault_or_unknown', np.where(frame.case_level2_label == 'normal', 'normal_flight', PREFAULT))
    if not np.array_equal(expected, frame.normal_origin):
        raise ValueError('Label-derived normal sources were not rebuilt consistently')
    frame['normal_origin'] = origin; frame['label'] = labels
    frame['flight_status'] = frame.case_uid.str.split('/').str[1]
    frame['cache_index'] = np.arange(len(frame))
    if not frame.case_level2_label.isin(['normal','motor','sensor']).all():
        raise ValueError('Unknown source family')
    return frame


def prepare(root, protocol):
    # Only explicit source paths are read/hashed. Never iterate root/cache/test or Real.
    for relative, expected in protocol['data_identities'].items():
        if sha(root / relative) != expected: raise ValueError('Frozen data identity mismatch: ' + relative)
    for relative, expected in protocol['code_identities'].items():
        if sha(ROOT / relative) != expected: raise ValueError('Frozen code identity mismatch: ' + relative)
    ds = {s: CausalCacheDataset(root, expected_version=VERSION, split=s, purpose='source_development') for s in ['train','val']}
    frame = {s: source_metadata(root,s,ds[s]) for s in ds}
    if set(frame['train'].case_uid) & set(frame['val'].case_uid): raise ValueError('Source case leakage')
    if set(frame['train'].association_group) & set(frame['val'].association_group): raise ValueError('Simulation pair group crosses source train/val')
    weights = {arm: sampling_weights(np.asarray(ds['train'].labels), frame['train'].normal_origin.to_numpy(), arm=='balanced_source_normal')
               for arm in protocol['arms']}
    tables = {arm: probability_table(w,np.asarray(ds['train'].labels),frame['train'].normal_origin.to_numpy()) for arm,w in weights.items()}
    for key in ['total_weight','normal_probability','fault_probability']:
        if not np.isclose(tables[protocol['arms'][0]][key],tables[protocol['arms'][1]][key],rtol=0,atol=1e-10):
            raise ValueError('Class probability/total mass changed in sampler contrast')
    if not np.isclose(tables['balanced_source_normal']['origins'][ORIGINAL]['probability'],
        tables['balanced_source_normal']['origins'][PREFAULT]['probability'],rtol=0,atol=1e-12):
        raise ValueError('Source normal pools not balanced')
    if len(ds['train'])!=protocol['data']['train_windows'] or len(ds['val'])!=protocol['data']['val_windows']:
        raise ValueError('Wrong versioned source population')
    if any(int(frame[s].case_uid.nunique())!=protocol['data'][s+'_cases'] for s in ['train','val']):
        raise ValueError('Wrong current source case coverage')
    return ds, frame, weights, tables


class IndexedSource(Dataset):
    def __init__(self, source, indices=None):
        self.source=source
        self.indices=np.arange(len(source),dtype=np.int64) if indices is None else np.asarray(indices,dtype=np.int64)
    def __len__(self): return len(self.indices)
    def __getitem__(self,index):
        idx=int(self.indices[index]);x,y=self.source[idx]
        return x,y,idx


def evaluate(model, loader, device, metadata):
    model.eval();probabilities=[];actual=[];indices=[]
    with torch.no_grad():
        for x,y,idx in loader:
            p=model(x.to(device)).softmax(1).cpu().numpy()
            if not np.isfinite(p).all():raise FloatingPointError('Nonfinite source validation')
            probabilities.append(p);actual.extend(y.tolist());indices.extend(idx.tolist())
    probs=np.concatenate(probabilities);labels=np.asarray(actual,dtype=np.int64)
    if indices!=metadata.cache_index.tolist():raise ValueError('Validation metadata ordering differs')
    return stratified_metrics(labels,probs.argmax(1),metadata),probs,labels


def environment():
    return {'python':sys.version,'torch':str(torch.__version__),'numpy':np.__version__,
        'cuda_available':torch.cuda.is_available(),'torch_cuda':torch.version.cuda}


def run(args,p):
    if args.output.exists():raise FileExistsError('New run directory required; never overwrite')
    if args.arm not in p['arms'] or args.seed not in p['seeds']:raise ValueError('Unregistered arm/seed')
    if not args.cpu_check and not torch.cuda.is_available():
        raise RuntimeError('Formal source retraining BLOCKED_NO_CUDA; no CPU fallback or resource allocation')
    if not args.cpu_check:
        actual=environment()
        for key,value in p['formal_environment'].items():
            if actual[key]!=value:raise RuntimeError('Formal compute environment mismatch: '+key)
    ds,frame,weights,tables=prepare(args.data,p)
    torch.set_num_threads(p['training']['cpu_threads'])
    torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.allow_tf32=True;torch.backends.cuda.matmul.allow_tf32=False
    np.random.seed(args.seed);torch.manual_seed(args.seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(args.seed)
    cfg=p['training'];device=torch.device('cpu' if args.cpu_check else 'cuda:0')
    model=OnlineKGMSTCA(in_channels=26,hidden_size=cfg['hidden_size'],num_layers=cfg['num_layers'],
        num_classes=2,dropout=cfg['dropout'],columns=p['data']['columns']).float()
    if sum(v.numel() for v in model.parameters())!=cfg['parameters']:raise ValueError('Unchanged model size mismatch')
    initial={name:value.detach().clone() for name,value in model.state_dict().items()}
    out=args.output;out.mkdir(parents=True,exist_ok=False);start=time.perf_counter()
    report={'status':'RUNNING_CPU_PIPELINE_CHECK' if args.cpu_check else 'RUNNING_FINITE_SOURCE_RETRAIN',
        'protocol_sha256':sha(args.protocol),'seed':args.seed,'arm':args.arm,'cpu_check':args.cpu_check,
        'environment':environment(),'device':str(device),'initial_state_digest':state_digest(initial),
        'data_version':VERSION,'sampling':tables[args.arm],'source_train_windows':len(ds['train']),
        'source_val_windows_full':len(ds['val']),'no_final_test_or_Real_reads':True,'performance_claim':False,
        'started_at':datetime.now(timezone.utc).isoformat()}
    write(out/'report.json',report);save_state(out/'initial.pt',initial)
    for name,path in [('protocol.json',args.protocol),('norm_stats.json',args.data/'norm_stats.json'),('trainer.py',Path(__file__))]:
        shutil.copyfile(path,out/name)
    model.to(device);profile=p['cpu_check'] if args.cpu_check else cfg
    draws=profile['draws_per_epoch'] if args.cpu_check else len(ds['train'])
    epochs=profile['epochs'] if args.cpu_check else cfg['max_epochs'];batch=profile['batch_size']
    metadata=frame['val']
    if args.cpu_check:
        picked=[]
        for _,g in metadata.groupby('case_uid',sort=False):
            picks=np.unique(np.linspace(0,len(g)-1,min(profile['val_windows_per_case'],len(g))).round().astype(int));picked.extend(g.iloc[picks].cache_index.tolist())
        metadata=metadata.iloc[picked].copy().reset_index(drop=True)
    sampler=WeightedRandomSampler(weights[args.arm],draws,replacement=True,generator=torch.Generator().manual_seed(args.seed+11))
    trainloader=DataLoader(IndexedSource(ds['train']),batch_size=batch,sampler=sampler,num_workers=0,
        generator=torch.Generator().manual_seed(args.seed+101))
    valloader=DataLoader(IndexedSource(ds['val'],metadata.cache_index.to_numpy()),
        batch_size=profile['validation_batch_size'] if args.cpu_check else p['validation']['batch_size'],shuffle=False,num_workers=0,
        generator=torch.Generator().manual_seed(args.seed+101))
    criterion=FocalLoss(gamma=cfg['loss_gamma']);optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay'])
    selection={'best_score':-float('inf'),'best_epoch':None,'significant_reference':-float('inf'),'bad_epochs':0};history=[]
    try:
        for epoch in range(1,epochs+1):
            lr=get_warmup_cosine_lr(epoch-1,cfg['warmup_epochs'],cfg['max_epochs'],cfg['lr'],cfg['min_lr'])
            for group in optimizer.param_groups:group['lr']=lr
            model.train();losses=[];realized=Counter()
            for x,y,indices in trainloader:
                if not torch.isfinite(x).all() or not torch.isin(y,torch.tensor([0,1])).all():raise ValueError('Invalid sample before model forward')
                optimizer.zero_grad(set_to_none=True);loss=criterion(model(x.to(device)),y.to(device))
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite source loss')
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['gradient_clip'],error_if_nonfinite=True);optimizer.step()
                losses.append(float(loss.detach().cpu()));realized.update(frame['train'].normal_origin.iloc[indices.numpy()].tolist())
            metrics,probs,labels=evaluate(model,valloader,device,metadata)
            selection,best,stop=selection_step(selection,float(metrics['overall']['macro_f1']),epoch,p['validation']['early_stop'])
            if best:save_state(out/'best.pt',model.state_dict())
            save_state(out/'last.pt',model.state_dict())
            history.append({'epoch':epoch,'lr':lr,'train_loss':float(np.mean(losses)),
                'gradient_steps':len(losses),'realized_training_origins':dict(realized),'source_validation':metrics,'selection':dict(selection)})
            write(out/'history.json',history)
            print('CAUSAL_SOURCE_JOB',args.arm,'CPU_CHECK' if args.cpu_check else 'FORMAL',epoch,'/',epochs,'loss',float(np.mean(losses)),'source_val_macro_f1',metrics['overall']['macro_f1'],flush=True)
            if stop:break
        model.load_state_dict(torch.load(out/'best.pt',map_location=device,weights_only=True))
        reloaded,probabilities,labels=evaluate(model,valloader,device,metadata)
        if reloaded!=history[selection['best_epoch']-1]['source_validation']:raise RuntimeError('Best checkpoint reload changed validation')
        if not any(not torch.equal(initial[name],v.detach().cpu()) for name,v in model.named_parameters()):raise RuntimeError('No trainable parameters changed')
        for rel,expected in p['data_identities'].items():
            if sha(args.data/rel)!=expected:raise ValueError('Data changed during training')
        if sha(args.protocol)!=report['protocol_sha256']:raise ValueError('Frozen config changed during training')
        np.savez_compressed(out/'source_validation_predictions.npz',probabilities=probabilities,labels=labels,
            compact_indices=metadata.cache_index.to_numpy(dtype=np.int64),original_indices=metadata.original_cache_index.to_numpy(dtype=np.int64))
        report.update(status='PASS_CPU_PIPELINE_CHECK_NOT_FORMAL' if args.cpu_check else 'COMPLETE_FINITE_SOURCE_RETRAIN',
            epochs=len(history),gradient_steps=sum(h['gradient_steps'] for h in history),best_epoch=selection['best_epoch'],
            validation_windows=len(metadata),checkpoint_reload_matches=True,parameters_changed=True,
            best_checkpoint_sha256=sha(out/'best.pt'),source_validation=reloaded,elapsed_s=time.perf_counter()-start,
            finished_at=datetime.now(timezone.utc).isoformat(),performance_claim=not args.cpu_check,
            result_scope='Source validation development only, never final-test/Real or causal-live deployment clearance')
        write(out/'report.json',report)
        return report
    except Exception as exc:
        report.update(status='FAILED',error=repr(exc),elapsed_s=time.perf_counter()-start);write(out/'report.json',report);raise


def preflight(args,p):
    if args.output.exists():raise FileExistsError('Preserve existing preflight')
    ds,frames,weights,tables=prepare(args.data,p)
    root=args.output;root.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(root/'sampling_weights.npz',**{arm:w.numpy() for arm,w in weights.items()},
        original_cache_index=ds['train'].original_cache_indices.numpy(),labels=np.asarray(ds['train'].labels,dtype=np.int64))
    write(root/'sampling_probabilities.json',tables)
    write(root/'preflight.json',{'status':'PASS_FROZEN_CAUSAL_SOURCE_PREFLIGHT','protocol_sha256':sha(args.protocol),
        'data_version':VERSION,'environment':environment(),'formal_training_blocked_no_cuda':not torch.cuda.is_available(),
        'sampling_weights_sha256':sha(root/'sampling_weights.npz'),'sampling':tables,
        'source_train_windows':len(ds['train']),'source_val_windows':len(ds['val']),
        'source_train_cases':int(frames['train'].case_uid.nunique()),'source_val_cases':int(frames['val'].case_uid.nunique()),
        'train_val_case_and_simulation_pair_isolated':True,'test_or_Real_data_read':False})
    print(json.dumps({'status':'PASS_FROZEN_CAUSAL_SOURCE_PREFLIGHT','sampling':tables,'cuda_available':torch.cuda.is_available()},ensure_ascii=False,indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=['preflight','run'])
    parser.add_argument('--protocol',type=Path,required=True);parser.add_argument('--data',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--arm',choices=['uniform_replacement','balanced_source_normal']);parser.add_argument('--seed',type=int,default=42);parser.add_argument('--cpu-check',action='store_true')
    args=parser.parse_args();p=load_protocol(args.protocol)
    if args.command=='preflight':preflight(args,p)
    else:run(args,p)


if __name__=='__main__':main()
