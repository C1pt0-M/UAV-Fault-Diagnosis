"""Eight fixed M3 jobs, exact shared anchors, original source-only protocol."""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'packages/uav_fault'), str(ROOT/'scripts/research')]
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import train_causal_source_v1 as source
from train_source_normal_sampling import state_digest, selection_step
from uav_fault.data.m3_features import M3CandidateDataset
from uav_fault.models.cnn.mstca_m3 import VARIANTS, build_m3_model
from analyze_13_feature_screening import observed_motion
from run_normal_replay_v1 import classification


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def tensor_digest(value):
    return hashlib.sha256(value.contiguous().cpu().numpy().tobytes()).hexdigest()


def protocol(path):
    execution = read(path)
    if source.sha(path) != path.with_suffix('.json.sha256').read_text().strip():
        raise ValueError('M3 execution lock mismatch')
    parent_path = ROOT/execution['parent_path']
    if source.sha(parent_path) != execution['parent_sha256']:
        raise ValueError('Frozen preflight changed')
    p = read(parent_path)
    for name, expected in execution['code_identities'].items():
        if source.sha(ROOT/name) != expected:
            raise ValueError('Execution code identity differs: '+name)
    for name, expected in p['code_identities'].items():
        if name not in execution['authorized_code_changes'] and source.sha(ROOT/name) != expected:
            raise ValueError('Historical pinned code changed: '+name)
    if execution['jobs'] != list(VARIANTS) or execution['overall_delta_pp'] != 0 or any(execution['per_fault_delta_pp'].values()) or execution['domain_overall_delta_pp'] != {'SIL':0., 'HIL':0.}:
        raise ValueError('Authorized eight jobs/recall constraints changed')
    return p, execution


def build(p, variant, anchor):
    spec = p['initialization']['source']
    if source.sha(anchor) != spec['sha256']:
        raise ValueError('Untrained M0 anchor changed')
    raw = torch.load(anchor, map_location='cpu', weights_only=True)
    common = {'encoder.'+k[len('raw_encoder.'):]:v for k,v in raw.items() if k.startswith('raw_encoder.')}
    common.update({k:v for k,v in raw.items() if k.startswith('head.')})
    if state_digest(raw) != spec['state_digest'] or state_digest(common) != p['initialization']['shared_state_digest']:
        raise ValueError('M0 public tensors differ')
    model = build_m3_model(variant, common)
    if variant != 'C0' and state_digest(model.adapter.state_dict()) != p['initialization']['adapter_initial_state_digest_local']:
        raise ValueError('Loaded adapter differs from local frozen tensors')
    return model


def prepare(data, features, p):
    ds = {}; frames = {}
    for split in ['train', 'val']:
        ds[split] = M3CandidateDataset(data, features, ROOT/p['normalization']['path'], split=split,
                                      normalization_sha256=p['normalization']['sha256'])
        full = source.source_metadata(data, split, ds[split].source.base)
        frame = full.iloc[ds[split].base_indices.numpy()].copy().reset_index(drop=True)
        frame['cache_index'] = np.arange(len(frame))
        frame['fault_directory'] = frame.case_uid.str.split('/').str[2]
        expected = p['data']['source_split_identities'][split]
        if tensor_digest(ds[split].original_cache_indices) != expected['original_ids_sha256'] or tensor_digest(torch.tensor(ds[split].labels)) != expected['labels_sha256']:
            raise ValueError('Frozen original IDs/labels changed')
        if not np.array_equal(frame.original_cache_index, ds[split].original_cache_indices):
            raise ValueError('Metadata alignment changed')
        frames[split] = frame
    if set(frames['train'].case_uid)&set(frames['val'].case_uid) or set(frames['train'].association_group)&set(frames['val'].association_group):
        raise ValueError('Case/pair crossing')
    weights = source.sampling_weights(np.asarray(ds['train'].labels), frames['train'].normal_origin.to_numpy(), True)
    if tensor_digest(weights) != p['sampling']['weights_sha256']:
        raise ValueError('Sampler weights changed')
    return ds, frames, weights


class Indexed(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, i):
        return (*self.dataset[i], i)
    def __getitems__(self, indices):
        return [(*item, int(i)) for item,i in zip(self.dataset.__getitems__(indices),indices)]


def finite(value, label):
    if not torch.isfinite(value).all():
        raise FloatingPointError('Nonfinite '+label)


def forward(model, variant, x, f):
    finite(x, 'Base26 input'); finite(f, 'all13 candidate input')
    z = model(x) if variant == 'C0' else model(x, f)
    finite(z, 'logits')
    return z


def binary_metrics(y, pred):
    return classification(np.bincount(y*2+pred, minlength=4).reshape(2,2))


def ap_score(y, score):
    if not len(y) or not y.sum():
        return None
    order = np.argsort(-score, kind='stable'); yy=y[order]; ss=score[order]
    ends=np.r_[np.flatnonzero(ss[:-1]!=ss[1:])+1,len(ss)]
    tp=np.cumsum(yy)[ends-1]; recall=tp/yy.sum()
    return float(np.sum(np.diff(np.r_[0.,recall])*(tp/ends)))


def metric_report(y, probabilities, frame):
    pred=probabilities.argmax(1);score=probabilities[:,1]
    metrics=source.stratified_metrics(y,pred,frame)
    def part(indices):
        indices=np.asarray(indices,dtype=int)
        m=binary_metrics(y[indices],pred[indices]);m['windows']=len(indices)
        return m
    tp=int(((y==1)&(pred==1)).sum());fp=int(((y==0)&(pred==1)).sum())
    metrics['fault_precision']=tp/(tp+fp) if tp+fp else None
    metrics['average_precision']=ap_score(y,score)
    metrics['six_faults']={str(k):part(g.index) for k,g in frame[frame.label==1].groupby('fault_directory')}
    metrics['domains']={str(k):part(g.index) for k,g in frame.groupby('domain')}
    metrics['faults_by_domain']={str(domain):{str(k):part(g.index) for k,g in group[group.label==1].groupby('fault_directory')} for domain,group in frame.groupby('domain')}
    if 'observed_state' in frame:
        metrics['normal_observed_states']={str(k):part(g.index) for k,g in frame[frame.label==0].groupby('observed_state')}
    hover=frame[(frame.normal_origin=='original_normal_case')&(frame.flight_status=='hover')]
    metrics['hover_crop_segments']={name:part(g.index) for name,g in [('first30',hover[(hover.window_end_row-1)*.02<=30]),('after30',hover[(hover.window_end_row-1)*.02>30])]}
    return metrics


def evaluate(model, variant, loader, device, frame):
    model.eval(); probs=[];labels=[];seen=[]
    with torch.inference_mode():
        for x,f,y,idx in loader:
            q=forward(model,variant,x.to(device),f.to(device)).softmax(1)
            finite(q,'validation probabilities')
            probs.append(q.cpu().numpy());labels.extend(y.tolist());seen.extend(idx.tolist())
    if seen!=frame.cache_index.tolist() or labels!=frame.label.tolist():
        raise ValueError('Validation order/labels changed')
    probability=np.concatenate(probs);y=np.asarray(labels,dtype=np.int64)
    return metric_report(y,probability,frame),probability,y


def configure(p):
    torch.set_num_threads(p['training']['cpu_threads'])
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.allow_tf32=True;torch.backends.cuda.matmul.allow_tf32=False
    np.random.seed(42);torch.manual_seed(42)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(42)


def add_states(dataset, frame):
    result=frame.copy();states=[]
    base=dataset.source.base
    columns=read(ROOT/'configs/experiments/M3_preflight_v1.json')['data']['base_columns']
    for i in range(len(dataset)):
        raw=base._x[int(dataset.base_indices[i])].numpy().T
        states.append(observed_motion(raw,columns)['observed_state'])
    result['observed_state']=states
    return result


def run(args):
    p,e=protocol(args.protocol)
    gate=read(args.gate)
    if gate['status']!='PASS_M3_FIXED_ANCHOR_AND_ENTRYPOINTS' or gate['execution_sha256']!=source.sha(args.protocol):
        raise RuntimeError('M3 entry gates incomplete')
    if args.variant not in e['jobs'] or args.output.exists():
        raise ValueError('Unregistered job or output already exists')
    if not torch.cuda.is_available():raise RuntimeError('Formal M3 requires CUDA')
    for k,v in p['formal_environment'].items():
        if source.environment()[k]!=v:raise RuntimeError('Formal environment differs: '+k)
    configure(p);ds,frames,weights=prepare(args.data,args.features,p)
    frames['val']=add_states(ds['val'],frames['val'])
    model=build(p,args.variant,args.initial_anchor).to('cuda:0');initial={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    sampler=WeightedRandomSampler(weights,len(ds['train']),replacement=True,generator=torch.Generator().manual_seed(53))
    loader=DataLoader(Indexed(ds['train']),batch_size=512,sampler=sampler,num_workers=0,generator=torch.Generator().manual_seed(143))
    validation=DataLoader(Indexed(ds['val']),batch_size=512,shuffle=False,num_workers=0,generator=torch.Generator().manual_seed(143))
    expected_draws=read(ROOT/p['sampling']['local_sampler_check_path'])['reference_epoch_originalIDs_SHA256']
    criterion=source.FocalLoss(gamma=2.0);optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4)
    args.output.mkdir(parents=True,exist_ok=False);out=args.output;started=time.perf_counter()
    report=dict(status='RUNNING_FORMAL_M3',variant=args.variant,seed=42,arm='balanced_source_normal',execution_sha256=source.sha(args.protocol),parent_sha256=e['parent_sha256'],initial_state_digest=state_digest(initial),parameters=sum(v.numel() for v in model.parameters()),started_at=datetime.now(timezone.utc).isoformat(),environment=source.environment(),optimizer_steps=0,Test_Real_used=False)
    source.save_state(out/'initial.pt',initial);source.write(out/'report.json',report)
    selection=dict(best_score=-float('inf'),best_epoch=None,significant_reference=-float('inf'),bad_epochs=0);history=[]
    try:
        for epoch in range(1,101):
            lr=source.get_warmup_cosine_lr(epoch-1,5,100,3e-4,1e-6)
            for group in optimizer.param_groups:group['lr']=lr
            model.train();losses=[];draw=[];realized=Counter()
            for x,f,y,idx in loader:
                if not torch.isin(y,torch.tensor([0,1])).all():raise ValueError('Invalid label')
                draw.extend(idx.tolist());realized.update(frames['train'].normal_origin.iloc[idx.numpy()].tolist())
                optimizer.zero_grad(set_to_none=True);z=forward(model,args.variant,x.to('cuda:0'),f.to('cuda:0'));loss=criterion(z,y.to('cuda:0'));finite(loss,'loss')
                loss.backward()
                for name,value in model.named_parameters():
                    if value.grad is not None:finite(value.grad,'gradient '+name)
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
                for name,value in model.state_dict().items():finite(value,'updated state '+name)
                report['optimizer_steps']+=1;losses.append(float(loss.detach().cpu()))
            draw_digest=tensor_digest(ds['train'].original_cache_indices[draw])
            if draw_digest!=expected_draws[epoch-1] or len(losses)!=297:raise ValueError('Per-epoch sampling ID digest or step count differs')
            metrics,probs,labels=evaluate(model,args.variant,validation,'cuda:0',frames['val'])
            selection,best,stop=selection_step(selection,metrics['overall']['macro_f1'],epoch,p['validation']['early_stop'])
            if best:
                source.save_state(out/'best.pt',model.state_dict())
                np.savez_compressed(out/'best_validation_predictions.npz',probabilities=probs,labels=labels,original_indices=ds['val'].original_cache_indices.numpy())
            source.save_state(out/'last.pt',model.state_dict())
            history.append(dict(epoch=epoch,lr=lr,train_loss=float(np.mean(losses)),gradient_steps=len(losses),sampling_original_ID_sha256=draw_digest,realized_training_origins=dict(realized),source_validation=metrics,selection=selection.copy()))
            source.write(out/'history.json',history)
            source.write(out/'progress.json',dict(variant=args.variant,epoch=epoch,best_epoch=selection['best_epoch'],best_macro_f1=selection['best_score'],optimizer_steps=report['optimizer_steps'],last_metrics=metrics,updated_at=datetime.now(timezone.utc).isoformat()))
            print('M3_EPOCH',args.variant,epoch,'macro_f1',metrics['overall']['macro_f1'],'recall',metrics['overall']['fault_recall'],flush=True)
            if stop:break
        model.load_state_dict(torch.load(out/'best.pt',map_location='cuda:0',weights_only=True))
        metrics,probs,labels=evaluate(model,args.variant,validation,'cuda:0',frames['val'])
        if metrics!=history[selection['best_epoch']-1]['source_validation']:raise ValueError('Best reload metrics differ')
        if not any(not torch.equal(initial[k],v.detach().cpu()) for k,v in model.named_parameters()):raise ValueError('No updates')
        protocol(args.protocol)
        report.update(status='COMPLETE_FORMAL_M3',epochs=len(history),best_epoch=selection['best_epoch'],source_validation=metrics,best_sha256=source.sha(out/'best.pt'),elapsed_s=time.perf_counter()-started,finished_at=datetime.now(timezone.utc).isoformat())
        source.write(out/'report.json',report)
    except Exception as exc:
        report.update(status='FAILED_STOPPED',error=repr(exc),traceback=traceback.format_exc());source.write(out/'report.json',report);raise


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ['protocol','data','features','initial-anchor','gate','output']:ap.add_argument('--'+name,type=Path,required=True)
    ap.add_argument('--variant',choices=list(VARIANTS),required=True)
    run(ap.parse_args())
