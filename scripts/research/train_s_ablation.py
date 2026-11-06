"""Frozen S-ablation r1 runner. Local use: --phase cv --cpu-smoke only.

No source.prepare/CausalCacheDataset: those would load val or scan full X.
Val identity files are byte-hashed only in CV; held-out X is accessed only after
training. Formal execution requires CUDA, exact environment and an external,
SHA-pinned implementation manifest prepared before the first formal run.
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'packages/uav_fault'), str(ROOT/'scripts/research')]
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from uav_fault.models.cnn.ablation_factory import build_variant, initialize_variant, _state_digest
from uav_fault.models.cnn.focal_loss import FocalLoss
from uav_fault.models.train_online_detector import get_warmup_cosine_lr
from uav_fault.model_contract import RFLY_SEQ_COLUMNS_V2

PROTOCOL_SHA = '0764635a75de22963b573180d1616b6da97e038b76565c247d12a2b0a6f61005'
DEFAULT_PROTOCOL = ROOT/'configs/experiments/s_ablation_mstca_v1.json'
BASELINE = ROOT/'configs/experiments/causal_pure_mstca_m0_v1.json'
VERSION = 'rflymad_causal_publication_replay_v1'
IMPLEMENTATION_FILES = ('scripts/research/train_s_ablation.py', 'scripts/research/evaluate_s_ablation.py',
                        'scripts/research/preflight_s_ablation.py', 'tests/test_s_ablation.py')
SMOKE_LIMITS = {'fit': 512, 'inner': 256, 'heldout': 256}
SMOKE_ROOT = ROOT/'artifacts/s_ablation_mstca_v1_smoke_20261106'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value, *, update=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    target = path.with_name(path.name+'.partial') if update else path
    with target.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    if update:
        target.replace(path)


def save_state(path, state):
    path = Path(path)
    temporary = path.with_name(path.name+'.partial')
    with temporary.open('xb') as stream:
        torch.save({k:v.detach().cpu().clone() for k,v in state.items()}, stream)
    temporary.replace(path)


def load_protocol(path=DEFAULT_PROTOCOL):
    path = Path(path)
    actual = sha(path)
    if actual != path.with_suffix(path.suffix+'.sha256').read_text(encoding='utf-8').strip() or actual != PROTOCOL_SHA:
        raise ValueError('Frozen S protocol/lock mismatch')
    p = read_json(path)
    pins = {}
    for relative, expected in p['code_identities'].items():
        if relative == 'new_scripts':
            continue
        pins[relative] = sha(ROOT/relative)
        if pins[relative] != expected:
            raise ValueError('Frozen code identity mismatch: '+relative)
    baseline = read_json(BASELINE)
    # Include the actual sampling/LR/shared helpers, not just the eight S pins.
    for relative, expected in baseline['code_identities'].items():
        pins[relative] = sha(ROOT/relative)
        if pins[relative] != expected:
            raise ValueError('Historical shared helper identity mismatch: '+relative)
    return p, pins


def implementation_manifest():
    return {'protocol_sha256': PROTOCOL_SHA,
            'files': {relative:sha(ROOT/relative) for relative in IMPLEMENTATION_FILES}}


def verify_data(root, *, smoke=False):
    """Nine identities; val files must NEVER be parsed here, only byte-hashed."""
    identities = read_json(BASELINE)['data_identities']
    matched, skipped = {}, []
    for relative, expected in identities.items():
        if smoke and relative == 'cache/train.pt':
            skipped.append(relative)
            continue
        actual = sha(Path(root)/relative)
        if actual != expected:
            raise ValueError('Frozen data identity mismatch: '+relative)
        matched[relative] = actual
    return {'matched_sha256': matched, 'skipped_sha256': skipped,
            'val_access': 'SHA256_BYTES_ONLY_NO_PARSE_OR_LOAD'}


def hash_group(group, prefix, modulus):
    if not isinstance(group, str) or not group:
        raise ValueError('association_group must be a nonempty string, never numeric/coerced')
    return int(hashlib.sha256((prefix+group).encode()).hexdigest(),16) % modulus


def split_ledger(frame, fold, phase='cv'):
    groups = frame.association_group
    if not groups.map(lambda x:isinstance(x,str) and bool(x)).all():
        raise ValueError('Invalid association_group strings')
    unique = groups.unique()
    outer_map = {g:hash_group(g,'h1cv|',3) for g in unique}
    inner_map = {g:hash_group(g,'s_ablation_inner_v1|',10)==0 for g in unique}
    outer = groups.map(outer_map).to_numpy(dtype=np.int64)
    inner = groups.map(inner_map).to_numpy(dtype=bool)
    available = outer != fold if phase == 'cv' else np.ones(len(frame),dtype=bool)
    return {'fit':np.flatnonzero(available & ~inner), 'inner':np.flatnonzero(available & inner),
            'heldout':np.flatnonzero(outer==fold) if phase=='cv' else np.array([],dtype=np.int64)}, outer


def read_ledger(root, split='train'):
    if split not in ('train','val'):
        raise ValueError('Only explicit source train/val ledgers supported')
    frame = pd.read_csv(Path(root)/'ledgers'/f'{split}_included_windows.csv',
        dtype={'association_group':str,'case_uid':str,'domain':str,'split':str,'role':str},
        keep_default_na=False, low_memory=False)
    if not frame.compact_index.eq(np.arange(len(frame))).all() or not frame.original_cache_index.is_unique:
        raise ValueError('Invalid compact/original ledger ordering')
    if not frame.domain.isin(['SIL','HIL']).all() or not frame.split.eq(split).all() or not frame.role.eq(split).all():
        raise ValueError('Non-source split/role in ledger')
    for key in ('input_valid','label_valid','supervised_valid'):
        if frame[key].dtype != bool or not frame[key].all():
            raise ValueError('Unqualified row in compact source ledger')
    if not frame.qualified_y.isin([0,1]).all():
        raise ValueError('Unknown source label')
    expected = np.where(frame.qualified_y==1,'fault_or_unknown',
                        np.where(frame.case_level2_label=='normal','normal_flight','normal_part_of_fault_case'))
    if not np.array_equal(expected,frame.normal_origin):
        raise ValueError('Normal-origin/label mismatch')
    return frame


def fold_sizes(frame, p):
    sizes = {}
    for fold in range(3):
        indices, _ = split_ledger(frame,fold)
        actual = {k:len(v) for k,v in indices.items()}
        expected = dict(p['data']['inner_selection']['expected'][f'fold{fold}'],
                        heldout=p['data']['outer_folds']['expected_windows'][fold])
        if actual != expected:
            raise ValueError(f'Fold {fold} population mismatch: {actual} != {expected}')
        sizes[str(fold)] = actual
    return sizes


def take_smoke(frame, indices, limit):
    """Deterministic metadata-only, origin-stratified subset; no X access."""
    indices = np.asarray(indices,dtype=np.int64)
    if len(indices) <= limit:
        return indices
    part = frame.iloc[indices]
    pools = [indices[np.flatnonzero(part.normal_origin.to_numpy()==origin)]
             for origin in sorted(part.normal_origin.unique())]
    chosen = []
    for i,pool in enumerate(pools):
        count = min(len(pool),limit//len(pools)+(i<limit%len(pools)))
        chosen.extend(pool[np.linspace(0,len(pool)-1,count).round().astype(int)].tolist())
    return np.asarray(sorted(chosen),dtype=np.int64)


class CacheWindows(Dataset):
    """Selected X only. Production reader's full-X validation is intentionally avoided."""
    def __init__(self, root, frame, indices, *, split='train'):
        root = Path(root)
        stats = read_json(root/'norm_stats.json')
        contract = read_json(root/'input_contract.json')
        if stats['fit_split'] != 'source_train_only' or stats['columns'] != RFLY_SEQ_COLUMNS_V2 or contract['columns'] != stats['columns']:
            raise ValueError('Invalid frozen normalization/schema')
        if stats['input_contract_sha256'] != sha(root/'input_contract.json'):
            raise ValueError('Normalization contract identity mismatch')
        payload = torch.load(root/'cache'/f'{split}.pt',map_location='cpu',weights_only=True,mmap=True)
        if payload['split'] != split or payload['data_version'] != VERSION or payload['normalization_applied'] is not False:
            raise ValueError('Cache version/split/preprocessing mismatch')
        if payload['norm_sha256'] != sha(root/'norm_stats.json') or payload['input_contract_sha256'] != sha(root/'input_contract.json'):
            raise ValueError('Cache normalization/contract identity mismatch')
        if payload['x'].dtype != torch.float32 or tuple(payload['x'].shape) != (len(frame),26,150):
            raise ValueError('Cache X layout mismatch')
        positions = np.asarray(indices,dtype=np.int64)
        self.x = payload['x'][positions].clone()
        self.y = payload['y'][positions].clone()
        self.ids = payload['original_cache_index'][positions].clone()
        if self.y.dtype != torch.int64 or self.ids.dtype != torch.int64:
            raise ValueError('Cache label/index dtype mismatch')
        if not np.array_equal(self.y.numpy(),frame.qualified_y.to_numpy()[positions]) or not np.array_equal(self.ids.numpy(),frame.original_cache_index.to_numpy()[positions]):
            raise ValueError('Selected labels/IDs do not match ledger')
        self.mean = torch.tensor(stats['mean'],dtype=torch.float32).view(26,1)
        self.std = torch.tensor(stats['std'],dtype=torch.float32).view(26,1)
        if not torch.isfinite(self.mean).all() or not torch.isfinite(self.std).all() or not (self.std>0).all():
            raise ValueError('Invalid normalization coefficients')
        if not torch.isfinite((self.x-self.mean)/self.std).all():
            raise ValueError('Nonfinite selected input before any BN/forward')
    def __len__(self):
        return len(self.y)
    def __getitem__(self,index):
        return (self.x[index]-self.mean)/self.std,self.y[index]


def class_balanced_ce(logits, labels):
    losses = F.cross_entropy(logits.double(),labels,reduction='none')
    if any(not bool((labels==c).any()) for c in (0,1)):
        raise ValueError('Class-balanced CE requires both classes')
    return .5*(losses[labels==0].mean()+losses[labels==1].mean())


def selection_step(state, loss, epoch):
    if not math.isfinite(loss):
        raise FloatingPointError('Nonfinite inner selection loss')
    result = dict(state)
    best = loss < state['best_loss']
    if best:
        result.update(best_loss=loss,best_epoch=epoch)
    significant = loss < state['reference']*.995
    if significant:
        result.update(reference=loss,bad_epochs=0)
    elif epoch > 30:
        result['bad_epochs'] += 1
    return result,best,epoch > 30 and result['bad_epochs'] >= 20


def inner_ce(model, loader, device):
    model.eval()
    sums = np.zeros(2,dtype=np.float64)
    counts = np.zeros(2,dtype=np.int64)
    with torch.inference_mode():
        for x,y in loader:
            z = model(x.to(device))
            losses = F.cross_entropy(z.double(),y.to(device),reduction='none').cpu().numpy()
            for c in (0,1):
                mask = y.numpy()==c
                sums[c] += losses[mask].sum(dtype=np.float64)
                counts[c] += mask.sum()
    if (counts==0).any():
        raise ValueError('Inner selection requires both classes')
    value = float(.5*np.sum(sums/counts))
    if not math.isfinite(value):
        raise FloatingPointError('Nonfinite inner CE')
    return value


def golden_digests():
    tree = ast.parse((ROOT/'tests/test_mstca_ablation_factory.py').read_text(encoding='utf-8'))
    for node in tree.body:
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='GOLDEN_DIGESTS' for t in node.targets):
            return ast.literal_eval(node.value)
    raise ValueError('Pinned golden digest table not found')


def environment():
    return {'python':sys.version,'torch':str(torch.__version__),'numpy':np.__version__,
            'torch_cuda':torch.version.cuda,'cuda_available':torch.cuda.is_available(),
            'cublas_workspace_config':os.environ.get('CUBLAS_WORKSPACE_CONFIG')}


def require_formal_environment(actual=None):
    actual = environment() if actual is None else actual
    expected = {'torch':'2.5.1+cu124','numpy':'2.1.3','torch_cuda':'12.4','cuda_available':True,
                'cublas_workspace_config':':4096:8'}
    if any(actual.get(k)!=v for k,v in expected.items()):
        raise RuntimeError('Formal environment mismatch/no CUDA/CUBLAS: '+str(actual))


def phase1_gate(path, expected_sha):
    if not path or not expected_sha:
        raise RuntimeError('Phase val requires --phase1-verdicts and --phase1-verdicts-sha256')
    if sha(path) != expected_sha:
        raise ValueError('Phase-1 verdict SHA mismatch')
    result = read_json(path)
    if result.get('status') != 'COMPLETE' or result.get('synthetic_demo') or result.get('phase') != 'cv' or result.get('protocol_sha256') != PROTOCOL_SHA:
        raise ValueError('Phase-1 verdicts must be complete, real, same-protocol CV results')
    if set(result.get('decisions',{})) != {'S1','S2','S3'}:
        raise ValueError('Missing phase-1 decisions')
    return result


def require_formal_slot(args, manifest_sha):
    """Canonical, ordered slots prevent extra/replacement completed formal jobs.

    Infrastructure failure recovery is deliberately manual: preserve/archive the
    failed run and its failure log before authorizing the same slot again.
    """
    parent=ROOT/'artifacts/s_ablation_mstca_v1'/args.phase
    slots=[(s,f,a) for s in (42,43,44) for f in (range(3) if args.phase=='cv' else [0])
           for a in ('S0','S1','S2','S3')]
    current=(args.seed,args.fold,args.arm)
    expected=parent/f'{args.arm}_seed{args.seed}_fold{args.fold}'
    if Path(args.output).resolve()!=expected.resolve():
        raise ValueError('Formal output must use the frozen canonical run slot')
    for seed,fold,arm in slots[:slots.index(current)]:
        path=parent/f'{arm}_seed{seed}_fold{fold}'/'report.json'
        r=read_json(path) if path.exists() else {}
        if r.get('status')!='COMPLETE' or r.get('protocol_sha256')!=PROTOCOL_SHA or r.get('implementation_manifest_sha256')!=manifest_sha:
            raise RuntimeError('Required preceding run is incomplete or has a different manifest: '+str(path))


def run(args, *, probe_epochs=None):
    # All refusal gates precede imports with setdefault side effects, data reads and output creation.
    if Path(args.output).exists():
        raise FileExistsError('Fresh output directory required')
    if args.phase == 'val':
        phase1_gate(getattr(args,'phase1_verdicts',None),getattr(args,'phase1_verdicts_sha256',None))
        if args.cpu_smoke:
            raise RuntimeError('CPU smoke is source-train CV only, never val')
    actual_env = environment()
    if not args.cpu_smoke:
        require_formal_environment(actual_env)
    elif not Path(args.output).resolve().is_relative_to(SMOKE_ROOT.resolve()):
        raise ValueError('CPU smoke output must remain under the task smoke artifact root')
    p,pins = load_protocol(args.protocol)
    if args.arm not in p['arms'] or args.seed not in p['seeds'] or args.fold not in (0,1,2):
        raise ValueError('Unregistered arm/seed/fold')
    if args.phase=='val' and args.fold!=0:
        raise ValueError('Phase val has no outer fold; use --fold 0 exactly once per arm/seed')
    manifest = implementation_manifest()
    manifest_sha = None
    if not args.cpu_smoke:
        path = getattr(args,'implementation_manifest',None)
        manifest_sha = getattr(args,'implementation_manifest_sha256',None)
        if not path or not manifest_sha or sha(path)!=manifest_sha or read_json(path)!=manifest:
            raise ValueError('Formal run needs the pre-run SHA-pinned implementation manifest')
        if probe_epochs is None:
            require_formal_slot(args,manifest_sha)
    identities = verify_data(args.data,smoke=args.cpu_smoke)
    frame = read_ledger(args.data)
    sizes = fold_sizes(frame,p)
    positions,_ = split_ledger(frame,args.fold,args.phase)
    full_counts = {k:len(v) for k,v in positions.items()}
    if args.cpu_smoke:
        positions = {k:take_smoke(frame,v,SMOKE_LIMITS[k]) for k,v in positions.items()}
    # No held-out feature access here. Even the selected held-out cache is created after all optimizer steps.
    fit = CacheWindows(args.data,frame,positions['fit'])
    inner = CacheWindows(args.data,frame,positions['inner'])
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.allow_tf32=True
    torch.backends.cuda.matmul.allow_tf32=False
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device('cpu' if args.cpu_smoke else 'cuda:0')
    if device.type=='cuda':
        torch.cuda.manual_seed_all(args.seed)
    variant = p['arms'][args.arm]['variant']
    before = torch.random.get_rng_state().clone()
    model = build_variant(variant,26,2,.2)
    receipt = initialize_variant(model,variant,args.seed,policy='module_seed_v1')
    if not torch.equal(before,torch.random.get_rng_state()):
        raise RuntimeError('Factory changed global CPU RNG')
    if receipt['full_state_digest'] != golden_digests()[(variant,args.seed)]:
        raise RuntimeError('Golden initialization mismatch: STOP, never regenerate goldens')
    # Lazy import: this historical module sets CUBLAS via setdefault. The formal
    # environment was already asserted above, so it cannot mask a missing value.
    from smoke_normal_source_sampling import sampling_weights, ORIGINAL, PREFAULT, FAULT
    selected = frame.iloc[positions['fit']]
    origins = np.where(selected.qualified_y==1,FAULT,np.where(selected.normal_origin=='normal_flight',ORIGINAL,PREFAULT))
    weights = sampling_weights(selected.qualified_y.to_numpy(),origins,True)
    sampler = WeightedRandomSampler(weights,len(fit),replacement=True,
        generator=torch.Generator().manual_seed(args.seed+11))
    trainloader = DataLoader(fit,batch_size=512,sampler=sampler,num_workers=0,
        generator=torch.Generator().manual_seed(args.seed+101))
    innerloader = DataLoader(inner,batch_size=512,shuffle=False,num_workers=0,
        generator=torch.Generator().manual_seed(args.seed+101))
    output = Path(args.output)
    output.mkdir(parents=True,exist_ok=False)
    write_json(output/'factory_receipt.json',receipt)
    write_json(output/'implementation_manifest.json',manifest)
    write_json(output/'population.json',{k:frame.iloc[v].original_cache_index.tolist() for k,v in positions.items()})
    report = {'status':'RUNNING','phase':args.phase,'arm':args.arm,'seed':args.seed,'fold':args.fold,
        'cpu_smoke':args.cpu_smoke,'determinism_probe':probe_epochs is not None,'synthetic_demo':False,
        'protocol_sha256':sha(args.protocol),'code_sha256':pins,'data_identities':identities,
        'implementation_manifest_sha256':manifest_sha or sha(output/'implementation_manifest.json'),
        'implementation_files':manifest['files'],'environment':actual_env,'device':str(device),
        'initial_digest':receipt['full_state_digest'],'param_count':receipt['param_count'],'macs':receipt['macs'],
        'fold_sizes':sizes,'full_population_counts':full_counts,'used_window_counts':{k:len(v) for k,v in positions.items()},
        'heldout_features_loaded_during_training':False,'epochs':0,'optimizer_steps':0,'history':[],
        'phase1_verdicts_sha256':getattr(args,'phase1_verdicts_sha256',None),
        'started_at':datetime.now(timezone.utc).isoformat()}
    write_json(output/'report.json',report)
    started = time.perf_counter()
    try:
        model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4)
        criterion = FocalLoss(gamma=2.,alpha=None)
        selection = {'best_loss':math.inf,'best_epoch':None,'reference':math.inf,'bad_epochs':0}
        epochs = 2 if args.cpu_smoke or probe_epochs is not None else p['training']['max_epochs']
        for epoch in range(1,epochs+1):
            lr = get_warmup_cosine_lr(epoch-1,5,100,3e-4,1e-6)
            for group in optimizer.param_groups:
                group['lr']=lr
            model.train()
            losses = []
            for x,y in trainloader:
                if not torch.isfinite(x).all() or not torch.isin(y,torch.tensor([0,1])).all():
                    raise ValueError('Invalid sample before forward/BN')
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(x.to(device)),y.to(device))
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite fit focal loss')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
                report['optimizer_steps'] += 1
            score = inner_ce(model,innerloader,device)
            selection,best,stop = selection_step(selection,score,epoch)
            if best:
                save_state(output/'best.pt',model.state_dict())
            save_state(output/'last.pt',model.state_dict())
            report['history'].append({'epoch':epoch,'lr':lr,'fit_focal_loss':float(np.mean(losses)),
                'inner_class_balanced_ce':score,'selection':dict(selection),'gradient_steps':len(losses)})
            report['epochs']=epoch
            write_json(output/'report.json',report,update=True)
            if stop:
                break
        # End of training: only now materialize the outer held-out X (or the gated val).
        held_frame = frame if args.phase=='cv' else read_ledger(args.data,'val')
        held_indices = positions['heldout'] if args.phase=='cv' else np.arange(len(held_frame))
        if args.phase=='val' and len(held_frame)!=34710:
            raise ValueError('Wrong val confirmation population')
        held = CacheWindows(args.data,held_frame,held_indices,split='train' if args.phase=='cv' else 'val')
        loader = DataLoader(held,batch_size=512,shuffle=False,num_workers=0,
            generator=torch.Generator().manual_seed(args.seed+101))
        prediction_hashes, state_digests = {},{}
        for checkpoint in ('best','last'):
            model.load_state_dict(torch.load(output/f'{checkpoint}.pt',map_location=device,weights_only=True))
            state_digests[checkpoint]=_state_digest(model.state_dict())
            model.eval()
            probabilities=[]
            with torch.inference_mode():
                for x,y in loader:
                    probabilities.append(model(x.to(device)).double().softmax(1).cpu().numpy())
            probs=np.concatenate(probabilities)
            if not np.isfinite(probs).all():
                raise FloatingPointError('Nonfinite held-out probabilities')
            path=output/f'heldout_predictions_{checkpoint}.npz'
            with path.open('xb') as stream:
                np.savez_compressed(stream,probabilities=probs,original_cache_index=held.ids.numpy())
            prediction_hashes[checkpoint]=sha(path)
        # Reject source/code drift during execution, without a second val-content access.
        _,after_pins=load_protocol(args.protocol)
        if after_pins!=pins or implementation_manifest()!=manifest:
            raise ValueError('Code changed during run')
        report.update(status='COMPLETE_CPU_SMOKE_NOT_FORMAL' if args.cpu_smoke else
                      ('COMPLETE_DETERMINISM_PROBE' if probe_epochs is not None else 'COMPLETE'),
            best_epoch=selection['best_epoch'],last_epoch=report['epochs'],
            wall_time_s=time.perf_counter()-started,prediction_sha256=prediction_hashes,
            checkpoint_state_digests=state_digests,heldout_prediction_passes={'best':1,'last':1},
            finished_at=datetime.now(timezone.utc).isoformat())
        write_json(output/'report.json',report,update=True)
        return report
    except Exception as exc:
        report.update(status='FAILED',error=repr(exc),wall_time_s=time.perf_counter()-started)
        write_json(output/'report.json',report,update=True)
        raise


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--protocol',type=Path,default=DEFAULT_PROTOCOL)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--arm',choices=['S0','S1','S2','S3'],required=True)
    p.add_argument('--seed',type=int,choices=[42,43,44],required=True)
    p.add_argument('--fold',type=int,choices=[0,1,2],required=True)
    p.add_argument('--phase',choices=['cv','val'],required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--cpu-smoke',action='store_true')
    p.add_argument('--implementation-manifest',type=Path)
    p.add_argument('--implementation-manifest-sha256')
    p.add_argument('--phase1-verdicts',type=Path)
    p.add_argument('--phase1-verdicts-sha256')
    return p


if __name__=='__main__':
    report=run(parser().parse_args())
    print(json.dumps({k:report[k] for k in ('status','arm','seed','fold','epochs','initial_digest')},indent=2))
