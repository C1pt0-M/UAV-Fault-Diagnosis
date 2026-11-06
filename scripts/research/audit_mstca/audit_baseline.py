"""Read-only MSTCA audit; CPU only, 64 train + 64 val feature windows.

Run from any directory: python -I -B <absolute path to this script>
No optimizer, no production Dataset full scan, no test/Real file access.
All fixtures and results are new files under the fixed audit artifact directory.
"""
from __future__ import annotations

import ast
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
os.environ['CUDA_VISIBLE_DEVICES'] = ''
ROOT = Path(__file__).resolve().parents[3]
DATA = Path('D:/UAV-Data/causal_input_v1')
OUT = ROOT / 'artifacts/mstca_baseline_audit_v1_20261106/checks_v2'
ALLOWED_DATA = {str((DATA / p).resolve()).lower() for p in (
    'norm_stats.json', 'input_contract.json', 'cache/train.pt', 'cache/val.pt',
    'ledgers/train_included_windows.csv', 'ledgers/val_included_windows.csv')}
opened_data = set()


def guard(event, args):
    if event != 'open' or not isinstance(args[0], (str, bytes, os.PathLike)):
        return
    path = Path(os.fsdecode(args[0])).resolve()
    key = str(path).lower()
    if key.startswith(str(DATA.parent.resolve()).lower() + os.sep):
        if key not in ALLOWED_DATA:
            raise PermissionError('Audit forbids this data read: ' + str(path))
        mode, flags = args[1], args[2]
        if (isinstance(mode, str) and any(c in mode for c in 'wax+')) or flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC):
            raise PermissionError('All source data are read-only')
        opened_data.add(str(path))


sys.addaudithook(guard)
sys.path[:0] = [str(ROOT / 'packages/uav_fault'), str(ROOT / 'scripts/research')]
import train_causal_mstca_m0 as m0
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from uav_fault.causal_input import canonical_window
from uav_fault.data.causal_cache import CausalCacheDataset, VERSION, sha
from uav_fault.model_contract import RFLY_SEQ_COLUMNS_V2
from uav_fault.models.cnn.focal_loss import FocalLoss
from uav_fault.models.cnn.lstm import OnlineMSTCA, OnlineKGMSTCA
from uav_fault.online.fault_detector import FaultDetector
from smoke_normal_source_sampling import ORIGINAL, PREFAULT, FAULT, sampling_weights, probability_table
from train_source_normal_sampling import state_digest, selection_step


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n')


def norm_diff(a, b):
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b))))


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    p = m0.load_protocol(ROOT / 'configs/experiments/causal_pure_mstca_m0_v1.json')
    anchor = ROOT / p['initialization_anchor']['source_path_for_preparation']
    model, receipt = m0.build_model(p, 42, anchor)
    model.eval()
    r = {'scope': {'device': 'cpu', 'optimizer_steps': 0, 'feature_windows_per_split': 64,
                   'full_source_label_metadata_only': True, 'production_dataset_full_scan': False,
                   'trained_checkpoint_used': False},
         'environment': {'python': sys.version, 'torch': str(torch.__version__), 'numpy': np.__version__,
                         'pandas': pd.__version__, 'threads': torch.get_num_threads()}, 'initialization': receipt}
    torch.manual_seed(42)
    direct = OnlineMSTCA(in_channels=26).eval()
    torch.manual_seed(42)
    direct2 = OnlineMSTCA(in_channels=26).eval()
    gen = torch.Generator().manual_seed(20261106)
    x = torch.randn(32, 26, 150, generator=gen)
    mapped2, _ = m0.build_model(p, 42, anchor)
    mapped2.eval()
    r['initialization'].update(direct_digest=state_digest(direct.state_dict()),
        direct_repeated_digest_equal=state_digest(direct.state_dict()) == state_digest(direct2.state_dict()),
        direct_repeated_forward_bitwise=torch.equal(direct(x), direct2(x)),
        mapped_repeated_forward_bitwise=torch.equal(model(x), mapped2(x)),
        direct_changed_keys=[k for k,v in model.state_dict().items() if not torch.equal(v, direct.state_dict()[k])])
    assert r['initialization']['direct_repeated_forward_bitwise']
    assert r['initialization']['mapped_repeated_forward_bitwise']
    archived = torch.load(anchor, map_location='cpu', weights_only=True)
    r['initialization']['discarded_keys'] = [k for k in archived if not k.startswith(('raw_encoder.', 'head.'))]
    r['initialization']['mapped_keys'] = list(model.state_dict())
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        fresh_legacy = OnlineKGMSTCA(in_channels=26)
    fresh_state = fresh_legacy.state_dict()
    fresh_mapped = {k: fresh_state['raw_encoder.'+k[len('encoder.'):] if k.startswith('encoder.') else k]
                    for k in model.state_dict()}
    r['initialization']['local_fresh_legacy_digest'] = state_digest(fresh_state)
    r['initialization']['local_fresh_mapped_digest'] = state_digest(fresh_mapped)
    r['initialization']['local_direct_vs_local_mapped_changed_keys'] = [k for k,v in direct.state_dict().items()
        if not torch.equal(v,fresh_mapped[k])]

    shapes, macs, handles = {}, {}, []
    def hook(name):
        def run(module, inputs, output):
            shapes[name] = list(output.shape)
            if isinstance(module, nn.Conv1d):
                macs[name] = int(output.numel() * module.in_channels // module.groups * module.kernel_size[0])
            elif isinstance(module, nn.Linear):
                macs[name] = int(output.numel() * module.in_features)
        return run
    for name, module in model.named_modules():
        if name:
            handles.append(module.register_forward_hook(hook(name)))
    with torch.no_grad(): model(x[:1])
    for h in handles: h.remove()
    r['structure'] = {'parameters': sum(v.numel() for v in model.parameters()),
                      'state_tensors': len(model.state_dict()), 'shapes': shapes, 'macs_by_layer': macs,
                      'conv_linear_macs': sum(macs.values())}
    assert r['structure']['parameters'] == 105938
    assert r['structure']['conv_linear_macs'] == 6764928

    # Equal perturbation templates at each end, no physical-fault interpretation.
    r['tail_probe'] = []
    base = model(x).detach()
    for k in (1, 3, 5, 10):
        delta = .1 * torch.randn(32, 26, k, generator=gen)
        entry = {'k': k}
        for name, sl in [('start', slice(0,k)), ('end', slice(150-k,150))]:
            altered = x.clone(); altered[:, :, sl] += delta
            with torch.no_grad(): logits = model(altered)
            dl = torch.linalg.vector_norm(logits-base, dim=1)
            dp = (logits.softmax(1)[:,1]-base.softmax(1)[:,1]).abs()
            entry[name] = {'logit_l2_mean': float(dl.mean()), 'logit_l2_median': float(dl.median()),
                           'fault_probability_abs_mean': float(dp.mean())}
        entry['end_start_mean_ratio'] = entry['end']['logit_l2_mean'] / entry['start']['logit_l2_mean']
        r['tail_probe'].append(entry)
    grad_x = x.clone().requires_grad_(True)
    (model(grad_x)[:,1]-model(grad_x)[:,0]).sum().backward()
    grad = grad_x.grad.abs().sum(1).mean(0)
    r['gradient'] = {'per_frame_abs_channel_sum_mean': grad.tolist(),
                     'first': float(grad[0]), 'middle': float(grad[75]), 'last': float(grad[-1])}
    # MaxPool's direct last input is discarded, but previous convolutions and SE retain paths.
    t = torch.arange(75., requires_grad=True).reshape(1,1,75); t.retain_grad()
    F.max_pool1d(t,2).sum().backward()
    r['pool2_probe'] = {'output_length': 37, 'last_direct_input_gradient': float(t.grad[0,0,-1])}

    # Same rectangular step/pulse: +1 z over 30 frames, equal amplitude/duration.
    r['step_probe'] = []
    zero_logits = model(torch.zeros(1,26,150)).detach()
    for name, start in [('start',0), ('middle',60), ('end',120)]:
        pulse = torch.zeros(1,26,150); pulse[:,:,start:start+30] = 1
        with torch.no_grad(): logits = model(pulse)
        r['step_probe'].append({'position': name, 'span': [start,start+30], 'logits': logits[0].tolist(),
            'fault_probability': float(logits.softmax(1)[0,1]),
            'logit_l2_from_zero': float(torch.linalg.vector_norm(logits-zero_logits))})
    r['batchnorm'] = {'dropout_modules': [n for n,v in model.named_modules() if isinstance(v,nn.Dropout)],
        'eval_b1_finite': bool(torch.isfinite(model(x[:1])).all()),
        'eval_b1_vs_b32_max_abs': float((model(x[:1])-model(x)[:1]).abs().max().detach())}
    training_probe, _ = m0.build_model(p, 42, anchor)
    training_probe.train(); loss = FocalLoss()(training_probe(x[:1]), torch.tensor([1])); loss.backward()
    r['batchnorm']['train_b1_forward_backward_finite'] = bool(torch.isfinite(loss) and all(
        v.grad is None or torch.isfinite(v.grad).all() for v in training_probe.parameters()))
    logits = torch.randn(19,2,generator=gen,requires_grad=True); targets=torch.arange(19)%2
    focal0=FocalLoss(gamma=0)(logits,targets); ce=F.cross_entropy(logits,targets)
    focal2=FocalLoss(gamma=2)(logits,targets)
    manual=(-(1-logits.softmax(1)[torch.arange(19),targets])**2 * logits.log_softmax(1)[torch.arange(19),targets]).mean()
    r['loss']={'gamma0_ce_abs_diff':float((focal0-ce).abs().detach()),
               'gamma2_formula_abs_diff':float((focal2-manual).abs().detach()),
               'gamma0_gradient_max_abs_diff':float((torch.autograd.grad(focal0,logits,retain_graph=True)[0]-torch.autograd.grad(ce,logits)[0]).abs().max())}
    assert r['loss']['gamma0_ce_abs_diff'] < 1e-6 and r['loss']['gamma2_formula_abs_diff'] < 1e-6

    stats=json.loads((DATA/'norm_stats.json').read_text(encoding='utf-8'))
    contract=json.loads((DATA/'input_contract.json').read_text(encoding='utf-8'))
    assert stats['columns']==contract['columns']==p['data']['columns']==RFLY_SEQ_COLUMNS_V2
    for name in ('norm_stats.json','input_contract.json'):
        assert sha(DATA/name)==p['data_identities'][name]
    frames={}; samples={}; sample_datasets={}
    # Read only 64 feature tensors/split using mmap. Full label/ledger arrays are needed for sampling masses.
    for split in ('train','val'):
        payload=torch.load(DATA/'cache'/f'{split}.pt',weights_only=True,map_location='cpu',mmap=True)
        assert payload['split']==split and payload['data_version']==VERSION and payload['normalization_applied'] is False
        frame=pd.read_csv(DATA/'ledgers'/f'{split}_included_windows.csv',keep_default_na=False)
        assert len(frame)==len(payload['y'])
        assert frame.domain.isin(['SIL','HIL']).all() and frame.split.eq(split).all()
        assert np.array_equal(frame.qualified_y,payload['y'].numpy())
        assert np.array_equal(frame.original_cache_index,payload['original_cache_index'].numpy())
        assert all(frame[k].all() for k in ('input_valid','label_valid','supervised_valid'))
        indices=np.linspace(0,len(frame)-1,64).round().astype(np.int64)
        raw=payload['x'][indices].clone(); y=payload['y'][indices].clone()
        samples[split]=raw
        # A faithful compact 64-window train/val fixture exercises the actual reader constructor + __getitem__.
        fixture=OUT/'reader_fixture'/split; (fixture/'cache').mkdir(parents=True)
        write(fixture/'build_manifest.json',{'status':'COMPLETE_FULL_FROZEN_SCOPE_CAUSAL_PUBLICATION_REPLAY_DATA_WITH_EXCLUSIONS','version':VERSION})
        write(fixture/'input_contract.json',contract)
        fixture_stats=dict(stats,input_contract_sha256=sha(fixture/'input_contract.json'))
        write(fixture/'norm_stats.json',fixture_stats)
        subset={k:payload[k] for k in ('data_version','split','normalization_applied')}
        subset.update(x=raw,y=y,original_cache_index=payload['original_cache_index'][indices].clone(),
            input_contract_sha256=sha(fixture/'input_contract.json'),norm_sha256=sha(fixture/'norm_stats.json'))
        torch.save(subset,fixture/'cache'/'subset.pt')
        write(fixture/f'{split}_cache_complete.json',{'cache_file':'subset.pt','cache_sha256':sha(fixture/'cache'/'subset.pt'),'original_windows':64})
        ds=CausalCacheDataset(fixture,expected_version=VERSION,split=split,purpose='source_development')
        batch_x,batch_y=next(iter(DataLoader(ds,batch_size=32,shuffle=False,num_workers=0)))
        assert torch.isfinite(batch_x).all() and torch.isin(batch_y,torch.tensor([0,1])).all()
        r.setdefault('source_batch',{})[split]={'feature_indices':indices.tolist(),'batch_shape':list(batch_x.shape),
            'batch_labels':batch_y.tolist(),'all_metadata_supervised_valid':True,
            'raw_baro_first_abs_max':float(raw[:,14,0].abs().max()), 'population':len(frame)}
        sample_datasets[split]=ds; frames[split]=frame
    assert not set(frames['train'].case_uid)&set(frames['val'].case_uid)
    assert not set(frames['train'].association_group)&set(frames['val'].association_group)

    # Real FaultDetector constructor, but explicit untrained fixture, never deployed assets.
    runtime=OUT/'runtime_fixture'
    cfg={'model':'mstca','in_channels':26,'hidden_size':64,'num_layers':1,'num_classes':2,'dropout':.2,
         'window_size':150,'feature_schema':'rfly_seq_columns_v2','protocol_name':VERSION,'task':'online_binary'}
    write(runtime/'config.json',cfg); write(runtime/'norm_stats.json',stats)
    torch.save(model.state_dict(),runtime/'best.pt')
    detector=FaultDetector(str(runtime),device='cpu')
    train_x=sample_datasets['val'][0][0]
    records=[dict(zip(stats['columns'],row.tolist())) for row in samples['val'][0].numpy().T]
    online_x=detector._normalize_feature_matrix(detector._build_feature_matrix(records)).T.copy()
    with torch.no_grad(): a=model(train_x[None]).numpy()
    b=np.asarray(detector.predict(records)['raw_logits'])
    r['parity']={'columns_exact':True,'val_tensor_max_abs_diff':norm_diff(train_x.numpy(),online_x),
        'val_logit_max_abs_diff':norm_diff(a[0],b),'detector_model_eval':not detector.model.training,
        'fixture_untrained':True,'raw_publication_reconstruction_not_attempted':True}
    assert r['parity']['val_tensor_max_abs_diff']==0 and r['parity']['val_logit_max_abs_diff']==0

    # Known differences: absolute-height float32 cancellation, batch whole-record origin, missing channels.
    synthetic=np.tile(np.asarray(stats['mean'],np.float64),(170,1))
    synthetic[:,14]=1000+np.arange(170)*.02
    synthetic[:,25]=np.linalg.norm(synthetic[:,11:14],axis=1)
    rows=[dict(zip(stats['columns'],row.tolist())) for row in synthetic]
    expected=canonical_window(synthetic[:150],stats['columns'],np.ones(150,dtype=bool),mean=stats['mean'],std=stats['std'])
    actual=detector._normalize_feature_matrix(detector._build_feature_matrix(rows[:150])).T
    captured=[]; predict_batch=detector._predict_batch_tensor
    def capture(batch):
        captured.extend(v.copy() for v in batch)
        return predict_batch(batch)
    detector._predict_batch_tensor=capture
    batch_preds=detector.predict_records_batch(rows,every_n=10)
    batch_inputs=list(captured);detector._predict_batch_tensor=predict_batch
    comparisons=[]
    for i,(end,pred) in enumerate(batch_preds):
        single_rows=rows[end-150:end]
        one_x=detector._normalize_feature_matrix(detector._build_feature_matrix(single_rows)).T
        single=detector.predict(single_rows)
        comparisons.append({'end':end,'input_max_abs_diff':norm_diff(batch_inputs[i],one_x),
            'baro_z_offset_first':float(batch_inputs[i][14,0]-one_x[14,0]),
            'logit_max_abs_diff':norm_diff(pred['raw_logits'],single['raw_logits'])})
    r['online_edge_cases']={'canonical_vs_detector_absolute_baro_max_abs_diff':norm_diff(expected,actual),
        'canonical_vs_detector_baro_max_abs_diff':norm_diff(expected[14],actual[14]),
        'batch_vs_window':comparisons,
        'missing_required_channel_silently_filled':detector._get_value({},'gyro_x')}
    assert comparisons[0]['input_max_abs_diff']==0 and comparisons[-1]['input_max_abs_diff']>0

    train=frames['train']; labels=train.qualified_y.to_numpy(dtype=np.int64)
    origins=np.where(labels==1,FAULT,np.where(train.case_level2_label.eq('normal'),ORIGINAL,PREFAULT))
    r['sampling']={}
    for arm in p['arms']:
        w=sampling_weights(labels,origins,arm=='balanced_source_normal')
        table=probability_table(w,labels,origins)
        draw=np.asarray(list(WeightedRandomSampler(w,100000,replacement=True,generator=torch.Generator().manual_seed(53))))
        counts=Counter(origins[draw]);normal=counts[ORIGINAL]+counts[PREFAULT]
        r['sampling'][arm]={'theory':table,'draw_counts':dict(counts),
            'empirical_original_fraction_within_normal':counts[ORIGINAL]/normal,
            'empirical_fault_fraction':counts[FAULT]/100000}
    rule=p['validation']['early_stop']
    state={'best_score':-float('inf'),'best_epoch':None,'significant_reference':-float('inf'),'bad_epochs':0}
    trajectory=[]
    for epoch,score in ((1,.8),(30,.8),(31,.801),(32,.803)):
        state,best,stop=selection_step(state,score,epoch,rule)
        trajectory.append({'epoch':epoch,'score':score,'state':state,'save_best':best,'stop':stop})
    r['selection_probe']=trajectory
    tied=torch.tensor([[0.,0.],[-1.,1.],[1.,-1.]])
    r['prediction']={'argmax':tied.argmax(1).tolist(),'softmax_argmax':tied.softmax(1).argmax(1).tolist(),
                     'strict_fault_prob_gt_half':(tied.softmax(1)[:,1]>.5).long().tolist()}

    # Source provenance with AST line ranges; no execution of any historical trainer/replay.
    evidence={}
    paths=list(p['code_identities'])+['packages/uav_fault/uav_fault/online/fault_detector.py',
        'scripts/data/build_full_causal_input.py','scripts/research/train_m0_groupcv.py',
        'scripts/research/train_causal_m3.py','scripts/research/replay_causal_m3.py',
        'scripts/research/replay_ens_events_val.py','tests/test_causal_mstca_m0.py',
        'tests/test_causal_cache.py','tests/test_fault_detector.py']
    for rel in dict.fromkeys(paths):
        path=ROOT/rel; source=path.read_text(encoding='utf-8-sig')
        definitions=[]
        for node in ast.walk(ast.parse(source)):
            if isinstance(node,(ast.ClassDef,ast.FunctionDef,ast.AsyncFunctionDef)):
                definitions.append({'symbol':node.name,'start':node.lineno,'end':node.end_lineno})
        evidence[rel]={'sha256':sha(path),'definitions':definitions,
                       'matches_frozen_pin':sha(path)==p['code_identities'][rel] if rel in p['code_identities'] else None}
    write(OUT/'source_evidence.json',evidence)
    r['scope']['opened_external_data_files']=sorted(opened_data)
    write(OUT/'results.json',r)
    print(json.dumps(r,ensure_ascii=False,indent=2,allow_nan=False))


if __name__=='__main__':
    main()
