"""Read-only diagnostic after failed M3 gate; never grants training clearance."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import sys
import numpy as np
import pandas as pd
import torch
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'packages/uav_fault'),str(ROOT/'scripts/research')]
from uav_fault.data.causal_cache import sha
from uav_fault.models.cnn.lstm import OnlineKGMSTCA
from uav_fault.models.cnn.mstca_m3 import VARIANTS,build_m3_model
from train_source_normal_sampling import state_digest
from smoke_normal_source_sampling import sampling_weights


def digest(t):
    return hashlib.sha256(t.contiguous().numpy().tobytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for n in ['protocol','data','features','initial-anchor','output']:
        ap.add_argument('--'+n,type=Path,required=True)
    a=ap.parse_args()
    if a.output.exists():raise FileExistsError('Keep every environment diagnostic')
    p=json.loads(a.protocol.read_text(encoding='utf-8'))
    if sha(a.protocol)!=a.protocol.with_suffix('.json.sha256').read_text().strip():raise ValueError('Protocol SHA changed')
    for n,h in p['code_identities'].items():
        if sha(ROOT/n)!=h:raise ValueError('Frozen code changed: '+n)
    spec=p['initialization']['source']
    if sha(a.initial_anchor)!=spec['sha256']:raise ValueError('Wrong archived anchor')
    archived=torch.load(a.initial_anchor,map_location='cpu',weights_only=True)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        fresh=OnlineKGMSTCA(in_channels=26,hidden_size=64,num_layers=1,num_classes=2,dropout=.2,columns=p['data']['base_columns'])
    exact=set(fresh.state_dict())==set(archived) and all(torch.equal(v,archived[k]) for k,v in fresh.state_dict().items())
    common={'encoder.'+k[len('raw_encoder.'):]:v for k,v in archived.items() if k.startswith('raw_encoder.')}
    common.update({k:v for k,v in archived.items() if k.startswith('head.')})
    torch.set_num_threads(4)
    gen=torch.Generator().manual_seed(991)
    x=torch.randn(4,26,150,generator=gen);f=torch.randn(4,13,150,generator=gen)
    with torch.inference_mode():ref=build_m3_model('C0',common).eval()(x)
    rows={};arrays={}
    for variant in VARIANTS:
        model=build_m3_model(variant,common).eval()
        with torch.inference_mode():out=model(x) if variant=='C0' else model(x,f)
        row=dict(parameters=sum(v.numel() for v in model.parameters()),common_exact=all(torch.equal(v,model.state_dict()[k]) for k,v in common.items()),initial_eval_equal_C0=torch.equal(ref,out))
        if variant!='C0':
            row['adapter_digest']=state_digest(model.adapter.state_dict())
            row['adapter_matches_frozen_local']=row['adapter_digest']==p['initialization']['adapter_initial_state_digest_local']
            row['adapter_tensors']={k:dict(sha256=digest(v),shape=list(v.shape),dtype=str(v.dtype),min=float(v.min()),max=float(v.max())) for k,v in model.adapter.state_dict().items()}
            if variant=='Ccap':arrays={k:v.numpy() for k,v in model.adapter.state_dict().items()}
        rows[variant]=row
    expected=p['data']['source_split_identities']['train'];export_path=a.features/'cache/train.pt'
    if sha(export_path)!=expected['residual_cache_sha256']:raise ValueError('FrozenTrain cache changed')
    export=torch.load(export_path,map_location='cpu',weights_only=True,mmap=True)
    m0=json.loads((ROOT/'configs/experiments/causal_pure_mstca_m0_v1.json').read_text(encoding='utf-8'))
    ledger=a.data/'ledgers/train_included_windows.csv'
    if sha(ledger)!=m0['data_identities']['ledgers/train_included_windows.csv']:raise ValueError('Wrong Train ledger')
    metadata=pd.read_csv(ledger,low_memory=False,keep_default_na=False).iloc[export['base_compact_index'].numpy()].reset_index(drop=True)
    if digest(export['original_cache_index'])!=expected['original_ids_sha256'] or digest(export['y'])!=expected['labels_sha256']:raise ValueError('Wrong IDs/labels')
    if not np.array_equal(metadata.original_cache_index,export['original_cache_index']) or not np.array_equal(metadata.qualified_y,export['y']):raise ValueError('Wrong linked metadata')
    origins=np.where(metadata.qualified_y==1,'fault_endpoint',np.where(metadata.case_level2_label=='normal','original_normal_case','normal_part_of_fault_case'))
    weights=sampling_weights(export['y'].numpy(),origins,True)
    if digest(weights)!=p['sampling']['weights_sha256']:raise ValueError('Wrong weights')
    local_path=ROOT/p['sampling']['local_sampler_check_path']
    if sha(local_path)!=p['sampling']['local_sampler_check_sha256']:raise ValueError('Wrong sampler reference')
    local=json.loads(local_path.read_text(encoding='utf-8'));samplers={}
    for variant in VARIANTS:
        sampler=torch.utils.data.WeightedRandomSampler(weights,151833,replacement=True,generator=torch.Generator().manual_seed(53))
        hashes=[digest(export['original_cache_index'][torch.tensor(list(sampler),dtype=torch.int64)]) for epoch in range(100)]
        samplers[variant]=dict(epochs=100,all_epoch_IDs_match_local=hashes==local['reference_epoch_originalIDs_SHA256'],mismatched_epochs=[i+1 for i,(actual,ref) in enumerate(zip(hashes,local['reference_epoch_originalIDs_SHA256'])) if actual!=ref])
    result=dict(status='DIAGNOSTIC_ONLY_M3_REMAINS_BLOCKED_BY_ADAPTER_INITIALIZATION',checked_at=datetime.now(timezone.utc).isoformat(),environment=dict(torch=str(torch.__version__),numpy=np.__version__,torch_cuda=torch.version.cuda),protocol_sha256=sha(a.protocol),fresh_legacy_seed42_matches_archived=exact,shared_state_digest=state_digest(common),models=rows,samplers=samplers,sampler_weights_sha256=digest(weights),no_training_clearance=True,optimizer_steps=0,formal_jobs_started=0,Test_Real_used=False,device='cpu')
    a.output.mkdir(parents=True,exist_ok=False)
    np.savez(a.output/'adapter_untrained_diagnostic.npz',**arrays)
    (a.output/'diagnostic.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(dict(status=result['status'],fresh_M0_exact=exact,sampler_all8_match=all(v['all_epoch_IDs_match_local'] for v in samplers.values()))),flush=True)


if __name__=='__main__':main()
