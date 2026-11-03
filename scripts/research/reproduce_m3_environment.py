"""Fail-closed M3 formal-environment reproduction, CPU only, no training.

Checks the frozen shared untrained initialization, all adapters and 8x100
sampler-ID digests against existing local evidence. Writes a receipt on failure.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import json
import sys
import traceback
import numpy as np
import pandas as pd
import torch
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'packages/uav_fault'), str(ROOT/'scripts/research')]
from uav_fault.data.causal_cache import sha
from uav_fault.models.cnn.lstm import OnlineKGMSTCA
from uav_fault.models.cnn.mstca_m3 import VARIANTS, build_m3_model
from train_source_normal_sampling import state_digest
from smoke_normal_source_sampling import sampling_weights


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(tensor):
    import hashlib
    return hashlib.sha256(tensor.contiguous().numpy().tobytes()).hexdigest()


def reproduce(args):
    p = read(args.protocol)
    if sha(args.protocol) != args.protocol.with_suffix('.json.sha256').read_text().strip():
        raise ValueError('Frozen M3 protocol identity mismatch')
    actual = dict(torch=str(torch.__version__), numpy=np.__version__, torch_cuda=torch.version.cuda)
    if actual != p['formal_environment']:
        raise ValueError('Formal environment mismatch: '+repr(actual))
    for name, expected in p['code_identities'].items():
        if sha(ROOT/name) != expected:
            raise ValueError('Frozen code identity mismatch: '+name)
    spec = p['initialization']['source']
    if sha(args.initial_anchor) != spec['sha256']:
        raise ValueError('Archived untrained M0 anchor file differs')
    archived = torch.load(args.initial_anchor, map_location='cpu', weights_only=True)
    if state_digest(archived) != spec['state_digest']:
        raise ValueError('Archived untrained state digest differs')
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        fresh = OnlineKGMSTCA(in_channels=26, hidden_size=64, num_layers=1,
                            num_classes=2, dropout=.2, columns=p['data']['base_columns'])
    if set(fresh.state_dict()) != set(archived) or any(not torch.equal(v, archived[k]) for k, v in fresh.state_dict().items()):
        raise ValueError('Exact fresh seed42 legacy initialization reproduction FAILED')
    common = {'encoder.'+k[len('raw_encoder.'):]: v for k, v in archived.items() if k.startswith('raw_encoder.')}
    common.update({k:v for k,v in archived.items() if k.startswith('head.')})
    if state_digest(common) != p['initialization']['shared_state_digest']:
        raise ValueError('Pure26 mapped initialization digest mismatch')
    torch.set_num_threads(4)
    generator = torch.Generator().manual_seed(991)
    x = torch.randn(4,26,150,generator=generator)
    features = torch.randn(4,13,150,generator=generator)
    reference = build_m3_model('C0', common).eval()
    with torch.inference_mode():
        expected_logits = reference(x)
    models = {}
    for variant in VARIANTS:
        before = torch.random.get_rng_state().clone()
        model = build_m3_model(variant, common).eval()
        if not torch.equal(before, torch.random.get_rng_state()):
            raise ValueError('Model construction consumed sampling/global RNG')
        if any(not torch.equal(v, model.state_dict()[k]) for k,v in common.items()):
            raise ValueError('Shared weights/buffers changed: '+variant)
        adapter_digest = None
        if variant != 'C0':
            adapter_digest = state_digest(model.adapter.state_dict())
            if adapter_digest != p['initialization']['adapter_initial_state_digest_local']:
                raise ValueError('Adapter initialization differs from frozen local digest: '+variant+' '+adapter_digest)
        with torch.inference_mode():
            logits = model(x) if variant == 'C0' else model(x, features)
        if not torch.equal(logits, expected_logits):
            raise ValueError('All8 exact initial output equality FAILED: '+variant)
        count = sum(v.numel() for v in model.parameters())
        if count != (105938 if variant == 'C0' else 109538):
            raise ValueError('Parameter count mismatch')
        models[variant] = dict(parameters=count, common_state_digest=state_digest(common),
                               adapter_state_digest=adapter_digest, initial_eval_equal_C0=True)
    expected = p['data']['source_split_identities']['train']
    path = args.features/'cache/train.pt'
    if sha(path) != expected['residual_cache_sha256']:
        raise ValueError('Source Train feature export hash differs')
    export = torch.load(path,map_location='cpu',weights_only=True,mmap=True)
    ledger = args.data/'ledgers/train_included_windows.csv'
    m0 = read(ROOT/'configs/experiments/causal_pure_mstca_m0_v1.json')
    if sha(ledger) != m0['data_identities']['ledgers/train_included_windows.csv']:
        raise ValueError('Source Train ledger identity mismatch')
    frame = pd.read_csv(ledger,low_memory=False,keep_default_na=False).iloc[export['base_compact_index'].numpy()].reset_index(drop=True)
    if len(frame) != expected['windows'] or digest(export['original_cache_index']) != expected['original_ids_sha256'] or digest(export['y']) != expected['labels_sha256']:
        raise ValueError('Train population or original labels/IDs differ')
    if not np.array_equal(frame.original_cache_index,export['original_cache_index']) or not np.array_equal(frame.qualified_y,export['y']):
        raise ValueError('Train metadata alignment failed')
    origins=np.where(frame.qualified_y==1,'fault_endpoint',np.where(frame.case_level2_label=='normal','original_normal_case','normal_part_of_fault_case'))
    weights=sampling_weights(export['y'].numpy(),origins,True)
    if digest(weights)!=p['sampling']['weights_sha256']:
        raise ValueError('Sampler weights differ')
    local_path=ROOT/p['sampling']['local_sampler_check_path']
    if sha(local_path)!=p['sampling']['local_sampler_check_sha256']:
        raise ValueError('Local sampler reference receipt hash differs')
    local=read(local_path);samplers={}
    for variant in VARIANTS:
        sampler=torch.utils.data.WeightedRandomSampler(weights,151833,replacement=True,
            generator=torch.Generator().manual_seed(53))
        hashes=[]
        for epoch in range(100):
            positions=torch.tensor(list(sampler),dtype=torch.int64)
            actual_digest=digest(export['original_cache_index'][positions]);hashes.append(actual_digest)
            if actual_digest!=local['reference_epoch_originalIDs_SHA256'][epoch]:
                raise ValueError('Sampler differs from local frozen IDs: '+variant+' epoch '+str(epoch+1))
        samplers[variant]=dict(epochs=100,all_epoch_original_ID_digests_match_local=True)
    return dict(status='PASS_M3_FORMAL_ENVIRONMENT_REPRODUCTION',environment=actual,
        protocol_sha256=sha(args.protocol),legacy_anchor_sha256=sha(args.initial_anchor),
        fresh_legacy_seed42_exact=True,common_state_digest=state_digest(common),models=models,
        sampler_weights_sha256=digest(weights),samplers=samplers,
        reference_epoch_originalIDs_SHA256=local['reference_epoch_originalIDs_SHA256'],
        optimizer_steps=0,formal_jobs_started=0,device_used_for_reproduction='cpu',Test_Real_used=False)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ['protocol','data','features','initial-anchor','output']:
        ap.add_argument('--'+name,type=Path,required=True)
    args=ap.parse_args()
    if args.output.exists():
        raise FileExistsError('Never overwrite an environment reproduction attempt')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    try:
        result=reproduce(args)
    except Exception as exc:
        result=dict(status='FAILED_M3_ENVIRONMENT_REPRODUCTION_NO_TRAINING',error=repr(exc),
                    traceback=traceback.format_exc(),optimizer_steps=0,formal_jobs_started=0)
        args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        raise
    result['checked_at']=datetime.now(timezone.utc).isoformat()
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(dict(status=result['status'],models=len(result['models']),samplers=len(result['samplers'])),ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
