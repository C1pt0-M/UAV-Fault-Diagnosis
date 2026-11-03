"""CPU-only M3 parameter, initialization and gradient connectivity checks.

No optimizer object/step, training loop, GPU or training-server connection.
The one nonzero projection probe is an isolated copy for derivative tests.
"""
from pathlib import Path
from datetime import datetime,timezone
import argparse,copy,json,sys,time
import numpy as np
import torch
from torch import nn
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'packages/uav_fault'),str(ROOT/'scripts/research')]
from uav_fault.models.cnn.mstca_m3 import VARIANTS,build_m3_model
from uav_fault.models.cnn.lstm import OnlineMSTCA
from uav_fault.models.cnn.focal_loss import FocalLoss
from uav_fault.data.m3_features import M3CandidateDataset
from uav_fault.data.causal_cache import sha
from train_source_normal_sampling import state_digest


def compute_macs(model,x,features):
    """Dense Conv/Linear multiply-accumulate count, runtime shape hooks."""
    total=[0];handles=[]
    def hook(module,args,out):
        if isinstance(module,nn.Conv1d):total[0]+=out.numel()*(module.in_channels//module.groups)*module.kernel_size[0]
        elif isinstance(module,nn.Linear):total[0]+=out.numel()*module.in_features
    for m in model.modules():
        if isinstance(m,(nn.Conv1d,nn.Linear)):handles.append(m.register_forward_hook(hook))
    with torch.inference_mode():model(x) if isinstance(model,OnlineMSTCA) else model(x,features)
    for h in handles:h.remove()
    return total[0]


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ['data','features','normalization','output']:ap.add_argument('--'+name,type=Path,required=True)
    a=ap.parse_args()
    if a.output.exists():raise FileExistsError('Preserve model preflight')
    p=json.loads((ROOT/'configs/experiments/causal_pure_mstca_m0_v1.json').read_text(encoding='utf-8'));anchor=ROOT/'artifacts/causal_pure_mstca_m0_v1_20261101/results/runs/uniform_replacement/initial.pt'
    original=ROOT/p['initialization_anchor']['source_path_for_preparation'];raw=torch.load(original,map_location='cpu',weights_only=True)
    if sha(original)!=p['initialization_anchor']['sha256'] or state_digest(raw)!=p['initialization_anchor']['state_digest']:raise ValueError('Original untrained anchor changed')
    common=torch.load(anchor,map_location='cpu',weights_only=True)
    if state_digest(common)!=p['initialization_anchor']['m0_mapped_state_digest']:raise ValueError('Pure M0 untrained identity mismatch')
    for key,v in common.items():
        source='raw_encoder.'+key[len('encoder.'):] if key.startswith('encoder.') else key
        if not torch.equal(v,raw[source]):raise ValueError('Untrained public state mapping changed')
    normalization_receipt=json.loads((a.normalization.parent/'data_preflight.json').read_text(encoding='utf-8'))
    torch.set_num_threads(4);ds=M3CandidateDataset(a.data,a.features,a.normalization,split='train',normalization_sha256=normalization_receipt['normalization_sha256'])
    n1_extreme=int(ds.source.residual[:,0].abs().amax(dim=1).argmax())
    acc_index=ds.source.base._x.shape[1]-1  # Frozen Base26 contract ends with acc_norm.
    acc_normalized=(ds.source.base._x[ds.base_indices,acc_index]-ds.source.base._mean[acc_index])/ds.source.base._std[acc_index]
    o6_extreme=int(acc_normalized.abs().amax(dim=1).argmax())
    indices=np.unique(np.r_[np.linspace(0,len(ds)-1,8).round().astype(int),n1_extreme,o6_extreme])
    batch=[ds[int(i)] for i in indices];x=torch.stack([t[0] for t in batch]);features=torch.stack([t[1] for t in batch]);y=torch.tensor([int(t[2]) for t in batch])
    criterion=FocalLoss(gamma=2.0)
    base=build_m3_model('C0',common).eval()
    with torch.inference_mode():reference=base(x)
    rows=[];adapter_reference=None;full_shared_digest=state_digest(common)
    for variant,names in VARIANTS.items():
        torch.manual_seed(719);before=torch.random.get_rng_state().clone();model=build_m3_model(variant,common)
        if not torch.equal(before,torch.random.get_rng_state()):raise ValueError('Constructor changed sampler/global RNG')
        for key,value in common.items():
            if not torch.equal(model.state_dict()[key],value):raise ValueError('Shared encoder/head initialization differs')
        shared_digest=state_digest({k:v for k,v in model.state_dict().items() if k.startswith(('encoder.','head.'))})
        if shared_digest!=full_shared_digest:raise ValueError('Common buffer/state differs')
        model.eval()
        with torch.inference_mode():out=model(x) if variant=='C0' else model(x,features)
        if not torch.equal(out,reference):raise ValueError('Initial evaluation functions not identical')
        params=sum(p.numel() for p in model.parameters());adapter_params=0 if variant=='C0' else sum(p.numel() for p in model.adapter.parameters())
        if params!=(105938 if variant=='C0' else 109538) or adapter_params!=(0 if variant=='C0' else 3600):raise ValueError('Actual architecture parameter count mismatch')
        row={'variant':variant,'features':list(names),'parameters':params,'adapter_parameters':adapter_params,'shared_state_digest':shared_digest,'init_eval_equal_C0':True,
             'effective_input_channels':0 if variant=='C0' else 13 if variant=='Ccap' else len(names),'sampler_RNG_unchanged_by_build':True,'formal_optimizer_steps':0}
        if variant=='C0':
            model.zero_grad(set_to_none=True);criterion(model(x),y).backward()
            if not all(torch.isfinite(v.grad).all() for v in model.parameters() if v.grad is not None) or model.encoder.branch_s[0].weight.grad.abs().sum()==0 or model.head[1].weight.grad.abs().sum()==0:
                raise ValueError('C0 common encoder/head derivative failed')
            row['backbone_and_head_gradients_finite_nonzero']=True
        if variant!='C0':
            ast=model.adapter.state_dict();digest=state_digest(ast)
            if adapter_reference is None:adapter_reference=digest
            elif digest!=adapter_reference:raise ValueError('Adapters not exactly identical initialized')
            row['adapter_initial_state_digest']=digest
            model.zero_grad(set_to_none=True);loss=criterion(model(x,features),y)
            if not torch.isfinite(loss):raise ValueError('Nonfinite frozen FocalLoss including extrema')
            loss.backward()
            if model.adapter[-1].weight.grad.abs().sum()==0 or model.encoder.branch_s[0].weight.grad.abs().sum()==0:raise ValueError('No initial projection or backbone gradient')
            if model.adapter[0].weight.grad.abs().sum()!=0:raise ValueError('Zero projection initial conv gradient invariant failed')
            row['first_backward_projection_and_backbone_nonzero']=True;row['first_backward_adapter_convolutions_zero_due_zero_projection']=True
            # A copy with fixed small nonzero projection tests graph reachability;
            # it is not an optimizer update or a candidate checkpoint.
            probe=copy.deepcopy(model);probe.zero_grad(set_to_none=True)
            with torch.no_grad():probe.adapter[-1].weight.copy_(torch.linspace(-.01,.01,probe.adapter[-1].weight.numel()).reshape_as(probe.adapter[-1].weight))
            probe_features=features.clone().requires_grad_(True);probe_x=x.clone().requires_grad_(True)
            probe_loss=criterion(probe(probe_x,probe_features),y)
            if not torch.isfinite(probe_loss):raise ValueError('Nonfinite FocalLoss in isolated derivative probe')
            probe_loss.backward()
            grad=probe.adapter[0].weight.grad;active=torch.ones(13,dtype=torch.bool) if variant=='Ccap' else model.feature_selector.flatten().bool();inactive=~active
            if inactive.any() and grad[:,inactive].abs().sum()!=0:raise ValueError('Excluded feature has nonzero structural gradient')
            if not all(torch.isfinite(v.grad).all() for v in probe.parameters() if v.grad is not None) or not torch.isfinite(grad).all() or not (grad[:,active].abs().sum(dim=(0,2))>0).all():raise ValueError('Nonfinite or missing active-feature derivative, including actual extrema')
            if variant!='Ccap' and (probe_features.grad is None or (inactive.any() and probe_features.grad[:,inactive].abs().sum()!=0)):raise ValueError('Bad candidate selector derivative')
            row['nonzero_probe_active_conv_grad_finite_nonzero']=True;row['nonzero_probe_inactive_input_grad_zero']=True
            row['structurally_input_connected_adapter_parameters']=2976+48*int(active.sum())
            row['ignored_first_conv_weight_coordinates']=48*int(inactive.sum())
            before_bn={k:v.clone() for k,v in model.state_dict().items()};model.train();invalid=features.clone();invalid[0,0,0]=float('nan')
            try:model(x,invalid)
            except ValueError:pass
            else:raise ValueError('Invalid full13 was not rejected')
            if not all(torch.equal(v,model.state_dict()[k]) for k,v in before_bn.items()):raise ValueError('Invalid input reached BatchNorm')
            row['invalid_even_excluded_channels_rejected_before_BN']=True;model.eval()
        row['dense_conv_linear_MACs_single_window']=compute_macs(model,x[:1],features[:1])
        rows.append(row)
    result={'status':'PASS_M3_CPU_STRUCTURE_INITIALIZATION_AND_DERIVATIVES','checked_at':datetime.now(timezone.utc).isoformat(),'device':'cpu',
            'environment':{'torch':str(torch.__version__),'numpy':np.__version__},'anchor_path':str(anchor),'anchor_sha256':sha(anchor),'anchor_state_digest':full_shared_digest,
            'legacy_untrained_mapping_checked':True,'formal_environment_fresh_seed_reproduction':'PENDING GPU-environment read-only check before future training; no server contact this turn',
            'actual_source_train_batch_original_IDs':ds.original_cache_indices[indices].tolist(),'actual_extreme_original_IDs':{'largest_abs_N1':int(ds.original_cache_indices[n1_extreme]),'largest_O6':int(ds.original_cache_indices[o6_extreme])},'checked_batch_max_abs_normalized_candidates':features.abs().amax(dim=(0,2)).tolist(),'variants':rows,'normalization_sha256':sha(a.normalization),
            'loss_for_derivative_checks':'Unchanged FocalLoss gamma2 alphaNone','optimizer_objects_created':0,'optimizer_steps':0,'learned_weights_changed':False,'diagnostic_copy_projection_changed_for_derivative_check_only':True,
            'training_loop_or_GPU_server_Test_Real_used':False,'MAC_definition':'Conv/Linear MACs from actual CPU shape hooks; excludes BN/activation/pooling/feature calculation; all13 slot convolutions execute densely regardless selector.'}
    a.output.mkdir(parents=True,exist_ok=False);(a.output/'model_preflight.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
