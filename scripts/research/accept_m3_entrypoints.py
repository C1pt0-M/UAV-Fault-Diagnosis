"""Only changed M3 fixed-anchor and new entrypoint checks; no optimizer steps."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,json,sys,traceback
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'packages/uav_fault'),str(ROOT/'scripts/research')]
from train_causal_m3 import protocol,prepare,build,configure,evaluate,Indexed,forward,read,source,state_digest
from uav_fault.models.cnn.mstca_m3 import VARIANTS,load_fixed_adapter
from replay_causal_m3 import replay,candidate_window
from replay_m1_continuous import aggregate_scores, stream_check
from torch.utils.data import DataLoader


def check(args):
    p,e=protocol(args.protocol)
    actual=source.environment()
    for k,v in p['formal_environment'].items():
        if actual[k]!=v:raise ValueError('Wrong formal version '+k)
    prior=read(ROOT/e['prior_reproduction_diagnostic'])
    if not prior['fresh_legacy_seed42_matches_archived'] or any(not x['all_epoch_IDs_match_local'] for x in prior['samplers'].values()):raise ValueError('Prior passed M0/sampler evidence missing')
    configure(p);ds,frames,weights=prepare(args.data,args.features,p)
    samples=sorted(set([0,len(ds['train'])//2,len(ds['train'])-1,98072 if len(ds['train'])>98072 else 1]))
    batch=ds['train'].__getitems__(samples)
    for i,item in zip(samples,batch):
        scalar=ds['train'][i]
        if any(not torch.equal(a,b) for a,b in zip(item,scalar)):raise ValueError('Batched/scalar candidate calculation differs')
    x=torch.stack([v[0] for v in batch]);f=torch.stack([v[1] for v in batch]);y=torch.tensor([int(v[2]) for v in batch])
    cpu_ref=build(p,'C0',args.initial_anchor).eval()
    with torch.no_grad():ref=cpu_ref(x)
    rows={};gpu_ref=None
    for variant in VARIANTS:
        model=build(p,variant,args.initial_anchor).eval()
        if variant!='C0' and any(not torch.equal(v,model.adapter.state_dict()[k]) for k,v in load_fixed_adapter().items()):raise ValueError('Adapter not bitwise equal')
        common={k:v for k,v in model.state_dict().items() if k.startswith(('encoder.','head.'))}
        if state_digest(common)!=p['initialization']['shared_state_digest']:raise ValueError('M0 identity differs')
        with torch.no_grad():z=forward(model,variant,x,f)
        if not torch.equal(z,ref):raise ValueError('Same-device initial CPU outputs differ')
        model.to('cuda:0');torch.cuda.reset_peak_memory_stats();model.train()
        gx=x.repeat(512//len(x)+1,1,1)[:512].to('cuda:0');gf=f.repeat(512//len(x)+1,1,1)[:512].to('cuda:0');gy=y.repeat(512//len(y)+1)[:512].to('cuda:0')
        z=forward(model,variant,gx,gf);loss=source.FocalLoss(gamma=2)(z,gy)
        if not torch.isfinite(loss):raise ValueError('Gate loss nonfinite')
        loss.backward()
        for name,v in model.named_parameters():
            if v.grad is not None and not torch.isfinite(v.grad).all():raise ValueError('Gate gradient nonfinite '+name)
        peak=int(torch.cuda.max_memory_allocated());model.zero_grad(set_to_none=True)
        model.eval()
        # Compare identical device/mode; cross CPU/GPU max difference is descriptive only.
        cpu_model=build(p,variant,args.initial_anchor).eval();model.load_state_dict(cpu_model.state_dict())
        with torch.no_grad():gz=forward(model,variant,x.to('cuda:0'),f.to('cuda:0')).cpu()
        if gpu_ref is None:gpu_ref=gz
        elif not torch.equal(gz,gpu_ref):raise ValueError('Same GPU initial variant output differs')
        rows[variant]=dict(adapter_exact=True,common_exact=True,cpu_initial_equal=True,gpu_initial_equal=True,cross_device_max_abs_error=float((gz-ref).abs().max()),finite_training_mode_loss_and_gradients=True,batch512_peak_bytes=peak)
        del model,gx,gf,gy,z,loss;torch.cuda.empty_cache()
    # New validation entry actually consumes all original IDs, no weights update.
    valmodel=build(p,'C0',args.initial_anchor).to('cuda:0')
    loader=DataLoader(Indexed(ds['val']),batch_size=512,shuffle=False,num_workers=0,generator=torch.Generator().manual_seed(143))
    metrics,probability,labels=evaluate(valmodel,'C0',loader,'cuda:0',frames['val'])
    if len(labels)!=32381 or set(metrics['six_faults'])!=set(e['per_fault_delta_pp']) or set(metrics['domains'])!={'SIL','HIL'}:raise ValueError('Validation population/strata mismatch')
    # Deterministic input checks for the new replay wrapper: O8 reanchors each window,
    # shared N sequence remains continuous, cold prefix and invalid gap exclude windows.
    bn=read(args.data/'norm_stats.json');norm=read(ROOT/p['normalization']['path'])
    fixed=read(ROOT/p['evaluation']['continuous_replay']['plan_path'])['cases'][0]
    key=fixed['case_uid'].replace('/','__')+'.npz'
    with np.load(args.replay_base/'cases'/key,allow_pickle=False) as z:real_base={k:z[k] for k in z.files}
    with np.load(args.features/'cases'/key,allow_pickle=False) as z:real_side={k:z[k] for k in z.files}
    stream_check(real_side,read(args.data/'input_contract.json')['max_age_us'])
    real_model=build(p,'CN',args.initial_anchor).eval()
    real_result=replay(real_model,'CN',real_base,real_side,bn,norm,torch.device('cpu'))
    if not real_result['available'].any():raise ValueError('Fixed actual continuous record unavailable')
    n=360;grid=1000000+np.arange(n,dtype=np.int64)*20000;raw=np.ones((n,26),dtype=float)
    raw[:,bn['columns'].index('baro_alt_relative')]=10+np.arange(n)*.01
    base=dict(timestamp_us=grid,canonical_rows=raw,row_input_valid=np.ones(n,bool))
    side=dict(timestamp_us=grid,residual=np.zeros((n,4)),valid=np.ones((n,4),bool))
    model=build(p,'CN',args.initial_anchor).eval()
    ongoing=replay(model,'CN',base,side,bn,norm,torch.device('cpu'))
    cold=replay(model,'CN',base,side,bn,norm,torch.device('cpu'),cold=True)
    if np.flatnonzero(ongoing['available'])[0]!=149 or np.flatnonzero(cold['available'])[0]!=159:raise ValueError('Cold/ongoing admission wrong')
    side['valid'][180]=False
    gap=replay(model,'CN',base,side,bn,norm,torch.device('cpu'))
    if gap['available'][180:330].any() or not gap['available'][330]:raise ValueError('Gap150 flushing qualification wrong')
    xp=x[0];candidate=candidate_window(xp,np.zeros((4,150)),norm)
    o8=(abs(xp[bn['columns'].index('vel_z'),0])-norm['mean'][7])/norm['std'][7]
    if abs(float(candidate[7,0]-o8))>1e-5:raise ValueError('O8 first difference not zero per window')
    score=np.full(200,.8);score[100:]=np.nan
    events=aggregate_scores(np.arange(200)*.02,score)
    if events['finalized_events'][0]['emission_reason']!='invalid_gap_flush' or events['finalized_events'][0]['emitted_t']!=2.:raise ValueError('Gap emitted clock wrong')
    return dict(status='PASS_M3_FIXED_ANCHOR_AND_ENTRYPOINTS',execution_sha256=source.sha(args.protocol),checked_at=datetime.now(timezone.utc).isoformat(),models=rows,source_windows={'train':len(ds['train']),'val':len(ds['val'])},full_Val_entry_verified=True,batched_scalar_exact=True,O8_per_window_checked=True,N_continuous_contract_inherited_unchanged=True,native_stream_case=fixed['case_uid'],real_replay_windows=int(real_result['available'].sum()),cold_and_gap_replay_checked=True,prior_M0_and_8x100_sampler_evidence_reused=True,optimizer_steps=0,formal_jobs_started=0,validation_metrics_untrained_not_performance=metrics)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for n in ['protocol','data','features','replay-base','initial-anchor','output']:ap.add_argument('--'+n,type=Path,required=True)
    a=ap.parse_args()
    if a.output.exists():raise FileExistsError('Never overwrite a gate')
    try:result=check(a)
    except Exception as exc:
        a.output.write_text(json.dumps(dict(status='FAILED_M3_FIXED_ANCHOR_ENTRYPOINTS',error=repr(exc),traceback=traceback.format_exc(),formal_jobs_started=0),indent=2)+'\n',encoding='utf-8');raise
    source.write(a.output,result);print(result['status'],flush=True)


if __name__=='__main__':main()
