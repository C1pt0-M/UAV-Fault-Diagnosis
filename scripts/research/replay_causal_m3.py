"""Frozen M3 continuous replay path, using original publication stream/policy."""
from pathlib import Path
import argparse
import json
import sys
import time
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'packages/uav_fault'),str(ROOT/'scripts/research')]
from uav_fault.causal_input import canonical_window
from uav_fault.data.m3_features import old_candidate_features
from uav_fault.online.detection_policy import OnlineDetectionPolicy
from replay_m1_continuous import aggregate_scores,stream_check,first_full_window,stats
from train_causal_m3 import read,protocol,build,forward,source


def candidate_window(x, residual, normalization):
    if residual.shape!=(4,150) or not np.isfinite(residual).all():
        raise ValueError('Full150 signed continuous N data required')
    old=old_candidate_features(x.unsqueeze(0))[0]
    mu=torch.tensor(normalization['mean'],dtype=torch.float32).view(13,1)
    std=torch.tensor(normalization['std'],dtype=torch.float32).view(13,1)
    values=torch.cat([old,torch.from_numpy(residual.astype(np.float32))])
    out=(values-mu)/std
    if not torch.isfinite(out).all():raise ValueError('Invalid normalized candidate window')
    return out


def replay(model,variant,base,side,base_norm,norm,device,*,cold=False,natural=False,residual_ns=None):
    grid=base['timestamp_us'];times=grid/1e6;n=len(grid)
    if not np.array_equal(grid,side['timestamp_us']):raise ValueError('Replay clocks differ')
    common=side['valid'].all(1).copy()
    if cold:common[:10]=False
    if natural and variant not in ['C0','Ccap']:raise ValueError('Natural Base26 supplement only C0/Ccap')
    policy=OnlineDetectionPolicy();scores=np.full(n,np.nan);available=np.zeros(n,bool)
    infer=np.zeros(n,np.int64);pre=np.zeros(n,np.int64);finish=np.full(n,np.nan);backlog=np.zeros(n)
    costs=np.zeros(n,np.int64) if residual_ns is None else residual_ns
    last=float(times[0]);model.eval()
    with torch.inference_mode():
        for i,t in enumerate(times):
            last=max(last,float(t))+(0 if natural else costs[i]/1e9)
            backlog[i]=max(0,last-t-.02)
            if i<149:continue
            start=i-149
            if not base['row_input_valid'][start:i+1].all() or (not natural and not common[start:i+1].all()):continue
            began=time.perf_counter_ns()
            x=torch.from_numpy(canonical_window(base['canonical_rows'][start:i+1],base_norm['columns'],base['row_input_valid'][start:i+1],mean=base_norm['mean'],std=base_norm['std']))
            if natural:
                # Ccap reads only valid Base26 slices; do not invent residuals.
                if variant=='Ccap':
                    aux=x[model.ccap_indices.cpu()].unsqueeze(0).to(device)
                    xt=x.unsqueeze(0).to(device)
            else:
                f=candidate_window(x,side['residual'][start:i+1].T,norm).unsqueeze(0).to(device)
            xt=x.unsqueeze(0).to(device)
            if device.type=='cuda':torch.cuda.synchronize()
            pre[i]=time.perf_counter_ns()-began;began=time.perf_counter_ns()
            if natural:
                z=model(xt) if variant=='C0' else model.head(model.encoder(xt)+model.adapter(aux))
                if not torch.isfinite(z).all():raise FloatingPointError('Nonfinite natural replay logits')
            else:z=forward(model,variant,xt,f)
            prob=z.softmax(1)[0,1].item()
            if device.type=='cuda':torch.cuda.synchronize()
            if not np.isfinite(prob):raise FloatingPointError('Nonfinite replay score')
            infer[i]=time.perf_counter_ns()-began;scores[i]=prob;available[i]=True
            last+=(pre[i]+infer[i])/1e9;finish[i]=last;backlog[i]=max(0,last-t-.02)
    return dict(timestamps_s=times,scores=scores,available=available,inference_ns=infer,preprocess_ns=pre,
                simulated_result_ready_s=finish,queue_backlog_s=backlog,events=aggregate_scores(times,scores,policy=policy))


def describe(case,base,result):
    times=result['timestamps_s'];scores=result['scores'];finite=np.isfinite(scores);known=base['row_label_valid'];state=base['reference_state']
    normal=known&(state==0);fault=known&(state==1);events=result['events']
    gtd=case.get('gtd_events',[]);onset=gtd[0]['start_s'] if len(gtd)==1 else None
    measurable=bool(onset is not None and times[0]<=onset<=times[-1])
    post=np.flatnonzero(finite&(times>=onset)) if measurable else []
    cross=np.flatnonzero(finite&(scores>=.5)&(times>=onset)) if measurable else []
    confirmed=[e for e in events['confirmations'] if measurable and e['candidate_start_t']>=onset]
    final=[e for e in events['finalized_events'] if measurable and e['start_t']>=onset]
    false=[]
    for e in events['finalized_events']:
        mask=(times>=e['start_t'])&(times<=e['end_t'])
        if mask.any() and normal[mask].all():false.append(e)
    false_confirm=0
    for e in events['confirmations']:
        mask=(times>=e['candidate_start_t'])&(times<=e['confirmed_t'])
        false_confirm+=int(mask.any() and normal[mask].all())
    exposure=float((normal&finite).sum()*.02)
    return dict(case_uid=case['case_uid'],domain=case['domain'],subtype=case['subtype'],kind=case['kind'],
        onset_reference_s=onset,onset_measurable=measurable,command_reference_events=case.get('command_events',[]),
        first_available_after_reference_s=float(times[post[0]]) if len(post) else None,
        first_cross05_after_reference_s=float(times[cross[0]]) if len(cross) else None,
        first_confirmation_s=confirmed[0]['confirmed_t'] if confirmed else None,
        first_final_emission_s=final[0]['emitted_t'] if final else None,
        first_final_emission_reason=final[0]['emission_reason'] if final else None,
        preexisting_cross_onset_event=bool(measurable and any(e['start_t']<onset<=e['end_t'] for e in events['finalized_events'])),
        no_post_reference_inference=bool(measurable and not len(post)),no_cross05=bool(measurable and not len(cross)),
        no_confirmation=bool(measurable and not confirmed),no_final_emission=bool(measurable and not final),
        normal_effective_exposure_s=exposure,normal_raw_known_exposure_s=float(normal.sum()*.02),
        false_confirmations=false_confirm,false_final_events=len(false),false_online_events=sum(e['emission_reason']=='online_update' for e in false),
        false_flush_events=sum(e['emission_reason']!='online_update' for e in false),false_events_per_effective_hour=len(false)*3600/exposure if exposure else None,
        fault_windows=int((fault&finite).sum()),fault_recall=float((scores[fault&finite]>=.5).mean()) if (fault&finite).any() else None,
        normal_windows=int((normal&finite).sum()),normal_fpr=float((scores[normal&finite]>=.5).mean()) if (normal&finite).any() else None,
        inference_ms=stats(result['inference_ns'][finite]/1e6),preprocessing_ms=stats(result['preprocess_ns'][finite]/1e6),
        backlog_s=stats(result['queue_backlog_s']),frames_over20ms=int((result['queue_backlog_s']>0).sum()),events=events)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ['protocol','data','features','replay-base','runs','initial-anchor','output']:ap.add_argument('--'+name,type=Path,required=True)
    ap.add_argument('--variant',required=True)
    ap.add_argument('--device',default='cuda:0')
    a=ap.parse_args();p,e=protocol(a.protocol)
    if a.variant not in e['jobs'] or a.output.exists():raise ValueError('Invalid variant or existing replay')
    plan=read(ROOT/p['evaluation']['continuous_replay']['plan_path'])
    if source.sha(ROOT/p['evaluation']['continuous_replay']['plan_path'])!=p['evaluation']['continuous_replay']['plan_sha256']:raise ValueError('Replay plan changed')
    model=build(p,a.variant,a.initial_anchor)
    checkpoint=a.runs/a.variant/'best.pt';model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True));device=torch.device(a.device);model.to(device)
    torch.set_num_threads(1);a.output.mkdir(parents=True,exist_ok=False)
    bn=read(a.data/'norm_stats.json');norm=read(ROOT/p['normalization']['path']);limits=read(a.data/'input_contract.json')['max_age_us'];rows=[]
    for case in plan['cases']:
        key=case['case_uid'].replace('/','__')+'.npz'
        bp=a.replay_base/'cases'/key;sp=a.features/'cases'/key
        with np.load(bp,allow_pickle=False) as z:base={k:z[k] for k in z.files}
        with np.load(sp,allow_pickle=False) as z:side={k:z[k] for k in z.files}
        # Existing native arrival algorithm is unchanged; check all fixed18 once per variant.
        costs=stream_check(side,limits)
        for cold,natural in [(False,False),(True,False)]+([(False,True)] if a.variant in ['C0','Ccap'] else []):
            name='natural' if natural else 'cold' if cold else 'ongoing'
            result=replay(model,a.variant,base,side,bn,norm,device,cold=cold,natural=natural,residual_ns=costs)
            row=describe(case,base,result);row.update(variant=a.variant,mode=name,checkpoint_sha256=source.sha(checkpoint),source_base_sha256=source.sha(bp),source_sidecar_sha256=source.sha(sp))
            rows.append(row);np.savez_compressed(a.output/(key[:-4]+'__'+name+'.npz'),**{k:v for k,v in result.items() if isinstance(v,np.ndarray)})
        source.write(a.output/'summary.json',dict(status='RUNNING' if len(rows)<18 else 'REPLAY_PROGRESS',variant=a.variant,rows=rows))
    source.write(a.output/'summary.json',dict(status='COMPLETE_M3_CONTINUOUS_REPLAY',variant=a.variant,rows=rows,physical_onset_truth=False,Test_Real_used=False))


if __name__=='__main__':main()
