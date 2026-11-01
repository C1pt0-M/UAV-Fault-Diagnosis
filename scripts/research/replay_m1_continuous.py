"""Source-development M0/M1 continuous timing; no policy or threshold tuning.

Uses real deployment policy/aggregator/session, not lead_time_s. Recorded GTD
activation is a reference, never invented from a selected-window label.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'packages/uav_fault'));sys.path.insert(0,str(ROOT/'scripts/research'))
from uav_fault.online.detection_policy import OnlineDetectionPolicy
from uav_fault.online.monitoring_session import OnlineMonitoringSession
from uav_fault.online.event_aggregator import event_timing
from uav_fault.causal_input import canonical_window
from uav_fault.models.cnn.lstm import OnlineMSTCA
from uav_fault.models.cnn.mstca_kinematic_m1 import OnlineMSTCAKinematicM1
from continuous_kinematic_residuals import ContinuousResidualStream
from validate_kinematic_residuals import sha


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def stats(x):
    x=np.asarray(x,dtype=float)
    return {'count':len(x),'median':float(np.median(x)) if len(x) else None,'p90':float(np.quantile(x,.9)) if len(x) else None,'max':float(np.max(x)) if len(x) else None}


def aggregate_scores(times, probabilities, *, policy=None):
    times=np.asarray(times,dtype=float);probabilities=np.asarray(probabilities,dtype=float)
    if times.shape!=probabilities.shape or np.any(np.diff(times)<=0):raise ValueError('Ordered inference timestamps required')
    policy=policy or OnlineDetectionPolicy();agg=policy.make_aggregator();session=OnlineMonitoringSession()
    runner=SimpleNamespace(aggregator=agg,flush=agg.flush);session.start_detection(runner)
    events=[];confirmed=[];previous_available=False
    def record(ev,reason):
        timing=event_timing(ev)
        events.append({'start_t':ev.start_t,'end_t':ev.end_t,'confirmed_t':timing.get('confirmed_t'),'emitted_t':timing.get('emitted_t'),'emission_reason':reason,'deployment_timing':timing})
    for i,(t,p) in enumerate(zip(times,probabilities),1):
        if not np.isfinite(p):
            if previous_available:
                for ev in agg.flush():record(ev,'invalid_gap_flush')
                agg=policy.make_aggregator();runner=SimpleNamespace(aggregator=agg,flush=agg.flush);session=OnlineMonitoringSession();session.start_detection(runner)
            previous_available=False;continue
        previous_available=True
        before=getattr(agg,'_active',None)
        new=agg.update(t=float(t),fault_prob=float(p),frame_idx=i)
        active=getattr(agg,'_active',None)
        if before is None and active is not None:
            confirmed.append({'candidate_start_t':active.start_t,'confirmed_t':active.confirmed_t})
        session.ingest_external(result={'fault_prob':float(p),'label':'fault' if p>=.5 else 'normal'},
            new_events=new,frame_idx=i,time_s=float(t),onset_th=policy.onset_th)
        for ev in new:record(ev,'online_update')
    for ev in agg.flush():record(ev,'end_of_stream_flush')
    return {'confirmations':confirmed,'finalized_events':events,'policy':policy.to_dict()}


def stream_check(sidecar, limits):
    d=sidecar;grid=d['timestamp_us'];pre=d['prehistory_timestamp_us'];origin=int(pre[0]);extended=np.r_[pre,grid]
    valid=np.r_[d['prehistory_input_valid'],d['original_row_input_valid']]
    engine=ContinuousResidualStream({g:limits[g] for g in ['baro','velocity','attitude','gyro']},origin_us=origin)
    groups=['baro','velocity','attitude','gyro'];positions={g:0 for g in groups};out=[];dur=[]
    for t,base_valid in zip(extended,valid):
        # Merge chronological arrivals without ever sorting a reversed topic.
        while True:
            candidates=[(int(d[g+'__native_available_us'][positions[g]]),k,g) for k,g in enumerate(groups) if positions[g]<len(d[g+'__native_available_us']) and d[g+'__native_available_us'][positions[g]]<=t]
            if not candidates:break
            available,_,g=min(candidates);j=positions[g]
            prefix=g+'__native_';names=[name[len(prefix):] for name in d if name.startswith(prefix) and name[len(prefix):] in ['baro_device_id','z_reset_counter','vz_reset_counter','quat_reset_counter']]
            engine.push(g,int(d[g+'__native_observation_us'][j]),available,d[g+'__native_values'][j],
                counters={name:d[prefix+name][j] for name in names},valid=bool(d[g+'__native_valid'][j]));positions[g]+=1
        r=engine.emit(int(t),input_valid=bool(base_valid));out.append(r['residual']);dur.append(r['compute_ns'])
    out=np.asarray(out)[10:];dur=np.asarray(dur)[10:]
    np.testing.assert_allclose(out,d['residual'],rtol=0,atol=1e-10,equal_nan=True)
    return dur


def replay_case(model,variant,base,residual,normalization,policy,device,*,common_valid,residual_ns=None):
    grid=base['timestamp_us'];times=grid/1e6;n=len(grid);scores=np.full(n,np.nan);infer_ns=np.zeros(n,dtype=np.int64);pre_ns=np.zeros(n,dtype=np.int64)
    norm=normalization['base'];rnorm=normalization['residual'];mu=np.array(rnorm['mean'],dtype=np.float32)[:,None];std=np.array(rnorm['std'],dtype=np.float32)[:,None]
    available=np.zeros(n,dtype=bool);backlog=np.zeros(n);finish=np.full(n,np.nan);last_finish=times[0];model.eval()
    residual_ns=np.zeros(n,dtype=np.int64) if residual_ns is None else np.asarray(residual_ns,dtype=np.int64)
    if residual_ns.shape!=(n,):raise ValueError('Per-grid residual costs required')
    with torch.inference_mode():
        for i in range(n):
            # Single-threaded20ms service-budget simulation includes residual
            # work every frame, even when no complete window can be inferred.
            last_finish=max(last_finish,times[i])+(residual_ns[i]/1e9 if variant=='M1' else 0.)
            backlog[i]=max(0,last_finish-times[i]-.02)
            if i<149:continue
            s=i-149
            if not policy.should_infer(frame_idx=i+1,window_length=150,window_size=150):continue
            if not base['row_input_valid'][s:i+1].all() or not common_valid[s:i+1].all():continue
            begun=time.perf_counter_ns();x=canonical_window(base['canonical_rows'][s:i+1],norm['columns'],base['row_input_valid'][s:i+1],mean=norm['mean'],std=norm['std'])
            xt=torch.from_numpy(x[None]).to(device);r=(residual[s:i+1].astype(np.float32).T-mu)/std;rt=torch.from_numpy(r[None]).to(device)
            pre_ns[i]=time.perf_counter_ns()-begun
            if device.type=='cuda':torch.cuda.synchronize()
            begun=time.perf_counter_ns();z=model(xt,rt) if variant=='M1' else model(xt);p=z.softmax(1)[0,1].item()
            if device.type=='cuda':torch.cuda.synchronize()
            infer_ns[i]=time.perf_counter_ns()-begun;scores[i]=p;available[i]=True
            last_finish+=(pre_ns[i]+infer_ns[i])/1e9;finish[i]=last_finish;backlog[i]=max(0,last_finish-times[i]-.02)
    events=aggregate_scores(times,scores,policy=policy)
    return {'timestamps_s':times,'scores':scores,'available':available,'inference_ns':infer_ns,'preprocess_ns':pre_ns,'queue_backlog_s':backlog,'simulated_result_ready_s':finish,'events':events}


def first_full_window(grid, valid):
    valid=np.asarray(valid,dtype=bool)
    if len(valid)<150:return None
    ends=np.flatnonzero(np.convolve(valid.astype(np.int64),np.ones(150,dtype=np.int64),'valid')==150)+149
    return float(grid[ends[0]]/1e6) if len(ends) else None


def latency_groups(rows):
    groups=[]
    for variant in ['M0_matched','M1']:
        for arm in ['uniform_replacement','balanced_source_normal']:
            available=[r for r in rows if r.get('status')=='COMPLETE_REPLAY' and r['variant']==variant and r['arm']==arm]
            for subtype in ['motor','accelerometer','gyroscope','barometer','magnetometer','GPS']:
                selected=[r for r in available if r['subtype']==subtype and r['onset_measurable_in_crop']]
                groups.append({'variant':variant,'arm':arm,'subtype':subtype,'measurable_cases':len(selected),
                    'score_detections':sum(not r['score_missed_after_reference'] for r in selected),
                    'final_event_detections':sum(not r['final_event_missed_after_reference'] for r in selected),
                    'detection_rate':sum(not r['score_missed_after_reference'] for r in selected)/len(selected) if selected else None,
                    'first_prediction_delay_s':stats([r['first_prediction_delay_s'] for r in selected if r['first_prediction_delay_s'] is not None]),
                    'cross05_delay_s':stats([r['cross05_delay_s'] for r in selected if r['cross05_delay_s'] is not None]),
                    'confirmed_delay_s':stats([r['confirmed_delay_s'] for r in selected if r['confirmed_delay_s'] is not None]),
                    'final_emission_delay_s':stats([r['final_emission_delay_s'] for r in selected if r['final_emission_delay_s'] is not None]),
                    'missed_score_case_ids':[r['case_uid'] for r in selected if r['score_missed_after_reference']],
                    'missed_final_event_case_ids':[r['case_uid'] for r in selected if r['final_event_missed_after_reference']],
                    'final_events_flush_only':[r['case_uid'] for r in selected if r['first_final_event_emission_reason']=='end_of_stream_flush'],
                    'final_event_detection_rate':sum(not r['final_event_missed_after_reference'] for r in selected)/len(selected) if selected else None,
                    'not_independent_or_blind':True})
    return groups


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--plan',type=Path,required=True);ap.add_argument('--data',type=Path,required=True);ap.add_argument('--features',type=Path,required=True);ap.add_argument('--runs',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--device',choices=['cpu','cuda:0'],default='cpu');a=ap.parse_args()
    if a.output.exists():raise FileExistsError('Preserve existing replay')
    p=read(a.plan);policy=OnlineDetectionPolicy(**p['policy']);a.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1)
    norm={'base':read(a.data/'norm_stats.json'),'residual':read(a.features/'residual_norm_stats.json')};limits=read(a.data/'input_contract.json')['max_age_us'];device=torch.device(a.device);rows=[]
    for case in p['cases']:
        uid=case['case_uid'];key=uid.replace('/','__');bp=a.data/'cases'/(key+'.npz');sp=a.features/'cases'/(key+'.npz')
        with np.load(bp,allow_pickle=False) as z:base={k:z[k] for k in z.files}
        with np.load(sp,allow_pickle=False) as z:side={k:z[k] for k in z.files}
        if 'prehistory_timestamp_us' not in side:
            rows.append({'case_uid':uid,'status':'UNAVAILABLE_NATIVE_SUPPORT','onset_not_guessed':True});continue
        residual_ns=stream_check(side,limits);common=side['valid'].all(1)
        cold=common.copy();cold[:10]=False
        grid=base['timestamp_us'];first_residual=np.flatnonzero(cold)
        startup={'stream_origin_s':float(grid[0]/1e6),
            'M0_cold_first_eligible_s':first_full_window(grid,base['row_input_valid']),
            'M1_cold_first_residual_s':float(grid[first_residual[0]]/1e6) if len(first_residual) else None,
            'M1_cold_first_eligible_s':first_full_window(grid,cold&base['row_input_valid']),
            'M1_ongoing_history_first_eligible_s':first_full_window(grid,common&base['row_input_valid']),
            'definition':'Cold service starts at crop first observed frame; ongoing starts with established original-native prehistory. These eligibility timestamps exclude inference wall time.'}
        startup['additional_cold_start_s']=startup['M1_cold_first_eligible_s']-startup['M0_cold_first_eligible_s'] if startup['M1_cold_first_eligible_s'] is not None and startup['M0_cold_first_eligible_s'] is not None else None
        for arm in ['uniform_replacement','balanced_source_normal']:
            for variant in ['M0_matched','M1']:
                run=a.runs/variant/arm;checkpoint=run/'best.pt'
                if not checkpoint.is_file():raise FileNotFoundError(str(checkpoint))
                model=OnlineMSTCAKinematicM1() if variant=='M1' else OnlineMSTCA(in_channels=26,hidden_size=64,dropout=.2)
                model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True),strict=True);model.to(device)
                result=replay_case(model,variant,base,side['residual'],norm,policy,device,common_valid=common,residual_ns=residual_ns)
                times,scores=result['timestamps_s'],result['scores'];finite=np.isfinite(scores);known=base['row_label_valid'];normal=known&(base['reference_state']==0)&finite;active=known&(base['reference_state']==1)&finite
                raw_events=case.get('gtd_events',[]);command=case.get('command_events',[]);onset=raw_events[0]['start_s'] if len(raw_events)==1 else None
                in_crop=onset is not None and times[0]<=onset<=times[-1]
                post=np.flatnonzero(finite&(times>=onset)) if in_crop else np.array([],dtype=int);cross=np.flatnonzero(finite&(scores>=.5)&(times>=onset)) if in_crop else np.array([],dtype=int)
                emitted=[e for e in result['events']['finalized_events'] if in_crop and e['start_t']>=onset]
                confirmed=[e for e in result['events']['confirmations'] if in_crop and e['candidate_start_t']>=onset]
                # Score false events only when the entire event interval has known normal reference.
                false=[]
                for e in result['events']['finalized_events']:
                    interval=(times>=e['start_t'])&(times<=e['end_t'])
                    if interval.any() and known[interval].all() and (base['reference_state'][interval]==0).all():false.append(e)
                fp_exposure=float(normal.sum()*.02)
                row={'case_uid':uid,'split':'val','domain':case['domain'],'subtype':case['subtype'],'kind':case['kind'],'arm':arm,'variant':variant,'status':'COMPLETE_REPLAY',
                    'onset_reference_s':onset,'command_reference_events':command,'onset_reference_is_physical_truth':False,'onset_measurable_in_crop':in_crop,
                    'first_available_prediction_s':float(times[finite][0]) if finite.any() else None,
                    'first_prediction_simulated_ready_s':float(result['simulated_result_ready_s'][finite][0]) if finite.any() else None,
                    'first_post_onset_simulated_ready_s':float(result['simulated_result_ready_s'][post[0]]) if len(post) else None,
                    'first_cross05_simulated_ready_s':float(result['simulated_result_ready_s'][cross[0]]) if len(cross) else None,
                    'first_prediction_after_onset_s':float(times[post[0]]) if len(post) else None,'first_cross05_after_onset_s':float(times[cross[0]]) if len(cross) else None,
                    'first_confirmed_after_onset_s':confirmed[0]['confirmed_t'] if confirmed else None,
                    'first_final_event_after_onset_s':emitted[0]['emitted_t'] if emitted else None,'first_final_event_emission_reason':emitted[0]['emission_reason'] if emitted else None,
                    'first_online_final_event_after_onset_s':next((e['emitted_t'] for e in emitted if e['emission_reason']=='online_update'),None),
                    'preexisting_event_spanning_onset':bool(in_crop and any(e['start_t']<onset<=e['end_t'] for e in result['events']['finalized_events'])),
                    'reference_outside_replay_or_missing':not in_crop,
                    'no_post_onset_available_prediction':bool(in_crop and not len(post)),
                    'window_fault_recall':float((scores[active]>=.5).mean()) if active.any() else None,
                    'window_normal_fpr':float((scores[normal]>=.5).mean()) if normal.any() else None,
                    'normal_effective_exposure_s':fp_exposure,'normal_raw_known_exposure_s':float(np.count_nonzero(known&(base['reference_state']==0))*.02),
                    'false_final_events':len(false),'false_events_per_hour':len(false)*3600/fp_exposure if fp_exposure else None,
                    'false_online_final_events':sum(e['emission_reason']=='online_update' for e in false),
                    'false_gap_or_end_flush_events':sum(e['emission_reason']!='online_update' for e in false),
                    'warning_rate_exposure_note':'Prediction-eligible known-normal20ms bins; raw known-normal exposure also reported. Gaps not bridged. Short-case/hour rates descriptive only.',
                    'inference_ms':stats(result['inference_ns'][finite]/1e6),'preprocess_ms':stats(result['preprocess_ns'][finite]/1e6),'residual_ms':stats(residual_ns/1e6),
                    'prediction_interval_s':stats(np.diff(times[finite])),'simulated_compute_backlog_s':stats(result['queue_backlog_s']),
                    'frames_exceeding20ms_budget':int(np.count_nonzero(result['queue_backlog_s']>0)),
                    'compute_budget_scope':'Serial residual+normalization+device transfer+inference; no network/browser, ULog decoding or live receive certification',
                    'score_missed_after_reference':bool(in_crop and not len(cross)),'final_event_missed_after_reference':bool(in_crop and not emitted),
                    'events':result['events'],'checkpoint_sha256':sha(checkpoint),'startup':startup,'replay_only_not_live_deployment':True}
                # Negative/pre-existing events are not silently re-labelled as new detections.
                for name,field in [('first_prediction_delay_s','first_prediction_after_onset_s'),('cross05_delay_s','first_cross05_after_onset_s'),('confirmed_delay_s','first_confirmed_after_onset_s'),('final_emission_delay_s','first_final_event_after_onset_s')]:
                    row[name]=row[field]-onset if in_crop and row[field] is not None else None
                command_onset=command[0]['start_s'] if len(command)==1 else None
                row['command_onset_reference_s']=command_onset
                row['gtd_minus_command_reference_s']=onset-command_onset if onset is not None and command_onset is not None else None
                row['cross05_delay_relative_to_command_s']=row['first_cross05_after_onset_s']-command_onset if row['first_cross05_after_onset_s'] is not None and command_onset is not None else None
                row['final_delay_relative_to_command_s']=row['first_final_event_after_onset_s']-command_onset if row['first_final_event_after_onset_s'] is not None and command_onset is not None else None
                row['reference_uncertainty_note']='GTD activation and command logging are distinct events. Delays are reference-specific, not certified physical damage latency.'
                row['first_cross05_compute_inclusive_delay_s']=row['first_cross05_simulated_ready_s']-onset if in_crop and row['first_cross05_simulated_ready_s'] is not None else None
                final_time=row['first_final_event_after_onset_s']
                final_index=int(np.searchsorted(times,final_time)) if final_time is not None else None
                row['first_final_event_simulated_ready_s']=float(result['simulated_result_ready_s'][final_index]) if final_index is not None and final_index<len(times) and np.isfinite(result['simulated_result_ready_s'][final_index]) else None
                row['final_compute_inclusive_delay_s']=row['first_final_event_simulated_ready_s']-onset if in_crop and row['first_final_event_simulated_ready_s'] is not None else None
                rows.append(row);np.savez_compressed(a.output/(key+'__'+arm+'__'+variant+'.npz'),**{k:v for k,v in result.items() if isinstance(v,np.ndarray)})
                print('REPLAY_CASE',uid,arm,variant,flush=True)
    summary={'status':'COMPLETE_SOURCE_VAL_CONTINUOUS_REPLAY','created_at':datetime.now(timezone.utc).isoformat(),'plan_sha256':sha(a.plan),'policy':p['policy'],'device':str(device),'rows':rows,'latency_by_fault_directory':latency_groups(rows),'scope':'Previously exposed small source-val hover; recorded GTD reference not physical damage; event flush distinguished; no final-test/Real/deployment certification.'}
    with (a.output/'summary.json').open('x',encoding='utf-8') as f:json.dump(summary,f,ensure_ascii=False,indent=2,allow_nan=False)


if __name__=='__main__':main()
