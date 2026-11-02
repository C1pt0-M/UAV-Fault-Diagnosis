"""Additional replay diagnostic: native M0 availability vs real M1 availability.

No training or new model. The main matched-window replay is retained. M0 is not
artificially blocked by unavailable residuals when measuring actual responsiveness.
"""
from pathlib import Path
from datetime import datetime,timezone
import argparse,json,sys,time
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'packages/uav_fault'));sys.path.insert(0,str(ROOT/'scripts/research'))
from replay_m1_continuous import aggregate_scores,canonical_window,read,stats,OnlineDetectionPolicy,OnlineMSTCA
from validate_kinematic_residuals import sha


def replay_base_only(model,base,norm,policy,device):
    times=base['timestamp_us']/1e6;n=len(times);scores=np.full(n,np.nan);cost=np.zeros(n,dtype=np.int64);model.eval()
    with torch.inference_mode():
        for i in range(149,n):
            s=i-149
            if not policy.should_infer(frame_idx=i+1,window_length=150,window_size=150) or not base['row_input_valid'][s:i+1].all():continue
            x=canonical_window(base['canonical_rows'][s:i+1],norm['columns'],base['row_input_valid'][s:i+1],mean=norm['mean'],std=norm['std']);tensor=torch.from_numpy(x[None]).to(device)
            if device.type=='cuda':torch.cuda.synchronize()
            start=time.perf_counter_ns();scores[i]=model(tensor).softmax(1)[0,1].item()
            if device.type=='cuda':torch.cuda.synchronize()
            cost[i]=time.perf_counter_ns()-start
    return {'timestamps_s':times,'scores':scores,'available':np.isfinite(scores),'inference_ns':cost,'events':aggregate_scores(times,scores,policy=policy)}


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--plan',type=Path,required=True);ap.add_argument('--data',type=Path,required=True);ap.add_argument('--features',type=Path,required=True);ap.add_argument('--runs',type=Path,required=True);ap.add_argument('--main-replay',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--device',default='cuda:0');a=ap.parse_args()
    if a.output.exists():raise FileExistsError('Preserve natural availability diagnostic')
    p=read(a.plan);main=read(a.main_replay/'summary.json');assert main['status']=='COMPLETE_SOURCE_VAL_CONTINUOUS_REPLAY'
    policy=OnlineDetectionPolicy(**p['policy']);norm={'base':read(a.data/'norm_stats.json'),'residual':read(a.features/'residual_norm_stats.json')};a.output.mkdir(parents=True,exist_ok=False);device=torch.device(a.device);torch.set_num_threads(1);rows=[]
    for case in p['cases']:
        uid=case['case_uid'];key=uid.replace('/','__');path=a.data/'cases'/(key+'.npz')
        with np.load(path,allow_pickle=False) as z:base={k:z[k] for k in z.files}
        for arm in ['uniform_replacement','balanced_source_normal']:
            model=OnlineMSTCA(in_channels=26,hidden_size=64,dropout=.2);checkpoint=a.runs/'M0_matched'/arm/'best.pt';model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True));model.to(device)
            # Matching training population is unchanged. Only runtime availability
            # returns to the actual M0 base-only contract; no residuals used.
            r=replay_base_only(model,base,norm['base'],policy,device)
            times,scores=r['timestamps_s'],r['scores'];valid=np.isfinite(scores);events=case['gtd_events'];onset=events[0]['start_s'] if len(events)==1 else None;measurable=bool(onset is not None and times[0]<=onset<=times[-1])
            post=np.flatnonzero(valid&(times>=onset)) if measurable else np.array([],dtype=int);cross=np.flatnonzero(valid&(scores>=.5)&(times>=onset)) if measurable else np.array([],dtype=int)
            final=[e for e in r['events']['finalized_events'] if measurable and e['start_t']>=onset];conf=[e for e in r['events']['confirmations'] if measurable and e['candidate_start_t']>=onset]
            normal=base['row_label_valid']&(base['reference_state']==0)&valid;active=base['row_label_valid']&(base['reference_state']==1)&valid;fp=[]
            for e in r['events']['finalized_events']:
                mask=(times>=e['start_t'])&(times<=e['end_t'])
                if mask.any() and base['row_label_valid'][mask].all() and (base['reference_state'][mask]==0).all():fp.append(e)
            m1=next(x for x in main['rows'] if x.get('case_uid')==uid and x.get('arm')==arm and x.get('variant')=='M1')
            with np.load(a.main_replay/(key+'__'+arm+'__M1.npz'),allow_pickle=False) as saved:
                m1_count=int(np.count_nonzero(saved['available']))
            base_row={'case_uid':uid,'kind':case['kind'],'subtype':case['subtype'],'domain':case['domain'],'arm':arm,'variant':'M0_matched_natural_base_availability',
                'first_prediction_s':float(times[valid][0]) if valid.any() else None,'post_onset_first_prediction_s':float(times[post[0]]) if len(post) else None,'cross05_s':float(times[cross[0]]) if len(cross) else None,
                'confirmed_s':conf[0]['confirmed_t'] if conf else None,'final_event_s':final[0]['emitted_t'] if final else None,'final_emission_reason':final[0]['emission_reason'] if final else None,
                'onset_reference_s':onset,'onset_measurable_in_crop':measurable,'physical_damage_truth':False,'inference_ms':stats(r['inference_ns'][valid]/1e6),'valid_prediction_windows':int(valid.sum()),
                'normal_exposure_s':float(normal.sum()*.02),'window_normal_fpr':float((scores[normal]>=.5).mean()) if normal.any() else None,'window_fault_recall':float((scores[active]>=.5).mean()) if active.any() else None,
                'false_events':len(fp),'false_events_per_hour':len(fp)*3600/(normal.sum()*.02) if normal.any() else None,'events':r['events'],
                'M1_valid_prediction_windows':m1_count,
                'M1_cross05_s':m1['first_cross05_after_onset_s'],'M1_final_event_s':m1['first_final_event_after_onset_s'],'M1_final_emission_reason':m1['first_final_event_emission_reason'],
                'M1_minus_M0_cross05_delay_s':m1['first_cross05_after_onset_s']-times[cross[0]] if len(cross) and m1['first_cross05_after_onset_s'] is not None else None,
                'M1_minus_M0_final_delay_s':m1['first_final_event_after_onset_s']-final[0]['emitted_t'] if final and m1['first_final_event_after_onset_s'] is not None else None,
                'scope':'Identical training population, native runtime availability different and explicitly reported. Not window-population fair FPR ranking; use main common-ID metrics for model increment.'}
            rows.append(base_row);np.savez_compressed(a.output/(key+'__'+arm+'__M0natural.npz'),timestamps_s=times,scores=scores,available=r['available'],inference_ns=r['inference_ns']);print('NATURAL_M0_REPLAY',uid,arm,flush=True)
    groups=[]
    for arm in ['uniform_replacement','balanced_source_normal']:
        for subtype in ['motor','accelerometer','gyroscope','barometer','magnetometer','GPS']:
            selected=[r for r in rows if r['arm']==arm and r['subtype']==subtype and r['onset_measurable_in_crop']]
            groups.append({'arm':arm,'subtype':subtype,'cases':len(selected),'M1_minus_M0_cross05_delay_s':stats([r['M1_minus_M0_cross05_delay_s'] for r in selected if r['M1_minus_M0_cross05_delay_s'] is not None]),'M1_minus_M0_final_delay_s':stats([r['M1_minus_M0_final_delay_s'] for r in selected if r['M1_minus_M0_final_delay_s'] is not None]),'M0_score_missed_case_ids':[r['case_uid'] for r in selected if r['cross05_s'] is None],'M1_score_missed_case_ids':[r['case_uid'] for r in selected if r['M1_cross05_s'] is None]})
    result={'status':'COMPLETE_NATIVE_AVAILABILITY_LATENCY_DIAGNOSTIC','completed_at':datetime.now(timezone.utc).isoformat(),'plan_sha256':sha(a.plan),'main_replay_summary_sha256':sha(a.main_replay/'summary.json'),'rows':rows,'fault_directory_delay_differences':groups,'new_training_or_final_test_Real':False,'not_actual_live_deployment':True}
    (a.output/'summary.partial.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+chr(10),encoding='utf-8')
    (a.output/'summary.partial.json').replace(a.output/'summary.json')


if __name__=='__main__':main()
