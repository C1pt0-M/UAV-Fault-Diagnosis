"""Verify saved continuous source-val arrays and summarize response/false events.

No inference, training, threshold fitting, or final-test/Real reads. Checks every
saved window's base/residual eligibility and recomputes metrics from its score.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'packages/uav_fault'),str(ROOT/'scripts/research')]
from replay_m1_continuous import aggregate_scores, stats
from uav_fault.online.detection_policy import OnlineDetectionPolicy


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def full_windows(valid):
    valid=np.asarray(valid,dtype=bool)
    sums=np.r_[0,np.cumsum(valid.astype(np.int64))]
    out=np.zeros(len(valid),dtype=bool)
    out[149:]=sums[150:]-sums[:-150]==150
    return out


def without_host_clock(events):
    result=json.loads(json.dumps(events))
    for event in result['finalized_events']:
        event['deployment_timing'].pop('confirmed_at_unix_s',None)
        event['deployment_timing'].pop('emitted_at_unix_s',None)
    return result


def metric_counts(base,scores):
    valid=np.isfinite(scores);known=base['row_label_valid'];y=base['reference_state']
    normal=valid & known & (y==0);fault=valid & known & (y==1)
    return {'normal_windows':int(normal.sum()),'normal_false_positive_windows':int(np.count_nonzero(scores[normal]>=.5)),
            'fault_windows':int(fault.sum()),'detected_fault_windows':int(np.count_nonzero(scores[fault]>=.5)),
            'normal_effective_exposure_s':float(normal.sum()*.02),'normal_raw_known_exposure_s':float(np.count_nonzero(known & (y==0))*.02)}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ['plan','data','features','replays','output']:
        ap.add_argument('--'+name,type=Path,required=True)
    a=ap.parse_args()
    if a.output.exists():raise FileExistsError('Preserve replay verification')
    p=read(a.plan);policy=OnlineDetectionPolicy(**p['policy']);main=read(a.replays/'continuous_replay_v2/summary.json');natural=read(a.replays/'natural_availability_replay_v2/summary.json')
    assert main['status']=='COMPLETE_SOURCE_VAL_CONTINUOUS_REPLAY' and natural['status']=='COMPLETE_NATIVE_AVAILABILITY_LATENCY_DIAGNOSTIC'
    plan_sha=hashlib.sha256(a.plan.read_bytes()).hexdigest()
    assert main['plan_sha256']==natural['plan_sha256']==plan_sha and main['policy']==p['policy']
    assert len(main['rows'])==72 and len(natural['rows'])==36
    rows=[];timings={};checked=0
    for case in p['cases']:
        uid=case['case_uid'];key=uid.replace('/','__')
        with np.load(a.data/'cases'/(key+'.npz'),allow_pickle=False) as z:base={k:z[k] for k in z.files}
        with np.load(a.features/'cases'/(key+'.npz'),allow_pickle=False) as z:side={k:z[k] for k in z.files}
        grid=base['timestamp_us']/1e6;common=full_windows(base['row_input_valid'] & side['valid'].all(1));native=full_windows(base['row_input_valid'])
        for arm in ['uniform_replacement','balanced_source_normal']:
            common_m0=None
            for variant in ['M0_matched','M1','M0_natural']:
                if variant=='M0_natural':
                    original=next(x for x in natural['rows'] if x['case_uid']==uid and x['arm']==arm)
                    path=a.replays/'natural_availability_replay_v2'/(key+'__'+arm+'__M0natural.npz');expected=native
                else:
                    original=next(x for x in main['rows'] if x['case_uid']==uid and x['arm']==arm and x['variant']==variant)
                    path=a.replays/'continuous_replay_v2'/(key+'__'+arm+'__'+variant+'.npz');expected=common
                with np.load(path,allow_pickle=False) as z:d={k:z[k] for k in z.files}
                scores=d['scores'];valid=np.isfinite(scores)
                assert np.array_equal(d['timestamps_s'],grid) and np.array_equal(d['available'],expected) and np.array_equal(valid,expected),str(path)
                assert np.all((scores[valid]>=0)&(scores[valid]<=1))
                events=aggregate_scores(grid,scores,policy=policy)
                assert without_host_clock(events)==without_host_clock(original['events'])
                if variant=='M0_matched':common_m0=scores
                if variant=='M0_natural':np.testing.assert_array_equal(scores[common],common_m0[common])
                counts=metric_counts(base,scores)
                first=grid[valid][0] if valid.any() else None
                original_first=original['first_prediction_s'] if variant=='M0_natural' else original['first_available_prediction_s']
                assert first==original_first
                onset=original['onset_reference_s'];measurable=bool(onset is not None and grid[0]<=onset<=grid[-1]);assert measurable==original['onset_measurable_in_crop']
                post=np.flatnonzero(valid & (grid>=onset)) if measurable else np.array([],dtype=int)
                cross=np.flatnonzero(valid & (scores>=.5) & (grid>=onset)) if measurable else np.array([],dtype=int)
                confirmations=[x for x in events['confirmations'] if measurable and x['candidate_start_t']>=onset]
                finalized=[x for x in events['finalized_events'] if measurable and x['start_t']>=onset]
                final=finalized[0] if finalized else None
                online=next((x for x in finalized if x['emission_reason']=='online_update'),None)
                spanning=[x for x in events['finalized_events'] if measurable and x['start_t']<onset<=x['end_t']]
                actual_cross=float(grid[cross[0]]) if len(cross) else None
                actual_final=final['emitted_t'] if final else None
                assert actual_cross==original['cross05_s' if variant=='M0_natural' else 'first_cross05_after_onset_s']
                assert actual_final==original['final_event_s' if variant=='M0_natural' else 'first_final_event_after_onset_s']
                false=[];false_confirmation=[]
                def normal_interval(start,end):
                    mask=(grid>=start)&(grid<=end)
                    return mask.any() and base['row_label_valid'][mask].all() and (base['reference_state'][mask]==0).all()
                for event in events['finalized_events']:
                    if normal_interval(event['start_t'],event['end_t']):false.append(event)
                for event in events['confirmations']:
                    if normal_interval(event['candidate_start_t'],event['confirmed_t']):false_confirmation.append(event)
                assert len(false)==original['false_events' if variant=='M0_natural' else 'false_final_events']
                if variant!='M0_natural':
                    for k in ['normal_effective_exposure_s','normal_raw_known_exposure_s']:assert np.isclose(counts[k],original[k],rtol=0,atol=1e-9)
                    if counts['normal_windows']:assert np.isclose(counts['normal_false_positive_windows']/counts['normal_windows'],original['window_normal_fpr'])
                    if counts['fault_windows']:assert np.isclose(counts['detected_fault_windows']/counts['fault_windows'],original['window_fault_recall'])
                row={'case_uid':uid,'domain':case['domain'],'subtype':case['subtype'],'kind':case['kind'],'arm':arm,'variant':variant,**counts,
                     'predictions':int(valid.sum()),'first_prediction_s':float(first) if first is not None else None,
                     'onset_reference_s':onset,'onset_measurable':measurable,'first_post_onset_delay_s':float(grid[post[0]]-onset) if len(post) else None,
                     'cross05_delay_s':actual_cross-onset if actual_cross is not None else None,
                     'candidate06_delay_s':confirmations[0]['candidate_start_t']-onset if confirmations else None,
                     'confirmation_delay_s':confirmations[0]['confirmed_t']-onset if confirmations else None,
                     'final_delay_s':actual_final-onset if actual_final is not None else None,'final_reason':final['emission_reason'] if final else None,
                     'online_final_delay_s':online['emitted_t']-onset if online else None,
                     'preexisting_alarm_spanning_onset':bool(spanning),'score_missed':bool(measurable and not len(cross)),
                     'new_confirmation_missing':bool(measurable and not confirmations),'new_final_missing':bool(measurable and not final),
                     'normal_false_final_events':len(false),'normal_false_online_final_events':sum(x['emission_reason']=='online_update' for x in false),
                     'normal_false_confirmations':len(false_confirmation),'normal_false_flush_events':sum(x['emission_reason']!='online_update' for x in false)}
                if variant!='M0_natural':
                    delta=d['simulated_result_ready_s'][valid]-grid[valid];inference=d['inference_ns'][valid]/1e6;preprocess=d['preprocess_ns'][valid]/1e6;backlog=d['queue_backlog_s']
                    assert np.isfinite(delta).all() and (delta>=0).all() and (backlog>=0).all()
                    assert np.count_nonzero(backlog>0)==original['frames_exceeding20ms_budget']
                    row['budget_exceeding_frames']=int(np.count_nonzero(backlog>0))
                    group=timings.setdefault((variant,arm),{'inference_ms':[],'preprocess_ms':[],'simulated_ready_ms':[],'simulated_backlog_ms':[]})
                    for k,v in [('inference_ms',inference),('preprocess_ms',preprocess),('simulated_ready_ms',delta*1000),('simulated_backlog_ms',backlog*1000)]:group[k].append(v)
                rows.append(row);checked+=1
    assert checked==108
    groups=[]
    for arm in ['uniform_replacement','balanced_source_normal']:
        for variant in ['M0_matched','M1','M0_natural']:
            selected=[r for r in rows if r['arm']==arm and r['variant']==variant]
            faults=[r for r in selected if r['kind']=='fault'];normal=[r for r in selected if r['kind']=='normal'];exp=sum(r['normal_effective_exposure_s'] for r in normal);raw=sum(r['normal_raw_known_exposure_s'] for r in normal)
            normal_n=sum(r['normal_windows'] for r in normal);fp=sum(r['normal_false_positive_windows'] for r in normal);fp_events=sum(r['normal_false_final_events'] for r in normal)
            group={'arm':arm,'variant':variant,'fault_cases':len(faults),'score_detected_cases':sum(not r['score_missed'] for r in faults),
                   'new_confirmed_cases':sum(not r['new_confirmation_missing'] for r in faults),'new_finalized_cases':sum(not r['new_final_missing'] for r in faults),
                   'online_finalized_cases':sum(r['online_final_delay_s'] is not None for r in faults),'preexisting_alarm_spanning_onset_cases':sum(r['preexisting_alarm_spanning_onset'] for r in faults),
                   'normal_cases':len(normal),'normal_windows':normal_n,'normal_false_positive_windows':fp,'normal_window_fpr':fp/normal_n if normal_n else None,
                   'normal_effective_exposure_s':exp,'normal_raw_known_exposure_s':raw,'false_final_events':fp_events,
                   'false_final_events_per_eligible_hour':fp_events*3600/exp if exp else None,'false_final_events_per_raw_hour':fp_events*3600/raw if raw else None,
                   'false_online_final_events':sum(r['normal_false_online_final_events'] for r in normal),'false_flush_events':sum(r['normal_false_flush_events'] for r in normal),
                   'false_confirmations':sum(r['normal_false_confirmations'] for r in normal),
                   'response':{k:stats([r[k] for r in faults if r[k] is not None]) for k in ['first_post_onset_delay_s','cross05_delay_s','candidate06_delay_s','confirmation_delay_s','final_delay_s']}}
            if (variant,arm) in timings:group['compute']={k:stats(np.concatenate(v)) for k,v in timings[(variant,arm)].items()}
            groups.append(group)
    differences=[]
    for arm in ['uniform_replacement','balanced_source_normal']:
        for subtype in ['motor','accelerometer','gyroscope','barometer','magnetometer','GPS']:
            for domain in ['SIL','HIL']:
                m1=next(r for r in rows if r['arm']==arm and r['subtype']==subtype and r['domain']==domain and r['variant']=='M1')
                base=next(r for r in rows if r['case_uid']==m1['case_uid'] and r['arm']==arm and r['variant']=='M0_natural')
                differences.append({'case_uid':m1['case_uid'],'arm':arm,'subtype':subtype,'domain':domain,
                                    **{k: m1[k]-base[k] if m1[k] is not None and base[k] is not None else None for k in ['cross05_delay_s','confirmation_delay_s','final_delay_s']},
                                    'M0_preexisting_alarm_spanning_onset':base['preexisting_alarm_spanning_onset']})
    result={'status':'PASS_ALL108_REPLAYS_MASK_SCORE_EVENTS_TIMING_AND_EXPOSURE_RECOMPUTED','checked_at':datetime.now(timezone.utc).isoformat(),'plan_sha256':plan_sha,
            'rows':rows,'groups':groups,'M1_minus_naturally_available_M0_delay_s':differences,
            'limits':'18 exposed source-val hover cases, six normal and12 paired fault cases. Event rates include actual gap/end flush and are not steady-flight rate estimates. Simulation times exclude message push/decode, policy/session cost, browser/network and actual device clocks. No physical-damage truth, blind test, added inference or training.'}
    a.output.mkdir(parents=True,exist_ok=False);(a.output/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(groups,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
