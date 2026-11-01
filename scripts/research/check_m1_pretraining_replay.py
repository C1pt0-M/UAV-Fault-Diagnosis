"""Basic real continuous replay gate before any GPU training.

Uses only registered source-val native sidecars and trusted prior event metadata.
No trained M1 required. Benchmarks residual stream and exact M0-shape CPU forward.
"""
import argparse
from pathlib import Path
from datetime import datetime,timezone
import json
import sys
import time
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'packages/uav_fault'));sys.path.insert(0,str(ROOT/'scripts/research'))
from replay_m1_continuous import read,stats,stream_check,first_full_window,aggregate_scores
from continuous_kinematic_residuals import ContinuousResidualStream
from uav_fault.models.cnn.mstca_kinematic_m1 import OnlineMSTCAKinematicM1
from uav_fault.models.cnn.lstm import OnlineMSTCA
from validate_kinematic_residuals import sha


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',type=Path,required=True);p.add_argument('--data',type=Path,required=True);p.add_argument('--features',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError('Preserve pretraining replay check')
    plan=read(a.plan);limits=read(a.data/'input_contract.json')['max_age_us'];rows=[];torch.set_num_threads(1)
    for selected in plan['cases']:
        uid=selected['case_uid'];key=uid.replace('/','__');receipt=read(a.data/'case_records'/(key+'.json'))
        assert receipt['split']==receipt['role']=='val'
        assert selected['raw_ulog_sha256']==receipt['raw_ulog_sha256']
        sp=a.features/'cases'/(key+'.npz');r=read(a.features/'case_records'/(key+'.json'));assert sha(sp)==r['sidecar_sha256']
        with np.load(sp,allow_pickle=False) as z:side={k:z[k] for k in z.files}
        with np.load(a.data/'cases'/(key+'.npz'),allow_pickle=False) as z:base={k:z[k] for k in z.files}
        grid=base['timestamp_us'];joint=side['valid'].all(1);cold=joint.copy();cold[:10]=False
        first0=first_full_window(grid,base['row_input_valid']);first1=first_full_window(grid,cold&base['row_input_valid'])
        row=dict(selected,sidecar_sha256=sha(sp),cold_M0_first_eligible_s=first0,cold_M1_first_eligible_s=first1,
            extra_cold_start_s=first1-first0 if first0 is not None and first1 is not None else None,
            ongoing_history_M1_first_eligible_s=first_full_window(grid,joint&base['row_input_valid']))
        if 'prehistory_timestamp_us' in side:
            # Every selected fixed val case must have batch/arrival parity; no
            # favorable subset is chosen. Timings include normal/fault values.
            duration=stream_check(side,limits);row['residual_compute_ms']=stats(duration/1e6);row['frames_exceeding20ms']=int((duration>20000000).sum())
        else:row['unavailable_reason']=r['whole_case_failure']
        row['injection_reference_measurable']=len(selected['gtd_events'])==1 and float(grid[0])/1e6<=selected['gtd_events'][0]['start_s']<=float(grid[-1])/1e6
        row['no_onset_inference_from_labels']=True;rows.append(row)
        print('PRETRAIN_STREAM_CASE',uid,flush=True)
    # CPU shape benchmark only: not trained classifier performance.
    torch.manual_seed(42);m0=OnlineMSTCA(in_channels=26);m1=OnlineMSTCAKinematicM1();m1.load_common_initial(m0.state_dict());m0.eval();m1.eval();x=torch.zeros(1,26,150);r=torch.zeros(1,4,150)
    benchmark={}
    with torch.inference_mode():
        for name,model in [('M0',m0),('M1',m1)]:
            for _ in range(5):model(x,r) if name=='M1' else model(x)
            durations=[]
            for _ in range(50):
                t=time.perf_counter_ns();model(x,r) if name=='M1' else model(x);durations.append((time.perf_counter_ns()-t)/1e6)
            benchmark[name]=stats(durations)
    t=np.arange(101)*.02;s=np.full(101,.8);events=aggregate_scores(t,s)
    assert events['confirmations'][0]['confirmed_t']==.04
    assert events['finalized_events'][0]['emitted_t']==2.0 and events['finalized_events'][0]['emission_reason']=='end_of_stream_flush'
    result={'status':'PASS_PRETRAINING_CONTINUOUS_REPLAY_BASIC_CHECK','checked_at':datetime.now(timezone.utc).isoformat(),'plan_sha256':sha(a.plan),'cases':rows,'CPU_synthetic_model_inference_ms_not_performance':benchmark,'default_three_point_confirmation_and_final_flush_distinguished':True,'cold_start_definition':'crop-as-service-start, with preceding source history not accessible before launch; ongoing uses established history; no per-window restarts','formal_GPU_training_or_test_Real':False,'limitations':'These are basic source-development timing feasibility checks; final trained replay still required. No device/network/browser guarantee.'}
    a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+chr(10),encoding='utf-8');print(json.dumps({'status':result['status'],'cases':len(rows),'model_shape_benchmark_ms':benchmark},indent=2))


if __name__=='__main__':main()
