"""Measure cold-start eligibility from actual ULog start, not each window.

Source-val fixed replay list only; no model or unlabeled-region fault scores.
"""
from pathlib import Path
from datetime import datetime,timezone
import argparse,json,sys
import numpy as np
from pyulog import ULog
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'packages/uav_fault'));sys.path.insert(0,str(ROOT/'scripts/data'));sys.path.insert(0,str(ROOT/'scripts/research'))
from uav_fault.causal_pipeline import GROUPS,TopicTrace,batch_frames
from build_full_causal_input import extract_group
from continuous_kinematic_residuals import Trace,compute_continuous
from replay_m1_continuous import first_full_window
from validate_kinematic_residuals import sha
MARKERS={'baro':['baro_device_id'],'velocity':['z_reset_counter','vz_reset_counter'],'attitude':['quat_reset_counter'],'gyro':[]}


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--plan',type=Path,required=True);ap.add_argument('--data',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    if a.output.exists():raise FileExistsError('Preserve startup evidence')
    p=read(a.plan);policy=read(a.data/'input_contract.json');rows=[]
    for selected in p['cases']:
        uid=selected['case_uid'];r=read(a.data/'case_records'/(uid.replace('/','__')+'.json'))
        assert r['split']==r['role']=='val' and r['raw_ulog_sha256']==selected['raw_ulog_sha256']
        assert sha(r['raw_ulog'])==r['raw_ulog_sha256']
        log=ULog(r['raw_ulog'],message_name_filter_list=sorted({v[0] for v in GROUPS.values()}))
        with np.load(a.data/'cases'/(uid.replace('/','__')+'.npz'),allow_pickle=False) as d:phase=int(d['timestamp_us'][0])%20000;crop_start=int(d['timestamp_us'][0]);crop_end=int(d['timestamp_us'][-1])
        origin=int(log.start_timestamp);first=phase+int(np.ceil((origin-phase)/20000))*20000
        # Scan complete native record for first readiness. No pre/postcrop
        # labels are inferred and no extra training samples are created.
        end=int(log.last_timestamp)
        grid=np.arange(first,end+1,20000,dtype=np.int64);traces={};offsets={};raws={}
        for g,(topic,_,_) in GROUPS.items():
            found=[d.data for d in log.data_list if d.name==topic and d.multi_id==0]
            if len(found)!=1:raise ValueError('Startup missing native topic:'+g)
            raws[g]=found[0];traces[g],offsets[g]=extract_group(found[0],g,grid)
        _,valid,_=batch_frames(traces,grid,policy['columns'],policy['max_age_us']);rt={}
        for g in MARKERS:
            t=traces[g];raw=raws[g];i=offsets[g];n=len(t.available_us);counters={k:raw[k][i:i+n] for k in MARKERS[g] if k in raw};flags=np.ones(n,dtype=bool)
            if g=='velocity' and 'v_z_valid' in raw:flags&=raw['v_z_valid'][i:i+n]==1
            rt[g]=Trace(t.observation_us,t.available_us,t.values[:,2:3] if g=='velocity' else t.values,int(policy['max_age_us'][g]),counters,flags)
        residual=compute_continuous(grid,valid,rt);f0=first_full_window(grid,valid);f1=first_full_window(grid,valid&residual['joint_valid']);first_r=np.flatnonzero(residual['joint_valid'])
        rows.append({'case_uid':uid,'original_log_start_s':origin/1e6,'first20ms_grid_s':first/1e6,'crop_start_s':crop_start/1e6,
            'first_any_valid_N1_N4_s':float(grid[first_r[0]]/1e6) if len(first_r) else None,'M0_first_input_ready_s':f0,'M1_first_input_ready_s':f1,
            'M0_input_warmup_from_log_start_s':f0-origin/1e6 if f0 is not None else None,'M1_input_warmup_from_log_start_s':f1-origin/1e6 if f1 is not None else None,'additional_M1_vs_M0_startup_s':f1-f0 if f0 is not None and f1 is not None else None,
            'timestamps_not_wall_model_inference':True,'unlabeled_pre_crop_data_not_used_for_training_or_scoring':True,'native_log_end_s':end/1e6,'startup_censored_at_full_log_end':f1 is None})
        print('LOG_STARTUP_CHECK',uid,flush=True)
    record={'status':'PASS_FIXED_SOURCE_VAL_TRUE_LOG_STARTUP_ELIGIBILITY','checked_at':datetime.now(timezone.utc).isoformat(),'plan_sha256':sha(a.plan),'rows':rows,'scope':'Native publication-replay first eligible window, not actual live transport or inference timing; originalcrop phase retained. No model/test/Real or labels guessed.'}
    a.output.write_text(json.dumps(record,ensure_ascii=False,indent=2)+chr(10),encoding='utf-8')


if __name__=='__main__':main()
