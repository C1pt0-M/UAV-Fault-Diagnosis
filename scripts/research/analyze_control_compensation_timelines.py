"""Bounded within-clock temporal diagnostics for the sampled original source cases.

No cross-clock subtraction is performed. TrueState phases use its own fault_state;
ULog phases use same-log rfly_ctrl_lxl event timestamps. Measurements from the
two clocks are not merged into a fabricated common timeline.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ.setdefault('OMP_NUM_THREADS','1')
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd
from pyulog import ULog

TRUE_FIELDS=[*[f'motorRPMs[{i}]' for i in range(1,5)],'Eular[1]','Eular[2]','Eular[3]',
             'AngRate[1]','AngRate[2]','AngRate[3]','AccB[1]','AccB[2]','AccB[3]',
             'VelNED[1]','VelNED[2]','VelNED[3]']
CTRL=[f'control[{i}]' for i in range(4)];OUT=[f'output[{i}]' for i in range(4)]

def robust_stats(x):
 x=pd.to_numeric(pd.Series(x),errors='coerce').to_numpy(float);x=x[np.isfinite(x)]
 return dict(n=len(x),median=float(np.median(x)) if len(x) else None,
  q10=float(np.quantile(x,.1)) if len(x) else None,q90=float(np.quantile(x,.9)) if len(x) else None,
  std=float(np.std(x)) if len(x) else None,zero_fraction=float(np.mean(x==0)) if len(x) else None)

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--results',type=Path,required=True);a=p.parse_args()
 root=a.results;meta=pd.read_csv(root/'case_fault_metadata.csv',keep_default_na=False)
 # Fixed16-case subset: split x hover/velocity x motor/normal x SIL/HIL.
 # One lexical-first pair member per cell; no outcome-based selection.
 meta=meta[meta.flight.isin(['hover','velocity'])&meta.fault_type.isin(['motor','no_fault'])]
 meta=meta.sort_values('case_uid').groupby(['split','flight','fault_type','domain'],sort=True).head(1).reset_index(drop=True)
 if len(meta)!=16:raise ValueError('Expected 16 fixed temporal source cases')
 if (root/'within_clock_timeline_summary.csv').exists():raise FileExistsError('Preserve existing timelines')
 rows=[];series=[]
 for _,task in meta.iterrows():
  uid=task.case_uid;case=Path('D:/RflyMAD')/Path(*uid.split('/'));truepath=case/'TrueData/TrueState_data.xlsx'
  # Recover original ULog path from the frozen source record, and verify it is unchanged.
  record=json.loads((Path('D:/UAV-Data/causal_input_v1/case_records')/(uid.replace('/','__')+'.json')).read_text(encoding='utf-8'))
  raw=Path(record['raw_ulog'])
  from audit_control_compensation_feasibility import sha
  if sha(raw)!=record['raw_ulog_sha256']:raise ValueError('Raw source ULog changed '+uid)
  tr=pd.read_excel(truepath,sheet_name='truedataset');t=pd.to_numeric(tr.trueTime,errors='coerce').to_numpy(float);valid=np.isfinite(t)
  fault=pd.to_numeric(tr.fault_state,errors='coerce').fillna(0).to_numpy()>0
  firsttrue=float(t[np.flatnonzero(fault)[0]]) if fault.any() else None
  timesteps=np.diff(t[valid]);timesteps=timesteps[timesteps>0]
  states=[]
  for phase,mask in [('pre',valid&~fault),('post',valid&fault)]:
   row=dict(case_uid=uid,split=task.split,domain=task.domain,flight=task.flight,fault_type=task.fault_type,
    clock='TrueState_trueTime',phase=phase,first_fault_state_trueTime=firsttrue,
    phase_start=float(t[mask].min()) if mask.any() else None,phase_end=float(t[mask].max()) if mask.any() else None,
    phase_records=int(mask.sum()),timestamp_median_step_s=float(np.median(timesteps)) if len(timesteps) else None)
   for col in TRUE_FIELDS:
    if col in tr:
     stats=robust_stats(tr.loc[mask,col]);row.update({col+'_'+k:v for k,v in stats.items()})
   rows.append(row)
  # ULog clock: Rfly event marker and actuator topics share this publication clock.
  log=ULog(str(raw));topics={}
  for name in ['rfly_ctrl_lxl','actuator_controls_0','actuator_outputs','vehicle_attitude','sensor_combined','vehicle_local_position']:
   found=[d.data for d in log.data_list if d.name==name and d.multi_id==0]
   if found:topics[name]=found[0]
  event=topics.get('rfly_ctrl_lxl');event_time=None
  if task.fault_type=='motor' and event is not None and 'id' in event:
   ix=np.flatnonzero((event['id']>=123450)&(event['id']<=123459))
   if len(ix):event_time=int(event['timestamp'][ix[0]])
  for name,fields in [('actuator_controls_0',CTRL),('actuator_outputs',OUT)]:
   d=topics.get(name)
   if d is None:continue
   ts=np.asarray(d['timestamp'],dtype=np.int64)
   for phase in ['pre','post']:
    if event_time is None:mask=np.ones(len(ts),dtype=bool);anchor='no_same_log_rfly_event'
    else:
     mask=ts<event_time if phase=='pre' else ts>=event_time;anchor='same_log_rfly_ctrl_lxl_id_123450_123459'
    row=dict(case_uid=uid,split=task.split,domain=task.domain,flight=task.flight,fault_type=task.fault_type,
     clock='ULog_publish_timestamp',topic=name,phase=phase,anchor=anchor,event_timestamp_us=event_time,
     phase_records=int(mask.sum()),median_step_us=float(np.median(np.diff(ts[mask]))) if mask.sum()>1 else None)
    for col in fields:
     if col in d:
      stat=robust_stats(d[col][mask]);row.update({col+'_'+k:v for k,v in stat.items()})
    rows.append(row)
  # Event time vs TestInfo elapsed is only a same-ULog-start relative diagnostic;
  # TestInfo's origin is unknown, so never interpret this as physical trigger latency.
  start=int(min(np.min(d['timestamp']) for d in topics.values() if 'timestamp' in d)) if topics else None
  rows.append(dict(case_uid=uid,split=task.split,domain=task.domain,flight=task.flight,fault_type=task.fault_type,
   clock='clock_alignment_audit',phase='event_metadata',testinfo_injection_time_s=float(task.injection_time_literal) if str(task.injection_time_literal).replace('.','',1).isdigit() else None,
   ulog_rfly_event_elapsed_s=(event_time-start)/1e6 if event_time is not None and start is not None else None,
   cross_clock_offset_known=False,note='TestInfo and Excel/ULog origins are not established as a common physical clock.'))
 pd.DataFrame(rows).to_csv(root/'within_clock_timeline_summary.csv',index=False)
 print('WITHIN_CLOCK_SUMMARIES',len(meta),'cases',len(rows),'rows')

if __name__=='__main__':main()
