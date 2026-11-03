"""Read-only controller-compensation feasibility audit on frozen source cases.

Reads only selected SIL/HIL Train/Val original ULogs, TrueState/UAVState and
TLogs. No Test/Real, merged cache feature values, training or inference.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ.setdefault('OMP_NUM_THREADS','1')
from pathlib import Path
from datetime import datetime,timezone
import argparse,csv,hashlib,json
import numpy as np
import pandas as pd
from pyulog import ULog
ROOT=Path(__file__).resolve().parents[2]
FAULTS=['motor','no_fault']
FLIGHTS=['hover','velocity','acce','dece','waypoint','circling']
MOTORS=[f'motorRPMs[{i}]' for i in range(1,5)]
OUTS=[f'output[{i}]' for i in range(4)]
CTRLS=[f'control[{i}]' for i in range(4)]
TOPICS=['actuator_controls_0','actuator_outputs','rfly_ctrl_lxl','test_motor','multirotor_motor_limits','vehicle_attitude','sensor_combined','vehicle_local_position']

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,d):Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def testinfo(p):
 with Path(p).open(encoding='utf-8-sig',newline='') as f:return {x[0]:x[1] for x in csv.reader(f) if len(x)>=2}
def hz_stats(t):
 t=np.asarray(t,dtype=np.int64);d=np.diff(t);positive=d[d>0]
 if not len(positive):return dict(n=len(t),median_step_us=None,p95_step_us=None,approx_hz=None,max_step_us=None,gap_gt2x=0,nonmonotone=int((d<0).sum()),duplicates=int((d==0).sum()))
 med=float(np.median(positive))
 return dict(n=len(t),median_step_us=med,p95_step_us=float(np.quantile(positive,.95)),approx_hz=1e6/med,max_step_us=int(positive.max()),gap_gt2x=int((positive>2*med).sum()),nonmonotone=int((d<0).sum()),duplicates=int((d==0).sum()))
def field_stats(values):
 x=np.asarray(values,dtype=float);finite=np.isfinite(x);v=x[finite]
 return dict(records=len(x),finite=int(finite.sum()),missing_nonfinite=int((~finite).sum()),nonfinite_fraction=float((~finite).mean()) if len(x) else None,
  zero=int(np.sum(v==0)),zero_fraction=float(np.mean(v==0)) if len(v) else None,
  min=float(v.min()) if len(v) else None,median=float(np.median(v)) if len(v) else None,max=float(v.max()) if len(v) else None)
def topic_fields(log,topic):
 ds=[d.data for d in log.data_list if d.name==topic]
 return ds
def summary_case(task,root,outdir):
 uid=task['case_uid'];p=Path(task['raw_ulog']);info=testinfo(p.parent.parent/'TestInfo.csv')
 if task['split'] not in ['train','val'] or task['domain'] not in ['SIL','HIL']:raise ValueError('Only frozen source Train/Val SIL/HIL')
 if sha(p)!=task['raw_ulog_sha256']:raise ValueError('Raw ULog SHA mismatch '+uid)
 truepath=p.parent.parent/'TrueData'/'TrueState_data.xlsx';uavpath=p.parent.parent/'TrueData'/'UAVState_data.xlsx'
 if not truepath.is_file() or not uavpath.is_file():raise ValueError('Missing TrueState/UAVState '+uid)
 if sha(truepath)!=task['true_sha256'] or sha(uavpath)!=task['uav_sha256']:raise ValueError('State file SHA mismatch '+uid)
 log=ULog(str(p),message_name_filter_list=TOPICS)
 rows=[];topic_data={}
 for topic in TOPICS:
  ds=topic_fields(log,topic);topic_data[topic]=ds
  for d in ds:
   ts=d.get('timestamp',np.array([],dtype=np.int64))
   row=dict(case_uid=uid,split=task['split'],domain=task['domain'],flight=task['flight'],fault_type=task['fault_type'],topic=topic,multi_id=d.get('multi_id',None),**hz_stats(ts))
   if topic=='actuator_controls_0':
    row['channel_count']=4;row['channel_complete']=all(k in d for k in CTRLS)
    row['sample_timestamp_available']='timestamp_sample' in d
    row['control_fields_present']=','.join(k for k in CTRLS if k in d)
    if row['channel_complete']:
     vals=np.column_stack([d[k] for k in CTRLS]);row['channels_finite_all']=bool(np.isfinite(vals).all());row['channels_all_zero_fraction']=float(np.mean(vals==0))
   elif topic=='actuator_outputs':
    no=d.get('noutputs',np.array([],dtype=int));n=int(np.median(no)) if len(no) else 0
    row['noutputs_median']=n;row['first4_output_fields_present']=all(k in d for k in OUTS)
    row['first4_finite_all']=bool(np.isfinite(np.column_stack([d[k] for k in OUTS])).all()) if row['first4_output_fields_present'] else False
    row['noutputs_values']=';'.join(map(str,np.unique(no)))
   elif topic=='rfly_ctrl_lxl':
    row['custom_id_values']=';'.join(map(str,np.unique(d['id'])[:20])) if 'id' in d else ''
    row['custom_mode_values']=';'.join(map(str,np.unique(d['mode'])[:20])) if 'mode' in d else ''
   rows.append(row)
 # Exact schemas and finite/zero audit for TrueState and UAVState, separately from ULog.
 tr=pd.read_excel(truepath,sheet_name='truedataset');uv=pd.read_excel(uavpath,sheet_name='extimatedataset')
 for kind,frame,timecol in [('TrueState',tr,'trueTime'),('UAVState',uv,'uavTime')]:
  for col in frame.columns:
   if any(x in col.lower() for x in ['motor','rpm','fault_state','cmd','time']):
    row=dict(case_uid=uid,split=task['split'],domain=task['domain'],flight=task['flight'],fault_type=task['fault_type'],topic=kind,field=col,**field_stats(pd.to_numeric(frame[col],errors='coerce')))
    intervals=np.diff(pd.to_numeric(frame[timecol],errors='coerce').dropna().to_numpy(dtype=float));intervals=intervals[intervals>0]
    seconds_per_unit=1.0 if timecol=='trueTime' else 0.001
    row['timestamp_unit']='s' if timecol=='trueTime' else 'ms'
    row['timestamp_median_step_native']=float(np.median(intervals)) if len(intervals) else None
    row['timestamp_hz']=float(1/(np.median(intervals)*seconds_per_unit)) if len(intervals) else None
    rows.append(row)
 # TestInfo is metadata, not a sample-aligned physical clock.
 metadata=dict(case_uid=uid,split=task['split'],domain=task['domain'],flight=task['flight'],fault_type=task['fault_type'],
  fault_id_literal=info.get('Fault ID',''),fault_parameter_literal=info.get('Fault Parameter',''),control_sequence_literal=info.get('Control Sequence',''),
  injection_time_literal=info.get('Fault injection time',''),test_end_literal=info.get('Test end time',''),declared_frequency_literal=info.get('Data collected Frequency',''))
 if task.get('trace_case'):
  trace_case(task,p,log,tr,uv,outdir)
 return rows,metadata

def trace_case(task,p,log,tr,uv,outdir):
 uid=task['case_uid'];key=uid.replace('/','__');dest=outdir/'traces'/key
 if dest.exists():raise FileExistsError(dest)
 dest.mkdir(parents=True)
 # Native ULog base clock and same-file topics. No cross-clock subtraction.
 base=[d.data for d in log.data_list if d.name=='sensor_combined' and d.multi_id==0]
 if not base:return
 grid=np.asarray(base[0]['timestamp'],dtype=np.int64)
 topics={}
 for name in TOPICS:
  ds=topic_fields(log,name)
  if len(ds)==1:topics[name]=ds[0]
 fields=[]
 # TrueState time is common within workbook and defines the platform trace only.
 tt=pd.to_numeric(tr['trueTime'],errors='coerce').to_numpy(dtype=float)
 uavt=pd.to_numeric(uv['uavTime'],errors='coerce').to_numpy(dtype=float)*.001
 chosen=['trueTime',*MOTORS,'fault_state','cmd','Eular[1]','Eular[2]','Eular[3]','AngRate[1]','AngRate[2]','AngRate[3]','AccB[1]','AccB[2]','AccB[3]','VelNED[1]','VelNED[2]','VelNED[3]']
 for col in chosen:
  if col in tr:fields.append(('TrueState',col,hz_stats(np.rint(tt*1e6).astype(np.int64))['approx_hz']))
  if col in uv:fields.append(('UAVState',col,hz_stats(np.rint(uavt*1e6).astype(np.int64))['approx_hz']))
 # Match Excel within its own timestamp axis only; keep both clock columns and no forced alignment to ULog.
 out=pd.DataFrame({'ulog_timestamp_us':grid})
 for name in ['actuator_controls_0','actuator_outputs','rfly_ctrl_lxl','vehicle_attitude','sensor_combined','vehicle_local_position']:
  ds=[d.data for d in log.data_list if d.name==name and d.multi_id==0]
  if not ds:continue
  d=ds[0];ts=d['timestamp'];
  columns=CTRLS if name=='actuator_controls_0' else OUTS if name=='actuator_outputs' else [f'controls[{i}]' for i in range(8)] if name=='rfly_ctrl_lxl' else []
  for c in columns:
   if c in d:
    ix=np.searchsorted(ts,grid,side='right')-1;ok=ix>=0;v=np.full(len(grid),np.nan);v[ok]=d[c][ix[ok]];out[name+'_'+c.replace('[','_').replace(']','')]=v
 for kind,frame,timecol,clock in [('TrueState',tr,'trueTime','trueTime_s'),('UAVState',uv,'uavTime','uavTime_ms')]:
  times=pd.to_numeric(frame[timecol],errors='coerce').to_numpy(dtype=float)
  out[clock]=np.nan
  for col in chosen:
   if col in frame:out[kind+'_'+col.replace('[','_').replace(']','')]=np.nan
 # Save native-source rows separately because cross-clock mapping has not been proven.
 tr[chosen].to_csv(dest/'TrueState_native_clock.csv',index=False)
 uv[['uavTime',*[c for c in chosen if c in uv]]].to_csv(dest/'UAVState_native_clock.csv',index=False)
 out.to_csv(dest/'ULog_same_clock.csv',index=False)

def main():
 ap=argparse.ArgumentParser(description=__doc__)
 ap.add_argument('--data',type=Path,default=Path('D:/UAV-Data/causal_input_v1'))
 ap.add_argument('--output',type=Path,required=True)
 args=ap.parse_args()
 if args.output.exists():raise FileExistsError('Preserve previous audit')
 args.output.mkdir(parents=True)
 # Case roster comes from frozen source ledgers. Sample one frozen SIL/HIL
 # association group per split/flight/fault stratum; no outcome-based selection.
 rosters=[];all_ledgers={}
 for split in ['train','val']:
  ledger=pd.read_csv(args.data/'ledgers'/(split+'_included_windows.csv'),keep_default_na=False,dtype={'association_group':str})
  cases=ledger.drop_duplicates('case_uid');all_ledgers[split]=cases
  for _,row in cases.iterrows():
   parts=row.case_uid.split('/');domain,flight,fault=parts[:3]
   if fault not in ['motor','no_fault']:continue
   case_dir=Path('D:/RflyMAD')/Path(*parts);record=read(args.data/'case_records'/(row.case_uid.replace('/','__')+'.json'))
   if record['split']!=split or record['role']!=split:raise ValueError('Frozen role mismatch')
   state=case_dir/'TrueData'/'TrueState_data.xlsx';uav=case_dir/'TrueData'/'UAVState_data.xlsx';raw=Path(record['raw_ulog'])
   if not state.is_file() or not uav.is_file() or not raw.is_file():raise FileNotFoundError(row.case_uid)
   rosters.append(dict(case_uid=row.case_uid,association_group=str(row.association_group),domain=domain,flight=flight,fault_type=fault,split=split,raw_ulog=str(raw),raw_ulog_sha256=record['raw_ulog_sha256'],true_sha256=sha(state),uav_sha256=sha(uav)))
 chosen=[]
 for split in ['train','val']:
  roster=[t for t in rosters if t['split']==split]
  for flight in FLIGHTS:
   for fault in FAULTS:
    layer=[t for t in roster if t['flight']==flight and t['fault_type']==fault]
    sil={t['association_group']:t for t in layer if t['domain']=='SIL'};hil={t['association_group']:t for t in layer if t['domain']=='HIL'}
    common=sorted(set(sil)&set(hil))
    if common:
     key=common[0];chosen.extend([sil[key],hil[key]])
    else:
     for domain in ['SIL','HIL']:
      options=sorted((t for t in layer if t['domain']==domain),key=lambda x:x['case_uid'])
      if options:chosen.append(options[0])
 if len(chosen)!=48:raise ValueError(f'Expected fixed one-pair-per-cell 48 source cases, got {len(chosen)}')
 tasks=chosen
 allrows=[];metas=[]
 for task in tasks:
  rows,meta=summary_case(task,args.data,args.output);allrows.extend(rows);metas.append(meta)
 pd.DataFrame(allrows).to_csv(args.output/'raw_signal_inventory.csv',index=False)
 pd.DataFrame(metas).to_csv(args.output/'case_fault_metadata.csv',index=False)
 coverage=[]
 for split,cases in all_ledgers.items():
  for (domain,flight,fault),g in cases[cases.case_uid.str.contains('/motor/|/no_fault/')].groupby(['domain',cases.case_uid.str.split('/').str[1],cases.case_uid.str.split('/').str[2]]):
   picked=[t for t in tasks if t['split']==split and t['domain']==domain and t['flight']==flight and t['fault_type']==fault]
   coverage.append(dict(split=split,domain=domain,flight=flight,fault=fault,frozen_cases=len(g),association_groups=g.association_group.nunique(),raw_cases_audited=len(picked),audit_fraction=len(picked)/len(g)))
 pd.DataFrame(coverage).to_csv(args.output/'frozen_motor_normal_coverage.csv',index=False)
 print(json.dumps({'frozen_motor_normal_cases':sum(len(c[c.case_uid.str.contains('/motor/|/no_fault/')]) for c in all_ledgers.values()),'raw_cases_audited':len(tasks),'records':len(allrows),'output':str(args.output)},indent=2))

if __name__=='__main__':main()
