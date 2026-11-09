"""Same52 dev ULogs: native IMU availability/finite values/clipping, no inference."""
from pathlib import Path
import sys,json
import numpy as np
import pandas as pd
r=Path.cwd();o=r/'artifacts/real_v2_b1_n2_audit_v1';sys.path[:0]=[str(r/'scripts/research'),str(r/'packages/uav_fault')]
import audit_real_v2_b1_n2 as a
p=a.read_json(o/'plan_r2.json');comparisons=pd.read_csv(o/'score_control_comparison_integer_us.csv');events=pd.read_csv(o/'events.csv').set_index('event_id')
allowed={Path(p['records'][uid]['raw_ulog']):{'kind':'raw_ULog_native_IMU_review','case_uid':uid} for uid in p['selected_cases']};a.install_guard(allowed)
rows=[];inventory=[]
for uid in p['selected_cases']:
 log=a.ULog(p['records'][uid]['raw_ulog'],message_name_filter_list=['sensor_combined','vehicle_imu_status','sensor_accel','sensor_gyro'])
 topics=[v for v in log.data_list if v.name in ['sensor_combined','vehicle_imu_status','sensor_accel','sensor_gyro']]
 inventory.append({'case_uid':uid,'topics':[{'name':v.name,'instance':v.multi_id,'fields':list(v.data)} for v in topics]})
 intervals=comparisons[comparisons.case_uid.eq(uid)].drop_duplicates(['event_id','case_uid'])
 for it in intervals.itertuples(index=False):
  for topic in topics:
   d=topic.data;t=d.get('timestamp')
   if t is None:continue
   mask=(t>=int(it.mapped_start_us))&(t<=int(it.mapped_end_us));n=int(mask.sum())
   row={'event_id':it.event_id,'case_uid':uid,'role':it.role,'topic':topic.name,'instance':topic.multi_id,'samples':n,
        'timestamp_reversals_entire_topic':int((np.diff(t.astype(np.int64))<0).sum()),'native_interval_uses_publication_clock':True}
   if topic.name=='sensor_combined':
    values=np.column_stack([d[k] for k in ['gyro_rad[0]','gyro_rad[1]','gyro_rad[2]','accelerometer_m_s2[0]','accelerometer_m_s2[1]','accelerometer_m_s2[2]']])[mask]
    row['nonfinite_gyro_accel_values']=int((~np.isfinite(values)).sum());row['gyro_absmax']=float(np.max(np.abs(values[:,:3]))) if n else None
    row['accel_absmax']=float(np.max(np.abs(values[:,3:]))) if n else None
   for key,value in d.items():
    if 'clip'in key.lower() or 'error_count'in key.lower():
     v=np.asarray(value)[mask];row[key+'_nonzero_records']=int(np.count_nonzero(v));row[key+'_min']=float(np.min(v)) if len(v) else None;row[key+'_max']=float(np.max(v)) if len(v) else None
     row[key+'_last_minus_first']=float(v[-1])-float(v[0]) if len(v) else None
   rows.append(row)
 del log
 print('NATIVE_REVIEW',uid,flush=True)
a.csv(o/'native_IMU_interval_review.csv',rows);a.save(o/'native_IMU_topic_inventory.json',inventory)
