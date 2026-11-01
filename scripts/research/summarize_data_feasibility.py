"""Aggregate audit evidence without reading or altering source data."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'docs/research/2026-10-08/data_feasibility'
d=pd.read_csv(OUT/'full_case_audit.csv')
raw=json.loads((OUT/'raw_real_timing_audit.json').read_text(encoding='utf-8'))
records=[json.loads(s) for s in (OUT/'full_case_audit.jsonl').read_text(encoding='utf-8').splitlines()]
domains=[]
for dom,g in d.groupby('domain'):
    normal=g[g.level1_label=='normal']
    faults=g[g.level1_label=='fault']
    row=dict(domain=dom,cases=len(g),fault_cases=len(faults),normal_cases=len(normal),
        duration_hours=float(g.duration_s.sum()/3600),duration_median_s=float(g.duration_s.median()),
        duration_min_s=float(g.duration_s.min()),duration_max_s=float(g.duration_s.max()),
        normal_case_hours=float(normal.duration_s.sum()/3600),
        normal_case_median_s=float(normal.duration_s.median()),normal_case_max_s=float(normal.duration_s.max()),
        prefault_median_s=float(faults.first_fault_s.median()),prefault_min_s=float(faults.first_fault_s.min()),
        prefault_max_s=float(faults.first_fault_s.max()),
        multiple_active_interval_cases=int((g.fault_event_count>1).sum()),
        label_inactive_hours=float(g.normal_exposure_s.sum()/3600))
    row['normal_runs_at_least_s']={s:int((g.normal_longest_run_s>=s-1e-6).sum()) for s in (30,60,120,300)}
    row['issues']={k:int(g[k].sum()) for k in ('missing_feature_count','nonfinite_feature_cells','timestamp_nonfinite',
        'nonincreasing_steps','steps_off_50hz','label_nonfinite','label_noninteger','unknown_label_rows')}
    row['mismatches']={k:int((~g[k]).sum()) for k in ('manifest_hash_matches','manifest_rows_match','manifest_columns_match','case_label_consistent')}
    domains.append(row)
    print(json.dumps(row,ensure_ascii=False))

hil_log=pd.read_csv('D:/UAV-Data/processing_logs/hil-50hz-20260813-110229/case_results.csv')
h=d[d.domain=='HIL'].copy()
h['output_filename']=h.final_output_path.map(lambda x:Path(x).name)
joined=h.merge(hil_log,on='output_filename')
imu=[r for r in raw if 'imu_linear_match_fraction' in r]
deltas=np.array([v for r in raw for v in r.get('onset_deltas_ms',[])])
id_type={123450:'motor',123452:'accelerometer',123453:'gyroscope',123454:'barometer',123455:'magnetometer',123456:'GPS'}
type_mismatch=[]
for r in records:
    if r['domain']=='REAL' and r['level1_label']=='fault':
        observed={id_type[v] for v in r['observed_label_values'] if v in id_type}
        if observed!={r['level3_label']}: type_mismatch.append(r['case_uid'])
summary=dict(domains=domains,total_hours=float(d.duration_s.sum()/3600),
    byte_identical_duplicates=int(d.sha256.duplicated().sum()),
    all_source_directories_exist=bool(d.source_directory_exists.all()),
    hil_current_hash_matches_aug13_log=int((joined.sha256==joined.output_sha256).sum()),
    real_active_id_subtype_mismatches=type_mismatch,
    real_raw_controls_read_ok=sum(r['status']=='ok' for r in raw),
    real_raw_phase_parse_errors=sum(r.get('phase_status')=='error' for r in raw),
    real_raw_imu_spots=len(imu),imu_linear_match_min=min(r['imu_linear_match_fraction'] for r in imu),
    imu_lookahead_max_ms=max(r['imu_future_bracket_max_ms'] for r in imu),
    imu_per_case_lookahead_p95_ms_range=[min(r['imu_future_bracket_p95_ms'] for r in imu),max(r['imu_future_bracket_p95_ms'] for r in imu)],
    paired_onsets=len(deltas),onset_delta_ms_min_median_max=np.quantile(deltas,[0,.5,1]).tolist(),
    processed_onsets_earlier_than_raw=int((deltas<0).sum()),
    raw_processed_event_count_mismatch_cases=[r['case_uid'] for r in raw if r['onset_pairing']=='count_mismatch'],
    normal_only_cases_with_raw_fault_outside_retained_span=sum(r['source_disposition']=='retain_normal_only' and r['raw_fault_events_total']>0 and not r['raw_has_fault_in_retained_span'] for r in raw))
(OUT/'feasibility_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
for group,name in [(['domain','level3_label'],'class_counts'),(['domain','flight_status'],'mode_counts'),(['domain','split'],'split_counts'),(['domain','source_disposition'],'disposition_counts')]:
    d.groupby(group).size().rename('case_count').reset_index().to_csv(OUT/f'{name}.csv',index=False,encoding='utf-8-sig')
print(json.dumps({k:v for k,v in summary.items() if k!='domains'},ensure_ascii=False,indent=2))
