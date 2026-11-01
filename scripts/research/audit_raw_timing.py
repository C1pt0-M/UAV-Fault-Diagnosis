"""Read-only Real raw control-onset audit plus stratified IMU interpolation checks."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'docs/research/2026-10-08/data_feasibility'
MANIFEST = Path('D:/UAV-Data/splits/rflymad_case_split_v2/rflymad_case_level_split_manifest.csv')
IDS = [123450,123452,123453,123454,123455,123456]

def audit(row):
    r = {k:row[k] for k in ('case_uid','flight_status','level3_label','source_disposition','source_path')}
    try:
        source = Path(row['source_path'])
        controls = list(source.glob('*/_slash_mavros_slash_rfly_ctrl_lxl_slash_send_data.csv'))
        if len(controls) != 1:
            raise ValueError(f'Expected one ROS control topic, found {len(controls)}')
        raw = pd.read_csv(controls[0],usecols=['rosbagTimestamp','id'])
        p = pd.read_csv(row['final_output_path'])
        # Work in nanoseconds relative to an integer epoch to avoid 1e18 float cancellation.
        origin = int(raw.rosbagTimestamp.iloc[0])
        rt = (raw.rosbagTimestamp.astype('int64').to_numpy()-origin)/1e9
        pt = (p.rosbagTimestamp.to_numpy()-origin)/1e9
        active = raw.id.isin(IDS).to_numpy()
        pmask = p[row['window_label_source']].isin(IDS).to_numpy()
        raw_onsets = np.flatnonzero(active & ~np.r_[False,active[:-1]])
        pro_onsets = np.flatnonzero(pmask & ~np.r_[False,pmask[:-1]])
        retained = raw_onsets[(rt[raw_onsets]>=pt[0]) & (rt[raw_onsets]<=pt[-1])]
        # Margin permits sub-grid boundary displacement; do not force pairing on count mismatch.
        near = raw_onsets[(rt[raw_onsets]>=pt[0]-.1) & (rt[raw_onsets]<=pt[-1]+.1)]
        r.update(raw_control_path=str(controls[0]), raw_control_rows=len(raw),
            raw_control_nonincreasing=int((np.diff(rt)<=0).sum()),
            raw_control_median_step_ms=float(np.median(np.diff(rt))*1000),
            raw_control_max_step_ms=float(np.max(np.diff(rt))*1000),
            raw_fault_events_total=len(raw_onsets),raw_onsets_in_retained_span=len(retained),
            processed_fault_events=len(pro_onsets),
            raw_has_fault_in_retained_span=bool(active[(rt>=pt[0]) & (rt<=pt[-1])].any()))
        if len(pro_onsets)==len(near) and len(near):
            delta = (pt[pro_onsets]-rt[near])*1000
            r.update(onset_pairing='ordered_equal_count',onset_deltas_ms=delta.tolist(),
                     first_onset_delta_ms=float(delta[0]))
        else:
            r.update(onset_pairing='no_events' if not len(pro_onsets) and not len(near) else 'count_mismatch')

        phases=list(source.glob('*/_slash_mavros_slash_offb_flight_mode_slash_send_data.csv'))
        if len(phases)==1:
            try:
                phase=pd.read_csv(phases[0])
                ph_t=(phase.rosbagTimestamp.astype('int64').to_numpy()-origin)/1e9
                if 'flight_status' in phase:
                    r['raw_phase_values_in_span']=phase.loc[(ph_t>=pt[0])&(ph_t<=pt[-1]),'flight_status'].unique().tolist()
                r['phase_status']='ok'
            except Exception as e:
                r['phase_status']='error'
                r['phase_error']=f'{type(e).__name__}: {e}'
        if row['imu_spot']:
            imu_paths=list(source.glob('*/*_sensor_combined_0.csv'))
            if len(imu_paths)!=1:
                raise ValueError(f'Expected one IMU topic, found {len(imu_paths)}')
            imu=pd.read_csv(imu_paths[0],usecols=['timestamp','gyro_rad[0]'])
            it=imu.timestamp.to_numpy(dtype=float)
            y=imu['gyro_rad[0]'].to_numpy(dtype=float)
            grid=p.Timestamp.to_numpy(dtype=float)
            good=(grid>=it[0])&(grid<=it[-1])
            grid=grid[good]
            observed=p['_sensor_combined_0_gyro_rad[0]'].to_numpy(dtype=float)[good]
            linear=np.interp(grid,it,y)
            past=np.searchsorted(it,grid,side='right')-1
            future=np.searchsorted(it,grid,side='left')
            future_gap=(it[future]-grid)/1000
            in_span=it[(it>=grid[0])&(it<=grid[-1])]
            r.update(imu_raw_path=str(imu_paths[0]),imu_nonincreasing=int((np.diff(it)<=0).sum()),
                imu_raw_median_step_ms=float(np.median(np.diff(in_span))/1000),
                imu_raw_max_step_ms=float(np.max(np.diff(in_span))/1000),
                imu_checked_grid_rows=len(grid),imu_out_of_range_rows=int((~good).sum()),
                imu_linear_match_fraction=float(np.isclose(observed,linear,atol=1e-7,rtol=1e-6).mean()),
                imu_previous_match_fraction=float(np.isclose(observed,y[past],atol=1e-7,rtol=1e-6).mean()),
                imu_future_bracket_fraction=float((future_gap>0).mean()),
                imu_future_bracket_p95_ms=float(np.quantile(future_gap,.95)),
                imu_future_bracket_max_ms=float(future_gap.max()))
        r['status']='ok'
    except Exception as e:
        r.update(status='error',error=f'{type(e).__name__}: {e}')
    return r

def main():
    d=pd.read_csv(MANIFEST,keep_default_na=False)
    d=d[d.domain=='REAL'].copy()
    selected=set(d.groupby(['flight_status','level3_label'],sort=True).head(1).case_uid)
    d['imu_spot']=d.case_uid.isin(selected)
    with ThreadPoolExecutor(max_workers=4) as pool:
        records=list(pool.map(audit,d.to_dict('records')))
    (OUT/'raw_real_timing_audit.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8')
    pd.DataFrame(records).to_csv(OUT/'raw_real_timing_audit.csv',index=False,encoding='utf-8-sig')
    print('Real controls audited:',len(records),'errors:',sum(r['status']!='ok' for r in records),'IMU spots:',len(selected))
    for r in records:
        if r['status']!='ok': print(r['case_uid'],r['error'])
    deltas=[v for r in records for v in r.get('onset_deltas_ms',[])]
    print('Paired onsets:',len(deltas),'delta_ms quantiles:',np.quantile(deltas,[0,.25,.5,.75,1]).tolist())
    print('Pair statuses:',pd.Series([r.get('onset_pairing') for r in records]).value_counts().to_dict())
    print('IMU linear match minimum:',min(r['imu_linear_match_fraction'] for r in records if 'imu_linear_match_fraction' in r))

if __name__=='__main__': main()
