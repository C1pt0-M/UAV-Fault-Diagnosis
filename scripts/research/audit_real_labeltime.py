"""Read-only Real role-set label timing audit, not a training or relabeling tool.

Compare processed ROS command IDs to timestamp-held original exported ROS IDs
and to original ULog IDs. No GTD exists for Real. Do not silently replace the
ROS-derived convention with ULog or certify exported CSVs as bag ground truth.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import numpy as np
import pandas as pd
from pyulog import ULog

from audit_rflymad_raw_sources import digest, active_ids

ROOT = Path(__file__).resolve().parents[2]
ROLES = ROOT / 'artifacts/normal_adaptation_audit_20261029/real_case_roles_v1.csv'
RAW_AUDIT = ROOT / 'artifacts/raw_source_audit_20261030'
LABEL = 'rfly_ctrl_lxl_slash_send_data_id'


def hold_integer(times, states, query):
    t, q = np.asarray(times), np.asarray(query)
    y = np.asarray(states)
    if t.ndim != 1 or not len(t) or y.shape != t.shape or q.ndim != 1:
        raise ValueError('Empty or mismatched timeline')
    if not np.issubdtype(t.dtype, np.integer) or not np.issubdtype(q.dtype, np.integer):
        raise ValueError('Integer timestamps required; avoid Unix-nanosecond float loss')
    if (np.diff(t) < 0).any() or (np.diff(q) < 0).any():
        raise ValueError('Clock reversal; no silent sorting')
    index = np.searchsorted(t, q, side='right') - 1
    covered = (index >= 0) & (q <= t[-1])
    reference = np.full(len(q), -1, dtype=np.int8)
    reference[covered] = y[index[covered]].astype(np.int8)
    return reference, covered


def first_time(t, states):
    hits = np.flatnonzero(states)
    return int(t[hits[0]]) if len(hits) else None


def read_export(path, fields):
    with path.open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    if not rows or any(k not in rows[0] for k in fields):
        raise ValueError('Missing export rows/fields: ' + str(path))
    # Python int from decimal strings, never float parsing nanosecond fields.
    return {key: np.asarray([int(r[key]) for r in rows], dtype=np.int64) for key in fields}


def selected_ends(n, mode):
    if mode == 'cap64':
        starts = np.arange(0, n - 150 + 1, 5, dtype=int)
        picks = np.unique(np.linspace(0, len(starts)-1, 64).round().astype(int)) if len(starts)>64 else np.arange(len(starts))
        return starts[picks]+149
    return np.arange(0, n - 150 + 1, 10, dtype=int)+149


def audit_case(task):
    role, inventory, oldraw = task
    result = dict(role)
    result.update(status='review_required', issues=[], warnings=[])
    try:
        path = Path(role['final_output_path'])
        if digest(path) != role['output_sha256']:
            raise ValueError('Frozen Real CSV identity changed')
        raw = Path(inventory['path'])
        syncs=list(raw.rglob('*timesync_status.csv'))
        if len(syncs)!=1:raise ValueError('Expected one ROS timesync export')
        commands=list(syncs[0].parent.glob('*rfly_ctrl_lxl*send_data.csv'))
        if len(commands)!=1:raise ValueError('Expected one ROS command export')
        logs=inventory['sources']['ulog']
        if len(logs)!=1:raise ValueError('Expected one original ULog')
        logpath=Path(logs[0]); rawhash=digest(logpath)
        if rawhash!=oldraw['ulog_sha256']:raise ValueError('Original ULog identity changed')
        used=[path,syncs[0],commands[0],logpath]
        before={str(p):(p.stat().st_size,p.stat().st_mtime_ns) for p in used}
        sync=read_export(syncs[0],['rosbagTimestamp','remote_timestamp_ns'])
        sent=read_export(commands[0],['rosbagTimestamp','id'])
        if (np.diff(sync['rosbagTimestamp'])<0).any() or (np.diff(sync['remote_timestamp_ns'])<0).any():
            result['warnings'].append('timesync_series_nonmonotonic')
        ros0=int(sync['rosbagTimestamp'][0]); px0=int(sync['remote_timestamp_ns'][0])//1000
        frame=pd.read_csv(path,usecols=['Timestamp','rosbagTimestamp',LABEL])
        v=frame.Timestamp.to_numpy(dtype=float)
        if not np.isfinite(v).all() or not np.equal(v,np.rint(v)).all():raise ValueError('Noninteger processed PX4 timestamps')
        q=v.astype(np.int64)
        if len(q)!=int(role['row_count']) or len(q)<150 or not np.all(np.diff(q)==20000):raise ValueError('Unexpected fixed crop grid')
        oldids=frame[LABEL].to_numpy(dtype=float)
        noninteger=not np.equal(oldids,np.rint(oldids)).all()
        if not np.isfinite(oldids).all():raise ValueError('Nonfinite processed command IDs')
        if noninteger:result['warnings'].append('noninteger_processed_command_ids')
        # Historical window generator truncates numeric IDs to int. Retain this
        # convention for OLD labels and separately disclose invalid numeric IDs.
        old=active_ids(oldids.astype(np.int64))
        ros_query=ros0+(q-px0)*1000
        ros_active=active_ids(sent['id'])
        rr,rc=hold_integer(sent['rosbagTimestamp'],ros_active,ros_query)
        log=ULog(str(logpath),message_name_filter_list=['rfly_ctrl_lxl'])
        topics=[d.data for d in log.data_list if d.name=='rfly_ctrl_lxl' and d.multi_id==0]
        if len(topics)!=1:raise ValueError('Missing original ULog injection topic')
        data=topics[0]; ut=data['timestamp'].astype(np.int64); ua=active_ids(data['id'])
        ur,uc=hold_integer(ut,ua,q)
        bo=first_time(q,old); ro=first_time(sent['rosbagTimestamp'],ros_active); uo=first_time(ut,ua)
        ros_px=(px0+(ro-ros0)/1000) if ro is not None else None
        bridge_error=np.abs(frame.rosbagTimestamp.to_numpy(dtype=float)-ros_query.astype(float))
        if not np.isfinite(bridge_error).all():raise ValueError('Nonfinite ROS/PX4 bridge columns')
        if bridge_error.max()>1024:result['warnings'].append('processed_clock_bridge_disagrees_with_original_first_sync')
        # Compare every sync record to first-anchor affine mapping. Diagnostic
        # residual includes transport/clock-estimation effects, not just drift.
        residual=sync['remote_timestamp_ns']-(px0*1000+(sync['rosbagTimestamp']-ros0))
        result.update(raw_ulog_sha256=rawhash,ros_command_export_sha256=digest(commands[0]),
            ros_timesync_export_sha256=digest(syncs[0]),ros_command_export=str(commands[0]),ros_timesync_export=str(syncs[0]),
            crop_start_us=int(q[0]),crop_end_us=int(q[-1]),rows=len(q),
            original_ros_ns=ros0,original_px4_us=px0,bridge_max_float_rounding_error_ns=float(bridge_error.max()),
            timesync_residual_ns_min=int(residual.min()),timesync_residual_ns_max=int(residual.max()),
            ros_reference_covered_rows=int(rc.sum()),ulog_reference_covered_rows=int(uc.sum()),
            old_vs_ros_row_disagreements=int(((old!=rr)&rc).sum()),
            ros_vs_ulog_row_disagreements=int(((rr!=ur)&rc&uc).sum()),
            old_vs_ulog_row_disagreements=int(((old!=ur)&uc).sum()),
            processed_active_rows=int(old.sum()),ros_active_rows_in_crop=int((rr==1).sum()),ulog_active_rows_in_crop=int((ur==1).sum()),
            old_onset_us=bo,ros_sent_onset_mapped_px4_us=ros_px,ulog_received_onset_us=uo,
            old_minus_ros_onset_s=(bo-ros_px)/1e6 if bo is not None and ros_px is not None else None,
            ros_minus_ulog_onset_s=(ros_px-uo)/1e6 if ros_px is not None and uo is not None else None,
            ros_fault_ids=sorted(set(int(v) for v in sent['id'][ros_active])),
            ulog_fault_ids=sorted(set(int(v) for v in data['id'][ua])),
            sampled={})
        for mode in ['cap64','stride10']:
            ends=selected_ends(len(q),mode);full=rc[ends]&uc[ends]
            change=(old[ends]!=rr[ends])&rc[ends]
            result['sampled'][mode]={'windows':len(ends),'ros_covered':int(rc[ends].sum()),'ulog_covered':int(uc[ends].sum()),
                'old_vs_ros_label_differences':int(change.sum()),
                'old_vs_ros_differences_ulog_agrees':int((change&full&(rr[ends]==ur[ends])).sum()),
                'ros_vs_ulog_disagreement_endpoints':int((full&(rr[ends]!=ur[ends])).sum()),
                'old_vs_ulog_label_differences':int(((old[ends]!=ur[ends])&uc[ends]).sum())}
        if not rc.all():result['warnings'].append('ROS_export_does_not_cover_entire_fixed_crop')
        if not uc.all():result['warnings'].append('ULog_topic_does_not_cover_entire_fixed_crop')
        if role['level2_label']=='normal' and ((rr[rc]==1).any() or (ur[uc]==1).any()):
            result['warnings'].append('normal_role_contains_recorded_injection_in_retained_crop')
        if result['ros_fault_ids']!=result['ulog_fault_ids']:result['warnings'].append('ROS_ULog_observed_fault_ID_sets_differ')
        if (bool(old.any()) != (role['role']=='test_fault')):result['warnings'].append('old_role_label_presence_mismatch')
        if any(before[str(p)]!=(p.stat().st_size,p.stat().st_mtime_ns) for p in used):raise ValueError('Input changed during read')
        result['status']='diagnostic_complete'
    except Exception as exc:
        result['issues'].append(type(exc).__name__+': '+str(exc))
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    out=args.output.resolve()
    for root in [Path('D:/RflyMAD').resolve(),Path('D:/UAV-Data').resolve()]:
        if out==root or root in out.parents:raise ValueError('Diagnostics must not write dataset directories')
    roles=list(csv.DictReader(ROLES.open(encoding='utf-8-sig',newline='')))
    expected={'test_fault':392,'adapt_normal':10,'test_normal':33,'calibrate_normal':5,'supplemental_normal_test':25}
    if len(roles)!=465 or Counter(r['role'] for r in roles)!=expected:raise ValueError('Frozen role scope changed')
    inv={r['case_uid'].lower():r for r in (json.loads(s) for s in (RAW_AUDIT/'case_inventory.jsonl').read_text(encoding='utf-8').splitlines()) if r['domain']=='Real'}
    prior={}
    for name in ['raw_timelines.jsonl','real_remaining_timelines.jsonl']:
        for line in (RAW_AUDIT/name).read_text(encoding='utf-8').splitlines():
            r=json.loads(line)
            if r['domain']=='Real':prior[r['case_uid'].lower()]=r
    roles_hash=digest(ROLES);out.mkdir(parents=True,exist_ok=False)
    plan={'created_at':datetime.now(timezone.utc).isoformat(),'role_manifest_sha256':roles_hash,'script_sha256':digest(__file__),
          'roles':expected,'scope':'All 465 frozen Real roles, LABEL/TIME audit only. User-requested raw label inspection, NOT blind performance test. No model/threshold/feature selection/training or role reassignment.',
          'reference':'ROS exported command CSV at integer nanoseconds, first timesync anchor as used by existing converter; hold discrete IDs. Original ULog independent injection-message crosscheck. No physical fault-onset certification.',
          'limits':'ROS export not freshly decoded/verified against original bag. Clock bridge residual and transmission delay preclude declaring one source absolute ground truth. Candidate hold comparison does not automatically change labels.'}
    write=lambda p,d:p.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    write(out/'plan.json',plan)
    tasks=[(r,inv[r['case_uid'].lower()],prior[r['case_uid'].lower()]) for r in roles]
    records=[]
    with (out/'case_results.jsonl').open('x',encoding='utf-8') as f:
        with ThreadPoolExecutor(max_workers=3) as pool:
            for i,r in enumerate(pool.map(audit_case,tasks),1):
                records.append(r);f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+chr(10));f.flush()
                if i%100==0:print('REAL_LABEL_AUDIT',i,'/465',flush=True)
    summary={'cases':len(records),'status_counts':dict(Counter(r['status'] for r in records)),
             'issues':dict(Counter(s for r in records for s in r['issues'])),
             'warnings':dict(Counter(s for r in records for s in r['warnings'])),'by_role':{}}
    for role in expected:
        good=[r for r in records if r['role']==role and r['status']=='diagnostic_complete']
        item={'cases':len(good),'old_vs_ros_changed_row_cases':sum(r['old_vs_ros_row_disagreements']>0 for r in good),
              'old_vs_ros_row_disagreements':sum(r['old_vs_ros_row_disagreements'] for r in good),
              'normal_role_injection_flags':sum('normal_role_contains_recorded_injection_in_retained_crop' in r['warnings'] for r in good),'sampled':{}}
        for mode in ['cap64','stride10']:
            item['sampled'][mode]={k:sum(r['sampled'][mode][k] for r in good) for k in (good[0]['sampled'][mode] if good else [])}
        summary['by_role'][role]=item
    good=[r for r in records if r['status']=='diagnostic_complete']
    for field in ['old_minus_ros_onset_s','ros_minus_ulog_onset_s']:
        v=[r[field] for r in good if r[field] is not None]
        summary[field]={'n':len(v),'min':min(v) if v else None,'median':float(np.median(v)) if v else None,'max':max(v) if v else None}
    summary['roles_unchanged']=digest(ROLES)==roles_hash
    summary['decision']='Audit only. Preserve old roles/crops and final-test labels until semantics/coverage issues resolved. Do not use test observations to optimize model or align labels to predictions.'
    write(out/'summary.json',summary);print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
