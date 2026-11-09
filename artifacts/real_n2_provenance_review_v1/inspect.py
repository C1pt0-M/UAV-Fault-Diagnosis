"""Read-only, eligible-Real provenance inventory; no models, scores or event replay.
Run: D:/python3.12.4/python.exe -X utf8 -E -P -B .../inspect.py --self-test
"""
import argparse
import ast
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path('D:/UAV-Fault-Diagnosis')
RAW = Path('D:/RflyMAD')
OUT = ROOT / 'artifacts/real_n2_provenance_review_v1'
ACCESS = []


def save(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + '\n', encoding='utf-8')


def table(name, rows):
    if not rows:
        raise ValueError('Empty evidence table: ' + name)
    with (OUT / name).open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def commands(text):
    # TestInfo stores NumPy string-array formatting, not a Python list literal.
    found = re.findall(r"'([^']*)'", str(text))
    if not found:
        raise ValueError('Missing quoted command sequence')
    result = []
    for command in found:
        values = [float(x) for x in command.split(',')]
        if len(values) < 2 or any(not __import__('math').isfinite(x) for x in values):
            raise ValueError('Malformed command')
        result.append(values)
    return result


def command_summary(meta):
    quoted = re.findall(r"'([^']*)'", str(meta['Control Sequence']))
    if not quoted:
        raise ValueError('Missing quoted command sequence')
    seq, parse_errors = [], []
    for raw_command in quoted:
        try:
            seq.extend(commands("['" + raw_command + "']"))
        except ValueError as exc:
            parse_errors.append(dict(raw_command=raw_command, error=str(exc)))
            # Keep an unrecognized placeholder, so adjacency is not changed.
            seq.append([])
    route = [x for x in seq if x[:2] == [2, 9]]
    inject = [x for x in seq if x[:2] == [2, 6]]
    route_index = next((i for i, x in enumerate(seq) if x[:2] == [2, 9]), None)
    wait = None
    if route_index is not None and route_index + 1 < len(seq):
        following = seq[route_index + 1]
        if following[:2] == [1, 1] and len(following) == 3:
            wait = following[2]
    speeds, geometry = [], []
    for x in route:
        if (len(x) - 2) % 5:
            raise ValueError('Unrecognized waypoint command shape')
        for i in range(2, len(x), 5):
            speeds.append(x[i])
            geometry.append(x[i + 1:i + 5])
    return dict(route=route, route_geometry=geometry, route_speeds=speeds,
                parse_errors=parse_errors, post_route_wait_s=wait, injection_commands=inject,
                injection_time_TestInfo=meta.get('Fault injection time'),
                end_time_TestInfo=meta.get('Test end time'))


def eligible(rows):
    return {x['case_uid']: x for x in rows
            if x['partition'] == 'dev' and x['label_quarantine'].lower() == 'false'}


def read_raw(path, uid, allowed, purpose):
    path = Path(path).resolve()
    if uid not in allowed or not path.is_relative_to((RAW / uid).resolve()):
        raise PermissionError('Not an eligible case input')
    if path.suffix.lower() not in {'.xlsx', '.csv', '.txt', '.ulg'}:
        raise PermissionError('Input type not authorized by this inventory')
    data = path.read_bytes()
    ACCESS.append(dict(case_uid=uid, path=str(path), purpose=purpose,
                       size=len(data), sha256=hashlib.sha256(data).hexdigest()))
    return data


def rows_csv(path, uid, allowed, purpose):
    return list(csv.DictReader(io.StringIO(read_raw(path, uid, allowed, purpose).decode('utf-8-sig'))))


def only(paths):
    paths = list(paths)
    if len(paths) != 1:
        raise ValueError(f'Expected exactly one path: {paths}')
    return paths[0]


def active_record(rows):
    for i, x in enumerate(rows):
        if 123450 <= int(float(x['id'])) <= 123456:
            return i, x
    return None, None


def self_test():
    assert set(eligible([
        dict(case_uid='a', partition='dev', label_quarantine='False'),
        dict(case_uid='b', partition='sealed', label_quarantine='False'),
        dict(case_uid='c', partition='dev', label_quarantine='True')])) == {'a'}
    s = command_summary({'Control Sequence': "['2,9,3,50,0,15,0,3,10,0,15,90' '1,1,80' '2,6,123450,11,10,1,.8,1,1,1']"})
    assert s['route_speeds'] == [3, 3] and s['post_route_wait_s'] == 80
    assert s['injection_commands'][0][3:] == [11, 10, 1, .8, 1, 1, 1]
    assert commands("['1,1,5' '1,1,10']") == [[1., 1., 5.], [1., 1., 10.]]
    assert active_record([{'id': '1500'}, {'id': '123450'}])[0] == 1
    broken = command_summary({'Control Sequence': "['2,1,0.0.66,1,15' '2,9,3,50,0,15,0' '1,1,80']"})
    assert len(broken['parse_errors']) == 1 and broken['route_speeds'] == [3] and broken['post_route_wait_s'] == 80
    for bad in ["['2,9,nan']", "unquoted", "['2,9,3,50']"]:
        try:
            commands(bad) if 'nan' in bad else command_summary({'Control Sequence': bad})
        except ValueError:
            pass
        else:
            raise AssertionError(bad)
    try:
        read_raw(RAW / 'REAL/waypoint/motor/124_1/no_wind_data.txt', 'locked', {}, 'test')
    except PermissionError:
        pass
    else:
        raise AssertionError('Eligibility guard')
    assert ACCESS == []
    print('SELF_TEST_PASS: eligibility, quoted-array parser, route shape, finite values, active ID, read guard; zero raw opens')


def git(*args, cwd=ROOT):
    r = subprocess.run(['git', '-C', str(cwd), *args], capture_output=True, text=True, encoding='utf-8', errors='replace')
    return dict(args=list(args), exit_code=r.returncode, stdout=r.stdout, stderr=r.stderr)


def snapshot():
    paths = git('ls-files', '--', 'artifacts/real_v2_b1_learning_curve', 'artifacts/real_v2_b1_n2_audit_v1',
                'docs/research/2026-11-08/跨域B1学习曲线结果_v1.md',
                'docs/research/2026-11-09/N2误报事件根因审计_v1.md')['stdout'].splitlines()
    # git may quote non-ASCII paths; add report literals explicitly.
    paths = [p for p in paths if not p.startswith('"')]
    paths += ['docs/research/2026-11-08/跨域B1学习曲线结果_v1.md', 'docs/research/2026-11-09/N2误报事件根因审计_v1.md',
              'scripts/research/case_relative_features_dev.py', 'tmp_control_compensation_audit.log',
              'tmp_control_compensation_audit_sampled.log']
    return {p: dict(size=(ROOT/p).stat().st_size, mtime_ns=(ROOT/p).stat().st_mtime_ns,
                    sha256=hashlib.sha256((ROOT/p).read_bytes()).hexdigest()) for p in sorted(set(paths))}


def run():
    if (OUT / 'inventory.json').exists() or (OUT / 'failure.json').exists():
        raise FileExistsError('Preserve prior run; this entry point does not silently rerun')
    import numpy as np
    import openpyxl
    from pyulog import ULog
    started = datetime.now(timezone.utc).isoformat()
    save('protected_before.json', snapshot())
    split = list(csv.DictReader((ROOT / 'configs/data/real_v2_split.csv').open(encoding='utf-8-sig')))
    allow = eligible(split)
    assert len(allow) == 315
    old = list(csv.DictReader((ROOT/'artifacts/real_v2_b1_n2_audit_v1/cohort_metadata.csv').open(encoding='utf-8-sig')))
    alerts = {x['case_uid'] for x in old if int(x['R_events']) + int(x['S_plus_R_events']) > 0}
    motor = {u for u, x in allow.items() if x['flight'] == 'waypoint' and x['family'] == 'motor'}
    normals = {u for u, x in allow.items() if x['family'] == 'normal'}
    selected = motor | normals | alerts
    assert len(alerts) == 20 and len(motor) == 51 and len(normals) == 50
    save('plan.json', dict(started=started, head=git('rev-parse', 'HEAD'),
        eligible_cases=315, TestInfo_cases=sorted(selected),
        selection='all 51 eligible waypoint/motor fault-family cases + all 50 eligible normal-family cases + original 20 N2 alert cases (union)',
        native_selection='all early-date waypoint/motor fault cases + all 50 normal-family cases + original 20 alerts; dates from directory-name metadata only',
        restrictions=['no source/internal/locked payloads', 'no fitting/inference/scores/event replay', 'no label/onset/threshold changes',
                      'no flight/control script execution', 'no physical hardware or network control'],
        prior_preflight='Read tools and metadata inspections before this instrumented inventory are disclosed separately in report.'))
    files, metadata = [], {}
    # Names and sizes only, within already-filtered eligible directories.
    for uid in sorted(allow):
        p = RAW / uid
        for f in p.rglob('*'):
            if f.is_file():
                files.append(dict(case_uid=uid, relative_path=str(f.relative_to(p)), bytes=f.stat().st_size))
    table('eligible_file_inventory.csv', files)
    for uid in sorted(selected):
        p = RAW / uid
        f = only(p.glob('TestInfo*.xlsx'))
        wb = openpyxl.load_workbook(io.BytesIO(read_raw(f, uid, allow, 'original TestInfo')), read_only=True, data_only=True)
        try:
            pairs = list(wb.active.values)
            meta = {str(r[0]): r[1] for r in pairs if len(r) >= 2 and r[0] is not None}
        finally:
            wb.close()
        ulog = only(p.glob('**/*.ulg'))
        dates = re.findall(r'(20\d{2})-(\d{1,2})-(\d{1,2})', ulog.stem)
        if len(dates) != 1:
            raise ValueError('Unrecognized date: ' + str(ulog))
        date = '-'.join([dates[0][0], dates[0][1].zfill(2), dates[0][2].zfill(2)])
        metadata[uid] = dict(**allow[uid], TestInfo=meta, **command_summary(meta), directory_date_proxy=date,
                             TestInfo_path=str(f), ULog_path=str(ulog))
    save('case_records.json', metadata)
    native = {u for u in motor if metadata[u]['directory_date_proxy'] in {'2023-05-22', '2023-05-23'}} | normals | alerts
    save('native_plan.json', dict(cases=sorted(native), n=len(native)))
    records, fieldstats, timing, details124 = [], [], [], {}
    csv_quality = []
    topics = ['rfly_ctrl_lxl', 'offb_flight_mode', 'esc_status', 'battery_status', 'system_power', 'vehicle_status',
              'sensor_accel', 'sensor_gyro', 'sensor_mag', 'wind_estimate', 'wind', 'actuator_outputs', 'actuator_outputs_real']
    for index, uid in enumerate(sorted(native), 1):
        print(f'{index}/{len(native)} {uid}', flush=True)
        m = metadata[uid]
        ul = ULog(io.BytesIO(read_raw(m['ULog_path'], uid, allow, 'native command/physical-evidence fields')), message_name_filter_list=topics)
        info = {k: str(v) for k, v in ul.msg_info_dict.items()}
        params = {k: v.item() if hasattr(v, 'item') else v for k, v in ul.initial_parameters.items()
                  if k.startswith(('CAL_', 'SYS_', 'MAV_SYS_ID', 'SENS_BOARD', 'BAT_', 'PWM_'))}
        inv = []
        received, pg = None, None
        for d in ul.data_list:
            inv.append(dict(topic=d.name, multi_id=int(d.multi_id), rows=len(d.data.get('timestamp', [])), fields=list(d.data)))
            if d.name == 'rfly_ctrl_lxl' and d.multi_id == 0:
                active = np.flatnonzero((d.data['id'] >= 123450) & (d.data['id'] <= 123456))
                received = {k: float(v[active[0]]) for k, v in d.data.items()} if len(active) else None
            if d.name == 'offb_flight_mode' and d.multi_id == 0:
                pg = d.data
            if d.name in {'esc_status', 'battery_status', 'wind_estimate', 'wind', 'system_power'}:
                for k, v in d.data.items():
                    if k == 'timestamp' or not np.issubdtype(v.dtype, np.number):
                        continue
                    finite = v[np.isfinite(v)]
                    fieldstats.append(dict(case_uid=uid, topic=d.name, multi_id=int(d.multi_id), field=k,
                        rows=len(v), finite=len(finite), nonzero=int(np.count_nonzero(finite)),
                        minimum=float(np.min(finite)) if len(finite) else None,
                        maximum=float(np.max(finite)) if len(finite) else None))
        texts = []
        for f in (RAW / uid).glob('*.txt'):
            texts.append(dict(path=str(f), content=read_raw(f, uid, allow, 'case text provenance/weather marker').decode('utf-8', errors='replace')))
        records.append(dict(case_uid=uid, directory_date_proxy=m['directory_date_proxy'], info=info,
                            initial_parameters=params, topics=inv, first_active_received=received, text_records=texts))
        control = only((RAW / uid).glob('**/_slash_mavros_slash_rfly_ctrl_lxl_slash_send_data.csv'))
        sent_rows = rows_csv(control, uid, allow, 'exported ROS sent injection records')
        for kind, required, source_rows in [('injection', ['rosbagTimestamp', 'index', 'id', 'mode', 'controls'], sent_rows)]:
            bad = [dict(row_index=i, record=r) for i, r in enumerate(source_rows) if any(r.get(k) in (None, '') for k in required)]
            csv_quality.append(dict(case_uid=uid, kind=kind, total=len(source_rows), malformed=bad))
            sent_rows = [r for r in source_rows if all(r.get(k) not in (None, '') for k in required)]
        active_i, sent = active_record(sent_rows)
        offb = only((RAW / uid).glob('**/_slash_mavros_slash_offb_flight_mode_slash_send_data.csv'))
        off_rows = rows_csv(offb, uid, allow, 'exported ROS program clock/status')
        required = ['rosbagTimestamp', 'program_time', 'sleep_status', 'flight_status']
        bad = [dict(row_index=i, record=r) for i, r in enumerate(off_rows) if any(r.get(k) in (None, '') for k in required)]
        csv_quality.append(dict(case_uid=uid, kind='program_clock', total=len(off_rows), malformed=bad))
        off_rows = [r for r in off_rows if all(r.get(k) not in (None, '') for k in required)]
        if not off_rows:
            raise ValueError('No interpretable clock rows: ' + uid)
        t = dict(case_uid=uid, TestInfo_injection_time=m['injection_time_TestInfo'],
                 sent_first_active=sent, received_first_active=received,
                 ROS_export_program_time_min=min(float(r['program_time']) for r in off_rows),
                 ROS_export_program_time_max=max(float(r['program_time']) for r in off_rows),
                 ROS_export_sleep_status=sorted({r['sleep_status'] for r in off_rows}),
                 ROS_export_flight_status=sorted({r['flight_status'] for r in off_rows}))
        if sent is not None:
            ns = int(sent['rosbagTimestamp'])
            before = [r for r in off_rows if int(r['rosbagTimestamp']) <= ns]
            nearest = before[-1] if before else None
            t['program_time_last_before_send'] = nearest
            t['program_time_record_age_s'] = (ns - int(nearest['rosbagTimestamp'])) / 1e9 if nearest else None
        if received is not None and pg is not None:
            ix = np.searchsorted(pg['timestamp'], int(received['timestamp']), side='right') - 1
            t['ULog_program_time_last_before_receive'] = {k: float(v[ix]) for k, v in pg.items()} if ix >= 0 else None
        timing.append(t)
        if uid == 'REAL/waypoint/motor/124_1':
            # Preserve command state transitions, not just the first active value.
            last = None
            transitions = []
            for r in sent_rows:
                key = (r['index'], r['id'], r['mode'], r['controls'])
                if key != last:
                    transitions.append(r)
                    last = key
            recv = next(d.data for d in ul.data_list if d.name == 'rfly_ctrl_lxl' and d.multi_id == 0)
            rlast, rtrans = None, []
            for i in range(len(recv['timestamp'])):
                key = tuple(float(recv[k][i]) for k in recv if k != 'timestamp')
                if key != rlast:
                    rtrans.append({k: float(v[i]) for k, v in recv.items()})
                    rlast = key
            ro = only((RAW/uid).glob('**/_slash_rosout.csv'))
            rosout = rows_csv(ro, uid, allow, '124_1 original exported ROS execution metadata')
            details124 = dict(sent_state_transitions=transitions, received_state_transitions=rtrans,
                ROS_execution_files=sorted({(r.get('name',''),r.get('file',''),r.get('function','')) for r in rosout}),
                ROS_execution_messages=rosout)
    save('native_records.json', records)
    save('CSV_record_quality.json', csv_quality)
    table('physical_field_statistics.csv', fieldstats)
    save('command_clock_checks.json', timing)
    save('124_1_command_transitions.json', details124)
    save('input_access.json', ACCESS)
    after = snapshot()
    before = json.loads((OUT/'protected_before.json').read_text(encoding='utf-8'))
    assert before == after, 'Historical or unrelated files changed'
    save('protected_after.json', after)
    save('inventory.json', dict(status='COMPLETE_READ_ONLY_EXTRACTION', started=started,
        completed=datetime.now(timezone.utc).isoformat(), eligible_file_inventory_cases=315,
        TestInfo_cases=len(metadata), native_cases=len(records), raw_opens=len(ACCESS),
        unique_raw_files=len({x['path'] for x in ACCESS}), historical_files_unchanged=len(after),
        no_models=True, no_scores=True, no_new_predictions=True, no_training=True,
        no_control_execution=True, no_network=True))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--attempt', type=int, default=1)
    args = parser.parse_args()
    if args.attempt > 1:
        OUT = OUT / ('attempt' + str(args.attempt))
        OUT.mkdir(exist_ok=True)
    if args.self_test:
        self_test()
    else:
        try:
            run()
        except Exception as exc:
            save('input_access_failure.json', ACCESS)
            save('failure.json', dict(error=repr(exc), traceback=__import__('traceback').format_exc()))
            raise
