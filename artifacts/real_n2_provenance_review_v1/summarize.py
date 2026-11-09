"""Summarize extracted provenance, not model performance. No raw data opens."""
from pathlib import Path
import csv
import hashlib
import importlib.util
import json
import statistics

BASE = Path('D:/UAV-Fault-Diagnosis/artifacts/real_n2_provenance_review_v1')
spec = importlib.util.spec_from_file_location('provenance_inspection', BASE/'inspect.py')
a = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a)
a.OUT = BASE
P = BASE/'attempt3'
m = json.loads((P/'case_records.json').read_text(encoding='utf-8'))
n = {x['case_uid']: x for x in json.loads((P/'native_records.json').read_text(encoding='utf-8'))}
timing = json.loads((P/'command_clock_checks.json').read_text(encoding='utf-8'))
old = list(csv.DictReader((a.ROOT/'artifacts/real_v2_b1_n2_audit_v1/cohort_metadata.csv').open(encoding='utf-8-sig')))
alerts = sorted(x['case_uid'] for x in old if x['flight']=='waypoint' and x['family']=='motor' and int(x['R_events'])+int(x['S_plus_R_events'])>0)
controls = sorted(u for u,x in m.items() if x['family']=='normal')
matched = []
for target in alerts:
    for control in controls:
        x,y = m[target],m[control]
        geometry = bool(x['route_geometry']) and x['route_geometry']==y['route_geometry']
        speed = bool(x['route_speeds']) and x['route_speeds']==y['route_speeds']
        wait = x['post_route_wait_s'] is not None and x['post_route_wait_s']==y['post_route_wait_s']
        same_fc = n[target]['info']['sys_uuid']==n[control]['info']['sys_uuid']
        matched.append(dict(target=target, control=control, control_role=y['role_v1'],
            same_route_geometry=geometry, same_commanded_speed=speed, same_post_route_wait=wait,
            same_recorded_FC_UUID=same_fc, same_date_proxy=x['directory_date_proxy']==y['directory_date_proxy'],
            same_airframe_motor_payload_certified='unknown', same_environment_certified='unknown',
            independent_health_certified='unknown', strict_healthy_control_established=False))
a.table('healthy_control_match_matrix.csv', matched)
control_rows=[]
for u in controls:
    x=m[u]
    cal={k:v for k,v in n[u]['initial_parameters'].items() if k.startswith('CAL_')}
    target_cal={k:v for k,v in n['REAL/waypoint/motor/124_1']['initial_parameters'].items() if k.startswith('CAL_')}
    control_rows.append(dict(case_uid=u, role=x['role_v1'], date_proxy=x['directory_date_proxy'],
        geometry=json.dumps(x['route_geometry']), commanded_speeds=json.dumps(x['route_speeds']), wait_s=x['post_route_wait_s'],
        FC_UUID=n[u]['info']['sys_uuid'], same_initial_CAL_parameters_as_124=cal==target_cal,
        TestInfo_injection_commands=json.dumps(x['injection_commands']),
        original_ULog_active_command_recorded=n[u]['first_active_received'] is not None,
        whole_flight_independent_health='not_certified'))
a.table('healthy_control_candidates.csv', control_rows)
# Static candidate source snapshots: never import/run the flight-control programs.
repositories=[]
for root, paths in [
 ('D:/BuaaAutoTestPlatForm',['README.md','AutoTestAPI/AutoTestAPI.py','AutoTestAPI/command.py','AutoTestAPI/PX4MavCtrlV4.py','AutoTestAPI/data_recorder.py']),
 ('D:/RflyMAD/Data_processing_tools',['README.md','timetrans.py','data_real_ROS.json','OtherTools/windextract.py','OtherTools/bag2csv.py']),
 ('D:/PX4PSP/Firmware',['msg/CMakeLists.txt','src/modules/mavlink/mavlink_receiver.cpp','BkFile/EnvFault.sh'])]:
    r=dict(root=root, head=a.git('rev-parse','HEAD',cwd=root), last_commit=a.git('log','-1','--format=%H %aI %s',cwd=root),
        candidate_path_status=a.git('status','--short','--untracked-files=no','--',*paths,cwd=root), files=[])
    for name in paths:
        p=Path(root)/name
        data=p.read_bytes()
        lines=data.decode('utf-8',errors='replace').splitlines()
        terms=['start_time = time.time()', 'fault_inject_time =', 'FaultParam =', 'FaultParam,', 'Fault parameter',
               'Fault Parameter', 'c_mode = self.silFloats[0] - 10', 'CTime =', 'Duration =',
               'fault_vec[i] = self.silFloats[2+i]', 'FaultParam = np.concatenate', 'self.wp_start_time',
               'HIL_ACTUATOR_CONTROLS','no wind','Wind_title','TimeBridgeofPX4andROS']
        excerpt=[dict(line=i+1,text=s) for i,s in enumerate(lines) if any(term in s for term in terms)]
        r['files'].append(dict(path=str(p),bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),matching_source_lines=excerpt))
    repositories.append(r)
a.save('candidate_source_versions.json',repositories)
# Shared storage provenance metadata; never open archived recordings or excluded cases.
metadata_sources=[]
for path in [Path('D:/RflyMAD/organization_logs/real_zip_preflight_20260809.csv'),
             Path('D:/RflyMAD/organization_logs/real_organization_summary_20260809-220946.json')]:
    data=path.read_bytes()
    metadata_sources.append(dict(path=str(path),sha256=hashlib.sha256(data).hexdigest(),
        content=data.decode('utf-8-sig'), note='historical organization/checksum metadata, not a new verification of archives or acquisition truth'))
a.save('storage_provenance_metadata.json',metadata_sources)
# Flatten clocks to make observed program-clock vs rounded TestInfo differences explicit.
clock_rows=[]
for x in timing:
    send=x.get('program_time_last_before_send')
    recv=x.get('ULog_program_time_last_before_receive')
    clock_rows.append(dict(case_uid=x['case_uid'], TestInfo_s=x['TestInfo_injection_time'],
        ROS_last_program_time_s=float(send['program_time']) if send else None,
        ROS_record_age_s=x.get('program_time_record_age_s'),
        ULog_last_program_time_s=recv['program_time'] if recv else None,
        ROS_program_minus_TestInfo_s=float(send['program_time'])-float(x['TestInfo_injection_time']) if send else None))
a.table('program_clock_summary.csv',clock_rows)
# Complete tracked B1/N2 history and unrelated dirty files from here to submission.
paths=a.git('ls-files','-z')['stdout'].split('\0')
protected_paths=[x for x in paths if x and (x.startswith(('artifacts/real_v2_b1_', 'scripts/research/real_v2_b1_',
    'scripts/research/resume_real_v2_b1_', 'scripts/research/audit_real_v2_b1_n2.', 'tests/test_real_v2_b1')) or
    ('2026-11-08/跨域B1' in x) or x.endswith('/N2误报事件根因审计_v1.md') or x.endswith('real_v2_b1_learning_curve.json') or x.endswith('real_v2_split.csv'))]
protected_paths+=['scripts/research/case_relative_features_dev.py','tmp_control_compensation_audit.log','tmp_control_compensation_audit_sampled.log']
protected={x:dict(sha256=hashlib.sha256((a.ROOT/x).read_bytes()).hexdigest(),mtime_ns=(a.ROOT/x).stat().st_mtime_ns)
           for x in protected_paths}
a.save('all_history_before_submission.json',protected)
access=[]
for file in [BASE/'input_access_failure.json',BASE/'attempt2/input_access_failure.json',P/'input_access.json',BASE/'supplement_input_access.json']:
    rows=json.loads(file.read_text(encoding='utf-8'))
    access.extend(dict(attempt=str(file.relative_to(BASE)),**x) for x in rows)
a.save('combined_registered_input_access.json',access)
a.save('results.json',dict(status='COMPLETE_PROVENANCE_REVIEW_NO_CERTIFIED_STRICT_HEALTHY_CONTROL',
    eligible_inventory_cases=315, eligible_inventory_files=38875, TestInfo_cases=112, native_ULog_cases=76,
    native_bag_cases=1, strict_original_bag_records=27363, normal_crop_checks=50,
    strict_healthy_controls_established=0,
    full_normal_command_route_speed_wait_matches=sum(x['same_route_geometry'] and x['same_commanded_speed'] and x['same_post_route_wait'] and x['control_role']!='supplemental_normal_test' for x in matched),
    supplemental_command_match_cases=sorted({x['control'] for x in matched if x['same_route_geometry'] and x['same_commanded_speed'] and x['same_post_route_wait']}),
    command_124_explanation='periodic wrapper normalization consistent with native sent/received records and candidate code; historical source/serializer version not authenticated',
    independent_motor_RPM_or_current=False, battery_current_and_later_weather_exports_available=True,
    no_independent_physical_damage_truth=True, B1_H1_H2_unchanged=True, prior_N2_primary_E_unchanged=True,
    registered_material_reads_all_attempts=len(access), unique_registered_material_files=len({x['path'] for x in access}),
    register_excludes_preflight_tool_reads=True, decoder_additional_known_opens=2,
    historical_files_covered_during_extraction=76, all_history_submission_snapshot_files=len(protected),
    no_model_load=True,no_prediction=True,no_training=True,no_new_label=True,no_new_onset=True,
    no_gpu=True,no_control_execution=True,no_B2_or_T1=True))
print(json.dumps(json.loads((BASE/'results.json').read_text(encoding='utf-8')),ensure_ascii=False,indent=2))
