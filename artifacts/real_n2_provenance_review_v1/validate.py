"""Minimal synthetic checks and submission validation; no real payload access."""
from pathlib import Path
import csv
import hashlib
import importlib.util
import json
import subprocess

ROOT = Path('D:/UAV-Fault-Diagnosis')
BASE = ROOT/'artifacts/real_n2_provenance_review_v1'
spec = importlib.util.spec_from_file_location('provenance_inspection', BASE/'inspect.py')
a = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a)
a.OUT = BASE
a.self_test()
protected = json.loads((BASE/'all_history_before_submission.json').read_text(encoding='utf-8'))
for name, expected in protected.items():
    p = ROOT/name
    assert hashlib.sha256(p.read_bytes()).hexdigest() == expected['sha256'], name
    assert p.stat().st_mtime_ns == expected['mtime_ns'], name
results = json.loads((BASE/'results.json').read_text(encoding='utf-8'))
assert results['strict_healthy_controls_established'] == 0
assert results['full_normal_command_route_speed_wait_matches'] == 0
assert results['supplemental_command_match_cases'] == ['REAL/waypoint/motor/125_1']
matched = list(csv.DictReader((BASE/'healthy_control_match_matrix.csv').open(encoding='utf-8')))
assert len(matched) == 450
crops = list(csv.DictReader((BASE/'normal_crop_boundaries.csv').open(encoding='utf-8')))
assert len(crops) == 50 and all(int(x['saved_valid_reference_active']) == 0 for x in crops)
assert sum(x['original_received_first_after_saved_end']=='True' for x in crops) == 18
bag = json.loads((BASE/'124_1_original_bag_check.json').read_text(encoding='utf-8'))
assert bag['decoded_target_records'] == bag['indexed_target_records'] == 27363
assert bag['full_command_export_exact'] and bag['active_intervals'] == 11
assert min(bag['active_start_periods_s']) > 9.9 and max(bag['active_start_periods_s']) < 10.1
assert min(bag['complete_active_durations_s']) > .9 and max(bag['complete_active_durations_s']) < 1.1
report = ROOT/'docs/research/2026-11-09/Real实验来源取证与健康对照审查_v1.md'
ledger = ROOT/'docs/文档索引与实验台账.md'
assert '前置检索范围偏差' in report.read_text(encoding='utf-8')
assert report.name in ledger.read_text(encoding='utf-8')
# Authored files only: raw generated CSV evidence retains its native CRLF bytes.
checks = []
for p in sorted([*BASE.glob('*.py'), report]):
    r = subprocess.run(['git','diff','--no-index','--check','--','/dev/null',str(p)], cwd=ROOT,
                       capture_output=True,text=True,encoding='utf-8',errors='replace')
    checks.append(dict(path=str(p.relative_to(ROOT)),exit_code=r.returncode,stdout=r.stdout,stderr=r.stderr))
r = subprocess.run(['git','diff','--check','--',str(ledger.relative_to(ROOT))],cwd=ROOT,
                   capture_output=True,text=True,encoding='utf-8',errors='replace')
checks.append(dict(path=str(ledger.relative_to(ROOT)),exit_code=r.returncode,stdout=r.stdout,stderr=r.stderr))
a.save('authored_whitespace_checks_r2.json',checks)
# --no-index implies --exit-code: 1 means different files, not a whitespace error.
assert all(x['exit_code'] in (0, 1) and not x['stdout'] for x in checks), 'See retained whitespace receipts'
a.save('review_validation.json',dict(status='PASS',synthetic_check='PASS; no raw opens',
    protected_history_and_unrelated_sha_mtime_unchanged=len(protected),
    exact_command_match_matrix_rows=450,normal_crop_checks=50,strict_original_bag_records=27363,
    authored_whitespace='PASS',raw_recordings_models_arrays_written=False,
    git_commit_executed=False,git_push_executed=False))
print('REVIEW_VALIDATION_PASS',len(protected),'protected files; 450 comparisons; 50 crops; 27363 original bag records')
