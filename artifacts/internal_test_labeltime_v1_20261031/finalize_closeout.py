from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import time

ROOT = Path('D:/UAV-Fault-Diagnosis')
ART = ROOT / 'artifacts/internal_test_labeltime_v1_20261031'
DATA = Path('D:/UAV-Data/internal_test_labeltime_v1')
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()

for _ in range(180):
    if all((ART / f'independent_part{i}.json').exists() for i in range(1, 6)) and (ART / 'independent_cache.json').exists():
        break
    for p in ART.glob('verification_job*.json'):
        d = json.loads(p.read_text(encoding='utf-8'))
        if d.get('returncode', 0):
            raise RuntimeError(str(d))
    time.sleep(5)
else:
    raise RuntimeError('Independent verification not finished; do not mark complete')

reports = {f'independent_part{i}.json': json.loads((ART / f'independent_part{i}.json').read_text(encoding='utf-8')) for i in range(1, 6)}
cache = json.loads((ART / 'independent_cache.json').read_text(encoding='utf-8'))
assert all(r['status'] == 'PASS' for r in reports.values()) and cache['status'] == 'PASS'
assert sum(r['cases'] for r in reports.values()) == 553
assert sum(r['windows'] for r in reports.values()) == 35392
assert sum(r['changed_state_rows'] for r in reports.values()) == 28644
assert cache['corrected_qualified_endpoint_labels'] == 603 and cache['ineligible_endpoints'] == 27
manifest = json.loads((DATA / 'repair_manifest.json').read_text(encoding='utf-8'))
assert manifest['status'] == 'COMPLETE_INTERNAL_TEST_LABELTIME_REPAIR_WITH_EXPLICIT_MASKS'
assert len(list((DATA / 'cases').rglob('*.csv'))) == 553
assert len(list((DATA / 'references').glob('*.npz'))) == 553
assert not (DATA / 'test.pt').exists() and not (DATA / 'train.pt').exists()
source = Path('D:/UAV-Data/online_cache_v2_labeltime_v1/repair_manifest.json')
real = Path('D:/UAV-Data/real_labeltime_v1/repair_manifest.json')
s = json.loads(source.read_text(encoding='utf-8')); r = json.loads(real.read_text(encoding='utf-8'))
assert s['status'] == 'COMPLETE_SOURCE_LABELTIME_REPAIR'
assert r['status'] == 'COMPLETE_REAL_LABELTIME_REPAIR_WITH_EXPLICIT_QUARANTINE'
assert not s['model_training_run'] and not r['training_run']
snapshot = ART / 'source_snapshots'; snapshot.mkdir(exist_ok=False)
for p in ['scripts/data/repair_internal_test_labeltime.py', 'scripts/research/verify_internal_test_labeltime.py',
          'tests/test_internal_test_labeltime_repair.py', 'scripts/data/repair_fixed_crop_labeltime.py']:
    shutil.copyfile(ROOT / p, snapshot / Path(p).name)
for name in ['repair_manifest.json', 'README.md', 'EXECUTION_REVISION.json']:
    shutil.copyfile(DATA / name, ART / ('delivered_' + name))
completion = {'status': 'PASS_ACTUAL_INTERNAL_TEST_LABELTIME_REPAIR', 'completed_at': datetime.now(timezone.utc).isoformat(),
    'data_root': str(DATA), 'cases': 553, 'original_windows_preserved': 35392, 'corrected_qualified_endpoint_labels': 603,
    'ineligible_endpoints': 27, 'regression_tests_passed': 91, 'all_original_X_preserved': True,
    'old_assets_and_normalization_unchanged': True, 'models_or_training_used': False, 'event_scoring_ready': False,
    'repair_script_sha256': sha(ROOT / 'scripts/data/repair_internal_test_labeltime.py'),
    'independent_reports_sha256': {name: sha(ART / name) for name in [*reports, 'independent_cache.json']},
    'output_manifest_sha256': sha(DATA / 'repair_manifest.json'),
    'scope': 'Confirmed label-time corrections completed for frozen internal tests. Unknowns remain explicit, no arbitrary extra filtering or model evaluation.'}
with (ART / 'completion.json').open('x', encoding='utf-8') as f:
    json.dump(completion, f, ensure_ascii=False, indent=2)
consolidated = {'status': 'LABELTIME_REPAIR_SCOPE_CLOSED_WITH_EXPLICIT_UNRESOLVED_LEDGER',
    'completed_at': completion['completed_at'], 'source_train_val': {'root': str(source.parent), 'corrected_retained_labels': 3780, 'held_out_windows': 343, 'manifest_sha256': sha(source)},
    'internal_test': {'root': str(DATA), 'corrected_qualified_labels': 603, 'unqualified_endpoints': 27, 'all_original_windows': 35392, 'manifest_sha256': sha(DATA / 'repair_manifest.json')},
    'real': {'root': str(real.parent), 'corrected_qualified_labels': 10, 'unqualified_endpoints': 775, 'quarantined_cases': 9, 'all_original_windows': 29319, 'manifest_sha256': sha(real)},
    'training_or_model_evaluation_run': False, 'strict_online_or_physical_truth_certified': False,
    'boundary': 'Not all data-quality issues solved: known disagreements and Real9 stay explicit. No remaining unexamined internal-test split. No new dataset/crop/model research in this closeout.'}
with (ART / 'consolidated_closeout.json').open('x', encoding='utf-8') as f:
    json.dump(consolidated, f, ensure_ascii=False, indent=2)
p = ROOT / '项目进程.md'
entry = '| 2026-10-31 | 内部SIL/HIL测试标签收尾，三个既定范围实际修订均交付 | internal_test_labeltime_v1含553修复CSV、35392原X／索引全保留，603合格末端改标、27来源分歧待核。全部原始事件／CSV按5批独立复核及全缓存身份检查通过，91项回归；源与Real旧交付保留，无模型训练或新成绩，9条Real及来源分歧不强行修好 |'
if entry not in p.read_text(encoding='utf-8'):
    with p.open('a', encoding='utf-8') as f:
        f.write(chr(10) + entry + chr(10))
print(json.dumps(consolidated, ensure_ascii=False, indent=2))
