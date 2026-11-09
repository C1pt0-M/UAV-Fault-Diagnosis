"""Scoped final checks and staging. Only audit outputs + authorized index update."""
from pathlib import Path
import subprocess,json,hashlib
R=Path.cwd();O=R/'artifacts/real_v2_b1_n2_audit_v1';C=R/'artifacts/real_v2_b1_n2_audit_v1_checks'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
load=lambda p:json.loads(p.read_text(encoding='utf-8'))
def staged():return [p.decode('utf-8')for p in subprocess.check_output(['git','diff','--cached','--name-only','-z']).split(b'\0')if p]
assert not staged(),'Do not stage over existing unrelated index entries'
report=R/'docs/research/2026-11-09/N2误报事件根因审计_v1.md'
for v in load(O/'B1_protected_before.json'):
 path=R/v['file'];assert sha(path)==v['sha256'] and path.stat().st_mtime_ns==v['mtime_ns']
audit=load(O/'execution_audit.json');audit['report_sha256']=sha(report);audit['index_update_authorized_by_current_PI']=True
(O/'execution_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
result=load(O/'results.json');assert result['new_fits']==result['new_predictions']==result['new_thresholds']==0
assert result['primary_class_counts']=={'R':{'A':0,'B':0,'C':0,'D':0,'E':18},'S_plus_R':{'A':0,'B':0,'C':0,'D':0,'E':27}}
assert load(C/'synthetic_audit_r2.json')['exit_code']==0 and not load(C/'synthetic_audit_r2.json')['forbidden_input_attempts']
assert load(C/'visual_review.json')['all_four_reviewed_pass']
files=[R/'scripts/research/audit_real_v2_b1_n2.py',R/'tests/test_real_v2_b1_n2_audit.py',report,R/'docs/文档索引与实验台账.md']
for folder in [O,C]:files.extend(p for p in folder.iterdir()if p.is_file())
files.extend(p for p in(O/'figures').iterdir()if p.is_file())
manifest=C/'submission_files.txt';manifest.write_text('preparing\n',encoding='utf-8');files.append(manifest)
files=sorted(set(files));allowed={p.relative_to(R).as_posix()for p in files}
for path in files:
 assert path.suffix.lower()not in ('.pkl','.pt','.pth','.npz','.npy','.xlsx','.ulg','.bag','.csv.partial')
 if path.suffix.lower()not in ('.png','.gz'):path.read_text(encoding='utf-8')
subprocess.run(['git','add','-f','--',*sorted(allowed)],check=True)
actual=staged();assert set(actual)<=allowed
assert not any(v.startswith(('artifacts/real_v2_b1_learning_curve/','artifacts/real_v2_b1_learning_curve_checks/','artifacts/real_v2_b1_learning_curve_checks_resume/'))for v in actual)
manifest.write_text('\n'.join(actual)+'\n',encoding='utf-8',newline='\n')
subprocess.run(['git','add','-f','--',manifest.relative_to(R).as_posix()],check=True)
authored=[p.relative_to(R).as_posix() for p in files if p.suffix=='.py' or p in (report,R/'docs/文档索引与实验台账.md')]
# Generated SVG path whitespace and raw logs are byte-preserved evidence, not source style.
subprocess.run(['git','-c','core.whitespace=trailing-space,space-before-tab,cr-at-eol','diff','--cached','--check','--',*authored],check=True)
print('B1_79_TEXTS_UNCHANGED;SCOPED_FILES',len(actual),'BYTES',sum((R/v).stat().st_size for v in actual),'NO_MODELS_ARRAYS_RAW_RECORDS')
print('STAGED_REPORT_SHA256',sha(report));print(subprocess.check_output(['git','diff','--cached','--stat']).decode('utf-8'))
