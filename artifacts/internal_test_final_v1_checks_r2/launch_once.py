"""Supplement-1 single formal invocation, only after full VAL preflight PASS."""
from pathlib import Path
import hashlib,json,subprocess,sys
root=Path.cwd();checks=root/'artifacts/internal_test_final_v1_checks_r2'
record=checks/'launch_receipt.json'
if record.exists():raise FileExistsError('No second r2 formal invocation')
pre=json.loads((checks/'preflight.json').read_text(encoding='utf-8'))
assert pre['status']=='PASS' and not pre['blocked_accesses'] and len(pre['cases'])==2
assert pre['config_sha256']==hashlib.sha256((root/'configs/experiments/internal_test_final_v1.json').read_bytes()).hexdigest()
unit=json.loads((checks/'synthetic_access_audit.json').read_text(encoding='utf-8'))
assert unit['exit_code']==0 and not unit['real_data_access_attempts_blocked']
assert not (root/'artifacts/internal_test_final_v1').exists()
scores=Path('D:/UAV-Data/internal_test_final_v1_scores')
assert scores.is_dir() and not any(scores.iterdir())
command=[sys.executable,'-X','utf8','-E','-P','-B',str(root/'scripts/research/internal_test_final_v1.py')]
with record.open('x',encoding='utf-8') as marker:
 with(checks/'invocation_r2.log').open('xb') as log:
  process=subprocess.Popen(command,cwd=root,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
      creationflags=subprocess.DETACHED_PROCESS|subprocess.CREATE_NEW_PROCESS_GROUP,close_fds=True)
  json.dump({'pid':process.pid,'command':command,'launches_this_authorization':1,'detached':True,
             'authority':'内部test最终评价_PI指令_补充1.md','preflight_pass':True,'no_metric_rerun':True},marker,indent=2,ensure_ascii=False)
 marker.flush()
 print('AUTHORIZED_FORMAL_STARTED_ONCE',process.pid,flush=True)
 exit_code=process.wait()
with(checks/'process_exit_receipt.json').open('x',encoding='utf-8') as f:
 json.dump({'pid':process.pid,'exit_code':exit_code,'wait_no_polling':True,'no_restart':True},f,indent=2)
print('FORMAL_EXIT',exit_code,flush=True)
raise SystemExit(exit_code)
