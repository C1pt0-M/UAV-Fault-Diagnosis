"""Detached supervisor: preserves final child exit even if observer disconnects."""
from pathlib import Path
import sys,subprocess,json,hashlib
r=Path.cwd();c=r/'artifacts/real_v2_b1_learning_curve_checks';marker=c/'launch.json'
if marker.exists():raise FileExistsError('One formal B1 launch only; preserve earlier attempt')
a=json.loads((c/'synthetic_audit_r1.json').read_text(encoding='utf-8'));assert a['exit_code']==0 and not a['real_data_model_access_attempts']
assert json.loads((c/'import_preflight.json').read_text(encoding='utf-8'))['status']=='PASS'
snap=json.loads((c/'protected_before.json').read_text(encoding='utf-8'))
assert all(hashlib.sha256((r/p).read_bytes()).hexdigest()==v['sha256'] and (r/p).stat().st_mtime_ns==v['mtime_ns'] for p,v in snap.items())
assert not(r/'artifacts/real_v2_b1_learning_curve').exists() and not Path('D:/UAV-Data/real_v2_b1_learning_curve_runtime').exists()
command=[sys.executable,'-X','utf8','-E','-P','-B',str(r/'scripts/research/real_v2_b1_learning_curve.py')]
with marker.open('x',encoding='utf-8') as f:
 with(c/'invocation.log').open('xb') as log:
  p=subprocess.Popen(command,cwd=r,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,close_fds=True,
   creationflags=subprocess.DETACHED_PROCESS|subprocess.CREATE_NEW_PROCESS_GROUP)
 json.dump({'pid':p.pid,'command':command,'source_sha256':hashlib.sha256((r/'scripts/research/real_v2_b1_learning_curve.py').read_bytes()).hexdigest(),
  'formal_launches':1,'no_auto_retry':True,'supplement1':'fallback A, tie lists retained'},f,indent=2);f.flush()
 print('FORMAL_B1_STARTED',p.pid,flush=True)
 code=p.wait()
with(c/'exit.json').open('x',encoding='utf-8') as f:json.dump({'pid':p.pid,'exit_code':code,'kernel_wait_no_polling':True,'no_restart':True},f,indent=2)
print('FORMAL_B1_EXIT',code,flush=True)
