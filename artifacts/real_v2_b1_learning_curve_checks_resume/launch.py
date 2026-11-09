"""Repeatable supplement-2 launch; the recovery entry owns the live OS lock."""
from pathlib import Path
import sys,subprocess,json
r=Path.cwd();c=r/'artifacts/real_v2_b1_learning_curve_checks_resume'
a=json.loads((c/'synthetic_audit_r2.json').read_text(encoding='utf-8'));assert a['exit_code']==0 and not a['real_data_model_access_attempts']
number=1
while(c/f'launch_{number}.json').exists():number+=1
cmd=[sys.executable,'-X','utf8','-E','-P','-B',str(r/'scripts/research/resume_real_v2_b1_learning_curve.py')]
with(c/f'launch_{number}.json').open('x',encoding='utf-8') as f:
 with(c/f'invocation_{number}.log').open('xb') as log:
  p=subprocess.Popen(cmd,cwd=r,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,close_fds=True,
   creationflags=subprocess.DETACHED_PROCESS|subprocess.CREATE_NEW_PROCESS_GROUP)
 json.dump({'pid':p.pid,'command':cmd,'authority':'B1 PI补充2','resume_launch_number':number,'not_a_new_experiment_repeat':True},f,indent=2);f.flush()
 print('AUTHORIZED_RESUME_STARTED',p.pid,flush=True)
 code=p.wait()
with(c/f'exit_{number}.json').open('x',encoding='utf-8') as f:json.dump({'pid':p.pid,'exit_code':code,'kernel_wait_no_polling':True},f,indent=2)
print('AUTHORIZED_RESUME_EXIT',code,flush=True)
