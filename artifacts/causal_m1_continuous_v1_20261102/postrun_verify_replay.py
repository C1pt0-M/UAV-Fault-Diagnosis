"""Wait for registered M1 jobs, verify, then replay. No training calls/retries."""
from pathlib import Path
from datetime import datetime,timezone
import json
import os
import subprocess
import time
import traceback
BASE=Path('/root/autodl-tmp/causal_m1_continuous_v1_20261102');CODE=BASE/'code';PY='/root/miniconda3/bin/python'
env=dict(os.environ,PYTHONPATH=str(CODE/'packages/uav_fault'),CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
state={'status':'WAITING_REGISTERED_FOUR_JOBS','started_at':datetime.now(timezone.utc).isoformat(),'training_calls':0,'final_test_or_Real':False}
def save():
 p=BASE/'postrun_status.json';q=p.with_suffix('.partial.json');q.write_text(json.dumps(state,ensure_ascii=False,indent=2),encoding='utf-8');q.replace(p)
save()
try:
 while True:
  status=json.loads((BASE/'remote_status.json').read_text())
  if status['phase']=='ALL_FORMAL_JOBS_COMPLETE_VERIFY_PENDING':break
  if status['phase'].startswith('FAILED'):raise RuntimeError('Formal job failure; no postrun retries:'+str(status))
  time.sleep(15)
 state['status']='VERIFYING_TRAINING_RESULTS';save()
 cmd=[PY,'-u',str(CODE/'scripts/research/verify_m1_training_results.py'),'--protocol',str(CODE/'configs/experiments/causal_m1_continuous150_v1.json'),'--data','/root/autodl-tmp/causal_source_v1_20261031/unpacked/data','--features',str(BASE/'features'),'--runs',str(BASE/'runs'),'--output',str(BASE/'verified_training')]
 with (BASE/'training_verification.log').open('x') as log:subprocess.run(cmd,cwd=CODE,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
 state['status']='CONTINUOUS_FIXED18_SOURCE_VAL_REPLAY';save()
 cmd=[PY,'-u',str(CODE/'scripts/research/replay_m1_continuous.py'),'--plan',str(BASE/'replay_plan.json'),'--data',str(BASE/'replay_base'),'--features',str(BASE/'features'),'--runs',str(BASE/'runs'),'--output',str(BASE/'continuous_replay'),'--device','cuda:0']
 with (BASE/'continuous_replay.log').open('x') as log:subprocess.run(cmd,cwd=CODE,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
 state.update(status='COMPLETE_TRAINING_VERIFICATION_AND_CONTINUOUS_REPLAY',completed_at=datetime.now(timezone.utc).isoformat());save()
except Exception as exc:
 state.update(status='FAILED_POSTRUN_STOPPED',error=repr(exc),traceback=traceback.format_exc());save();raise
