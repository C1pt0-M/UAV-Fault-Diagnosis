"""Finite gated M1 comparison executor; never retries or adds experiments."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib
import json
import os
import subprocess
import sys
import traceback
BASE=Path('/root/autodl-tmp/causal_m1_continuous_v1_20261102')
CODE=BASE/'code';DATA=Path('/root/autodl-tmp/causal_source_v1_20261031/unpacked/data')
FEATURES=BASE/'features';PROTOCOL=CODE/'configs/experiments/causal_m1_continuous150_v1.json';GATE=BASE/'pretraining_gate.json'
PYTHON='/root/miniconda3/bin/python';ANCHOR=Path('/root/autodl-tmp/causal_source_longbudget_v1_20261101/runs/uniform_replacement/initial.pt')


def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def write(p,d):
 q=p.with_suffix('.partial.json');q.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8');q.replace(p)


def main():
 with (BASE/'formal_execution_started.lock').open('x') as f:f.write(str(os.getpid()))
 if (BASE/'runs').exists() or (BASE/'remote_status.json').exists():raise FileExistsError('Preserve existing M1 formal sequence')
 state={'phase':'CHECK_GATES','pid':os.getpid(),'started_at':datetime.now(timezone.utc).isoformat(),'completed_jobs':[],'automatic_retries':False,'test_Real_or_deployment':False}
 def status(phase,**kw):state.update(phase=phase,updated_at=datetime.now(timezone.utc).isoformat(),**kw);write(BASE/'remote_status.json',state)
 try:
  status('CHECK_GATES');p=read(PROTOCOL);gate=read(GATE)
  if gate['status']!='PASS_ALL_M1_PRETRAINING_GATES' or gate['protocol_sha256']!=sha(PROTOCOL):raise ValueError('No completed frozen pretraining gates')
  if len(p['formal_jobs'])>4 or len(p['formal_jobs'])!=p['maximum_formal_jobs']:raise ValueError('Unexpected authorized job count')
  sys.path.insert(0,str(CODE/'scripts/research'));import train_causal_m1_continuous as trainer;trainer.load_protocol(PROTOCOL)
  if not trainer.torch.cuda.is_available():raise RuntimeError('CUDA not available')
  for k,v in p['formal_environment'].items():
   if trainer.source.environment()[k]!=v:raise ValueError('Registered environment mismatch:'+k)
  env=dict(os.environ,PYTHONPATH=str(CODE/'packages/uav_fault'),CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
  (BASE/'runs').mkdir(exist_ok=False);completed=[];reports={}
  for job in p['formal_jobs']:
   name=job['variant']+'__'+job['arm'];out=BASE/'runs'/job['variant']/job['arm'];status('FORMAL_TRAINING',current_job=name,completed_jobs=completed,registered_jobs=p['formal_jobs'])
   cmd=[PYTHON,'-u',str(CODE/'scripts/research/train_causal_m1_continuous.py'),'--protocol',str(PROTOCOL),'--data',str(DATA),'--features',str(FEATURES),'--output',str(out),'--initial-anchor',str(ANCHOR),'--gate',str(GATE),'--variant',job['variant'],'--arm',job['arm'],'--seed','42']
   out.parent.mkdir(exist_ok=True)
   with (BASE/(name+'.log')).open('x') as log:subprocess.run(cmd,cwd=CODE,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
   r=read(out/'report.json')
   if r['status']!='COMPLETE_FORMAL_SOURCE_M1_COMPARISON' or r['protocol_sha256']!=sha(PROTOCOL):raise ValueError('Wrong formal completion:'+name)
   reports[name]=r;completed.append(name)
  write(BASE/'paired_results.json',reports);status('ALL_FORMAL_JOBS_COMPLETE_VERIFY_PENDING',current_job=None,completed_jobs=completed)
 except Exception as exc:
  status('FAILED_STOPPED_NO_AUTOMATIC_RETRY',error=repr(exc),traceback=traceback.format_exc());raise


if __name__=='__main__':main()
