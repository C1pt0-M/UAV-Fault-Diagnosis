from pathlib import Path
from datetime import datetime,timezone
import os,subprocess,json,time
base=Path('/root/autodl-tmp/causal_m1_continuous_v1_20261102');code=base/'code'
while True:
 p=base/'postrun_status.json';s=json.loads(p.read_text())
 if s['status']=='COMPLETE_TRAINING_VERIFICATION_AND_CONTINUOUS_REPLAY':break
 if s['status'].startswith('FAILED'):raise RuntimeError('No natural replay after failed prereq:'+str(s))
 time.sleep(15)
env=dict(os.environ,PYTHONPATH=str(code/'packages/uav_fault'),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
cmd=['/root/miniconda3/bin/python','-u',str(base/'replay_m1_natural_availability.py'),'--plan',str(base/'replay_plan.json'),'--data',str(base/'replay_base'),'--features',str(base/'features'),'--runs',str(base/'runs'),'--main-replay',str(base/'continuous_replay'),'--output',str(base/'natural_availability_replay'),'--device','cuda:0']
with (base/'natural_availability_replay.log').open('x') as log:r=subprocess.run(cmd,cwd=code,env=env,stdout=log,stderr=subprocess.STDOUT)
record={'status':'COMPLETE_NATURAL_AVAILABILITY_REPLAY' if r.returncode==0 else 'FAILED_NATURAL_AVAILABILITY_REPLAY','exit_code':r.returncode,'completed_at':datetime.now(timezone.utc).isoformat(),'new_training':False}
(base/'natural_replay_status.json').write_text(json.dumps(record,indent=2));r.check_returncode()
