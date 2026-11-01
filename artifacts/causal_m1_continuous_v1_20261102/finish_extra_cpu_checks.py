"""Finish non-GPU prerequisites only; no model experiment launcher."""
from pathlib import Path
import json,subprocess,os,time
root=Path('artifacts/causal_m1_continuous_v1_20261102');features=Path('D:/UAV-Data/causal_m1_continuous_v1')
plan=json.loads((root/'generation_plan.json').read_text(encoding='utf-8'))
while len(list((features/'case_records').glob('*.json')))<len(plan['tasks']):time.sleep(10)
env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
commands=[('invalid_detail',['python','-u','scripts/data/explain_m1_invalid_positions.py','--plan',str(root/'generation_plan.json')]),
 ('basic_replay',['python','-u','scripts/research/check_m1_pretraining_replay.py','--plan',str(root/'replay_plan.json'),'--data','D:/UAV-Data/causal_input_v1','--features',str(features),'--output',str(root/'pretraining_replay_basic.json')])]
for name,cmd in commands:
 with (root/(name+'.log')).open('x',encoding='utf-8') as log:subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
print('COMPLETE_EXTRA_CPU_GATES_NO_FORMAL_GPU',flush=True)
