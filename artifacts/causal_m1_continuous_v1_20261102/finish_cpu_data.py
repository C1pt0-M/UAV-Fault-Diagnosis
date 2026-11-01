from pathlib import Path
import json,subprocess,os,time,sys
root=Path('artifacts/causal_m1_continuous_v1_20261102')
status=Path('D:/UAV-Data/causal_m1_continuous_v1/generation_status.json')
while True:
    s=json.loads(status.read_text(encoding='utf-8'))
    if s['status']=='COMPLETE_CASE_FEATURES_AWAITING_DATASET_FINALIZATION':break
    if s['status'].startswith('FAILED'):raise RuntimeError(s)
    time.sleep(10)
commands=[['python','-u','scripts/data/finalize_m1_continuous_features.py','--plan',str(root/'generation_plan.json')],
 ['python','-u','scripts/research/verify_m1_continuous_features.py','--plan',str(root/'generation_plan.json'),'--output',str(root/'independent_feature_verification.json')]]
env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
for i,cmd in enumerate(commands):
    with (root/('cpu_finalize_'+str(i)+'.log')).open('x',encoding='utf-8') as log:
        subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,env=env,check=True)
print('COMPLETE_ALL_CPU_FEATURE_EXPORT_AND_VERIFICATION_NO_TRAINING',flush=True)
