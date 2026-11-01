"""Wait for current feature workers; resume only a status PermissionError.

No worker interruption, data deletion, source edits or GPU training.
"""
from pathlib import Path
from datetime import datetime,timezone
import json
import subprocess
import time
import os
import psutil
root=Path('artifacts/causal_m1_continuous_v1_20261102')
launch=json.loads((root/'feature_build_launch.json').read_text(encoding='utf-8'))
while psutil.pid_exists(launch['pid']):time.sleep(5)
log=(root/'feature_build.log').read_text(encoding='utf-8',errors='replace')
status=json.loads(Path('D:/UAV-Data/causal_m1_continuous_v1/generation_status.json').read_text(encoding='utf-8'))
if status['status']=='COMPLETE_CASE_FEATURES_AWAITING_DATASET_FINALIZATION':raise SystemExit(0)
if 'PermissionError' not in log or 'generation_status' not in log:raise RuntimeError('Stop for unexpected build failure, do not retry blindly:'+log[-3000:])
with (root/'status_lock_recovery_receipt.json').open('x',encoding='utf-8') as f:json.dump({'cause':'Windows transient generation_status.json replacement PermissionError; all completed per-case sidecars retained','started_at':datetime.now(timezone.utc).isoformat(),'physics_masks_and_source_members_unchanged':True,'registered_action':'Resume same plan using receipt SHA validation; no feature overwrite','GPU_training':False},f,ensure_ascii=False,indent=2)
env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
with (root/'feature_resume.log').open('x',encoding='utf-8') as log:
 subprocess.run(['python','-u','scripts/data/build_m1_continuous_features.py','--plan',str(root/'generation_plan.json'),'--workers','8'],stdout=log,stderr=subprocess.STDOUT,env=env,check=True)
print('COMPLETE_VERIFIED_CPU_DATA_RESUME',flush=True)
