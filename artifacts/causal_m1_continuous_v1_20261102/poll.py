"""Read-only progress for registered M1 sequence and post-training replay."""
from pathlib import Path
import subprocess
import sys
REMOTE=r'''
from pathlib import Path
from datetime import datetime,timezone
import json,subprocess,os
base=Path('/root/autodl-tmp/causal_m1_continuous_v1_20261102')
result={'polled_at':datetime.now(timezone.utc).isoformat(),'remote_root':str(base)}
for name in ['remote_status.json','postrun_status.json','recovered_postrun_status.json','corrected_replays_status.json','verified_training/summary.json','verified_training/archive_receipt.json']:
 p=base/name
 if p.exists():result[name]=json.loads(p.read_text())
replay_state=result.get('corrected_replays_status.json',{})
verification=result.get('verified_training/summary.json',{})
if replay_state.get('status')=='COMPLETE_CORRECTED_CONTINUOUS_AND_NATURAL_REPLAYS' and verification.get('status')=='PASS_REGISTERED_M1_AND_MATCHED_SOURCE_RESULTS':
 result['current_status']='COMPLETE_FOUR_FORMAL_JOBS_AND_BOTH_REPLAYS'
 result['historical_failure_note']='postrun_status/recovered_postrun_status are preserved earlier failures. Corrected replay completion and verified results supersede them; no formal training rerun.'
else:
 result['current_status']=replay_state.get('status') or result.get('recovered_postrun_status.json',{}).get('status') or result.get('remote_status.json',{}).get('phase')
result['runs']={}
for path in (base/'runs').glob('*/*/report.json'):
 r=json.loads(path.read_text());name=str(path.parent.relative_to(base/'runs'));row={k:r.get(k) for k in ['status','variant','arm','epochs','best_epoch','gradient_steps','elapsed_s','model_parameters','started_at','finished_at']}
 h=path.parent/'history.json'
 if h.exists():
  history=json.loads(h.read_text());row['completed_epochs']=len(history)
  if history:row['last_epoch']={k:history[-1][k] for k in ['epoch','lr','train_loss','gradient_steps','selection']}
 result['runs'][name]=row
result['logs']={}
for name in ['formal_runner.log','postrun_runner.log','training_verification.log','continuous_replay.log','verifying_existing_training.log','corrected_replay_runner.log','continuous_replay_v2.log','natural_availability_replay_v2.log']:
 p=base/name
 if p.exists():result['logs'][name]=p.read_text()[-4500:]
p=subprocess.run(['nvidia-smi','--query-gpu=name,memory.used,utilization.gpu','--format=csv,noheader'],capture_output=True,text=True);result['gpu']=p.stdout.strip()
print(json.dumps(result,ensure_ascii=False,indent=2))
'''
if __name__=='__main__':
 r=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','autodl_UAV3','/root/miniconda3/bin/python -'],input=REMOTE,capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=65)
 print(r.stdout,end='')
 if r.stderr:print(r.stderr,file=sys.stderr,end='')
 raise SystemExit(r.returncode)
