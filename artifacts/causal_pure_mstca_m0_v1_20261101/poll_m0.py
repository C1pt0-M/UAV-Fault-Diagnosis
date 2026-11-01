"""Read-only M0 progress monitor; never launches/retries training."""
from pathlib import Path
import subprocess
import sys

REMOTE = r'''
from pathlib import Path
from datetime import datetime,timezone
import json,os,subprocess
base=Path('/root/autodl-tmp/causal_pure_mstca_m0_v1_20261102')
result={'polled_at':datetime.now(timezone.utc).isoformat(),'ssh_alias':'autodl_UAV3'}
status=json.loads((base/'remote_status.json').read_text())
result['status']=status
try:
 os.kill(status['pid'],0);result['runner_alive']=True
except ProcessLookupError:result['runner_alive']=False
result['arms']={}
for arm in ['uniform_replacement','balanced_source_normal']:
 root=base/'runs'/arm;entry={'started':(root/'report.json').exists()}
 if entry['started']:
  r=json.loads((root/'report.json').read_text())
  entry['report']={k:r.get(k) for k in ['status','model','parameters','source_train_windows','source_val_windows_full','started_at','finished_at','epochs','best_epoch','gradient_steps','elapsed_s','source_validation']}
 if (root/'history.json').exists():
  h=json.loads((root/'history.json').read_text());entry['completed_epochs']=len(h)
  if h:entry['last_epoch']={k:h[-1][k] for k in ['epoch','lr','train_loss','gradient_steps','selection']}
 path=base/(arm+'.log');entry['log_tail']=path.read_text()[-2500:] if path.exists() else None
 result['arms'][arm]=entry
for name in ['verified_m0_source_summary.json','verified_m0_results_receipt.json']:
 if (base/name).exists():result[name]=json.loads((base/name).read_text())
result['runner_log_tail']=(base/'runner.log').read_text()[-4000:]
p=subprocess.run(['nvidia-smi','--query-gpu=name,memory.used,utilization.gpu','--format=csv,noheader'],capture_output=True,text=True)
result['gpu']=p.stdout.strip()
print(json.dumps(result,ensure_ascii=False,indent=2))
'''

if __name__ == '__main__':
    result=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','autodl_UAV3',
        '/root/miniconda3/bin/python -'],input=REMOTE,capture_output=True,text=True,
        encoding='utf-8',errors='replace',timeout=65)
    print(result.stdout,end='')
    if result.stderr:
        print(result.stderr,file=sys.stderr,end='')
    raise SystemExit(result.returncode)
