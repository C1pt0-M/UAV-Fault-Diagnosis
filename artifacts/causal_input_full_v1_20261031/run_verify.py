from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path('D:/UAV-Fault-Diagnosis');ART=ROOT/'artifacts/causal_input_full_v1_20261031';OUT=Path('D:/UAV-Data/causal_input_v1')

def run(split):
    for _ in range(480):
        if (ART/'BUILD_FAILED.json').exists():return {'split':split,'status':'build_failed'}
        if (OUT/(split+'_cache_complete.json')).exists():break
        time.sleep(5)
    else:return {'split':split,'status':'build_not_complete'}
    with (ART/('verify_'+split+'.log')).open('x',encoding='utf-8') as f:
        code=subprocess.run([sys.executable,'-u','scripts/research/verify_full_causal_input.py','--stage','split','--split',split],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT).returncode
    result={'split':split,'returncode':code}
    (ART/('verify_'+split+'_execution.json')).write_text(json.dumps(result),encoding='utf-8')
    return result

with ThreadPoolExecutor(max_workers=2) as pool:
    results=list(pool.map(run,['train','val','test','real']))
(ART/'verification_execution.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
if any(r.get('returncode',1) for r in results):sys.exit(1)
for stage in ['norm','parity']:
    with (ART/('verify_'+stage+'.log')).open('x',encoding='utf-8') as f:
        code=subprocess.run([sys.executable,'-u','scripts/research/verify_full_causal_input.py','--stage',stage],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT).returncode
    (ART/('verify_'+stage+'_execution.json')).write_text(json.dumps({'returncode':code}),encoding='utf-8')
    if code:sys.exit(code)
(ART/'ALL_VERIFIED.json').write_text(json.dumps({'status':'PASS_ALL_SPLITS_NORM_PARITY'}),encoding='utf-8')
