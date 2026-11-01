import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path('D:/UAV-Fault-Diagnosis')
ART=ROOT/'artifacts/causal_input_full_v1_20261031'
OUT=Path('D:/UAV-Data/causal_input_v1')

sequence=[('cases','train'),('cases','val'),('cache','train'),('cache','val'),
          ('cases','test'),('cache','test'),('cases','real'),('cache','real')]
results=[]
for stage,split in sequence:
    marker=OUT/(split+('_frames_complete.json' if stage=='cases' else '_cache_complete.json'))
    if marker.exists():
        results.append({'stage':stage,'split':split,'status':'already_completed'})
        continue
    log=ART/(stage+'_'+split+'.log')
    with log.open('a',encoding='utf-8') as stream:
        env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
        code=subprocess.run([sys.executable,'-u','scripts/data/build_full_causal_input.py',
            '--stage',stage,'--split',split,'--workers','4'],cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT).returncode
    result={'stage':stage,'split':split,'returncode':code}
    results.append(result)
    (ART/'execution_status.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
    if code:
        (ART/'BUILD_FAILED.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
        sys.exit(code)
(ART/'BUILD_STAGES_COMPLETE.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
