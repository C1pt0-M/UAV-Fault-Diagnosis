"""Complete CPU-only M1 gates after sidecar delivery; no formal GPU launch."""
from pathlib import Path
from datetime import datetime,timezone
import argparse
import json
import os
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/research'))
from validate_kinematic_residuals import sha


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--protocol',type=Path,required=True);ap.add_argument('--data',type=Path,required=True);ap.add_argument('--features',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    if a.output.exists():raise FileExistsError('Preserve existing CPU gate execution')
    p=read(a.protocol);a.output.mkdir(parents=True,exist_ok=False);env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2');anchor=ROOT/'artifacts/causal_source_longbudget_v1_20261101/results/runs/uniform_replacement/initial.pt'
    checks={}
    try:
        # Actual sampled optimizer execution, tiny pipeline checks only; they do
        # not consume extra formal jobs or produce performance claims.
        for variant in ['M0_matched','M1']:
            dest=a.output/variant
            command=[sys.executable,'-u',str(ROOT/'scripts/research/train_causal_m1_continuous.py'),'--protocol',str(a.protocol),'--data',str(a.data),'--features',str(a.features),'--output',str(dest),'--initial-anchor',str(anchor),'--variant',variant,'--arm','uniform_replacement','--seed','42','--cpu-check']
            with (a.output/(variant+'.log')).open('x',encoding='utf-8') as log:subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
            r=read(dest/'report.json');assert r['status']=='PASS_CPU_SMALL_TRAINING_NOT_PERFORMANCE' and r['gradient_steps']==8
            checks[variant]={'status':r['status'],'gradient_steps':r['gradient_steps'],'report_sha256':sha(dest/'report.json'),'no_performance_claim':True}
        result={'status':'PASS_REAL_DATA_CPU_TRAINING_GATES','completed_at':datetime.now(timezone.utc).isoformat(),'protocol_sha256':sha(a.protocol),'checks':checks,'source_population':p['data_counts'],'formal_training_started':False,'test_Real_or_old_assets_modified':False}
    except Exception as exc:
        result={'status':'FAILED_STOP_FORMAL_TRAINING','error':repr(exc),'protocol_sha256':sha(a.protocol),'checks':checks,'formal_training_started':False}
        (a.output/'completion.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8');raise
    (a.output/'completion.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
