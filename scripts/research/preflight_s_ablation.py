"""S-ablation gates. --local never connects remotely, uses CUDA or parses val.

Formal use (future server): assert environment first, run pinned golden tests,
verify all nine data hashes/folds, record host/nvidia-smi evidence and a pre-run
implementation manifest, then two fresh S0/seed42/fold0 two-epoch GPU probes.
SSH login is operator-side only. No golden regeneration.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
from types import SimpleNamespace
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/research'))
import train_s_ablation as train


def preflight(args):
    output=Path(args.output)
    if output.exists():
        raise FileExistsError('Fresh preflight output required')
    if args.local and not output.resolve().is_relative_to(train.SMOKE_ROOT.resolve()):
        raise ValueError('Local preflight output must be under smoke artifact root')
    actual=train.environment()
    if not args.local:
        train.require_formal_environment(actual)
    p,pins=train.load_protocol(args.protocol)
    output.mkdir(parents=True,exist_ok=False)
    report={'status':'RUNNING_PREFLIGHT','local':args.local,'formal_clearance':False,
            'protocol_sha256':train.PROTOCOL_SHA,'code_sha256':pins,'environment':actual,
            'hostname':socket.gethostname(),'remote_connection':'OPERATOR_SIDE_ONLY_NOT_RUN',
            'nvidia_smi':'NOT_RUN','gpu_determinism':'NOT_RUN','val_access':'HASH_BYTES_ONLY'}
    try:
        report['data_identities']=train.verify_data(args.data,smoke=args.local)
        report['fold_sizes']=train.fold_sizes(train.read_ledger(args.data),p)
        manifest=train.implementation_manifest()
        train.write_json(output/'implementation_manifest.json',manifest)
        report['implementation_manifest_sha256']=train.sha(output/'implementation_manifest.json')
        tests=output/'golden_tests'
        tests.mkdir(exist_ok=False)
        env=dict(os.environ,PYTEST_DISABLE_PLUGIN_AUTOLOAD='1')
        if args.local:
            env['CUDA_VISIBLE_DEVICES']=''
        command=[sys.executable,'-I','-B','-m','pytest','-q','-p','no:cacheprovider',
            '--basetemp='+str(tests/'tmp'),'--junitxml='+str(tests/'junit.xml'),
            str(ROOT/'tests/test_mstca_ablation_factory.py')]
        result=subprocess.run(command,cwd=ROOT,env=env,capture_output=True,text=True)
        with (tests/'output.txt').open('x',encoding='utf-8') as stream:
            stream.write(result.stdout+'\n'+result.stderr)
        report['golden_test_returncode']=result.returncode
        if result.returncode:
            raise RuntimeError('Factory/golden tests failed: STOP; never regenerate goldens')
        cases=list(ET.parse(tests/'junit.xml').getroot().iter('testcase'))
        goldens=[c for c in cases if c.get('name','').startswith('test_full_state_digest_golden[')]
        if len(goldens)!=12 or any(c.find(tag) is not None for c in goldens for tag in ('failure','error','skipped')):
            raise RuntimeError('All 12 golden tests must actually pass, not skip or xfail')
        report['golden_tests_passed']=12
        report['cpu_smoke']={}
        for arm in p['arms']:
            path=train.SMOKE_ROOT/'runs'/f'{arm}_seed42_fold0'/'report.json'
            r=train.read_json(path) if path.exists() else {}
            report['cpu_smoke'][arm]=r.get('status')=='COMPLETE_CPU_SMOKE_NOT_FORMAL' and r.get('epochs')==2
        if args.local:
            report['status']='LOCAL_GATES_PASS_FORMAL_NOT_AUTHORIZED'
            report['formal_gates_not_run']=['formal host nvidia-smi evidence','registered CUDA environment','GPU deterministic two-run check','full train.pt SHA']
        else:
            # Run locally on the formal host. The full banner records the
            # driver-supported CUDA version; the query preserves full GPU names.
            smi=subprocess.run(['nvidia-smi'],capture_output=True,text=True,timeout=30)
            report['nvidia_smi']={'returncode':smi.returncode,'stdout':smi.stdout,'stderr':smi.stderr}
            if smi.returncode:
                raise RuntimeError('nvidia-smi host evidence failed')
            query=subprocess.run(['nvidia-smi','--query-gpu=name,driver_version','--format=csv,noheader'],
                capture_output=True,text=True,timeout=30)
            report['gpu']={'returncode':query.returncode,'model_driver_csv':query.stdout,
                           'stderr':query.stderr,'torch_cuda':actual['torch_cuda']}
            if query.returncode:
                raise RuntimeError('nvidia-smi GPU/driver query failed')
            digests=[]
            for repeat in (1,2):
                run_args=SimpleNamespace(protocol=args.protocol,data=args.data,arm='S0',seed=42,fold=0,phase='cv',
                    output=output/f'gpu_probe_{repeat}',cpu_smoke=False,
                    implementation_manifest=output/'implementation_manifest.json',
                    implementation_manifest_sha256=report['implementation_manifest_sha256'],
                    phase1_verdicts=None,phase1_verdicts_sha256=None)
                r=train.run(run_args,probe_epochs=2)
                digests.append(r['checkpoint_state_digests']['last'])
            report['gpu_determinism']={'state_digests':digests,'identical':digests[0]==digests[1]}
            if digests[0]!=digests[1] or not all(report['cpu_smoke'].values()):
                raise RuntimeError('Determinism or prior CPU smoke gate failed')
            report.update(status='ALL_PREFLIGHT_GATES_PASS',formal_clearance=True)
        train.write_json(output/'preflight.json',report)
        return report
    except Exception as exc:
        report.update(status='BLOCKED',error=repr(exc),formal_clearance=False)
        train.write_json(output/'preflight.json',report)
        raise


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--protocol',type=Path,default=train.DEFAULT_PROTOCOL)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--local',action='store_true')
    return p


if __name__=='__main__':
    result=preflight(parser().parse_args())
    print(json.dumps({'status':result['status'],'formal_clearance':result['formal_clearance']},indent=2))
