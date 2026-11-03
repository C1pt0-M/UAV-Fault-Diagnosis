"""Launch only the eight authorized M3 jobs after accepted fixed-anchor gate."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,json,os,subprocess,sys,time,traceback
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/research'))
from train_causal_m3 import read,protocol,source


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for n in ['protocol','data','features','initial-anchor','gate','output']:ap.add_argument('--'+n,type=Path,required=True)
    a=ap.parse_args();p,e=protocol(a.protocol);gate=read(a.gate)
    if gate['status']!='PASS_M3_FIXED_ANCHOR_AND_ENTRYPOINTS' or gate['execution_sha256']!=source.sha(a.protocol):raise ValueError('Incomplete entry gates')
    if a.output.exists():raise FileExistsError('Do not restart any formal M3 batch')
    a.output.mkdir(parents=True,exist_ok=False)
    (a.output/'execution_started.lock').write_text(str(os.getpid()))
    capacity=max(v['batch512_peak_bytes'] for v in gate['models'].values())
    import torch
    free,total=torch.cuda.mem_get_info()
    concurrency=2 if capacity*2*1.8<free else 1
    env=dict(os.environ,PYTHONPATH=str(ROOT/'packages/uav_fault'),CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
    pending=list(e['jobs']);active={};completed=[];handles={};failed=False
    state=dict(status='RUNNING',concurrency=concurrency,gate_batch512_peak_bytes=capacity,free_GPU_bytes=free,started_at=datetime.now(timezone.utc).isoformat(),completed=[],active={},pending=pending.copy(),automatic_retries=False)
    def save():
        state.update(updated_at=datetime.now(timezone.utc).isoformat(),completed=completed.copy(),active={v:proc.pid for v,proc in active.items()},pending=pending.copy())
        source.write(a.output/'status.json',state)
    try:
        while pending or active:
            while pending and len(active)<concurrency:
                variant=pending.pop(0);out=a.output/'runs'/variant;out.parent.mkdir(exist_ok=True)
                log=(a.output/(variant+'.log')).open('x');handles[variant]=log
                cmd=[sys.executable,'-u',str(ROOT/'scripts/research/train_causal_m3.py'),'--protocol',str(a.protocol),'--data',str(a.data),'--features',str(a.features),'--initial-anchor',str(a.initial_anchor),'--gate',str(a.gate),'--output',str(out),'--variant',variant]
                active[variant]=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT);save()
            for variant,proc in list(active.items()):
                rc=proc.poll()
                if rc is None:continue
                handles[variant].close();del active[variant]
                if rc!=0:
                    failed=True;raise RuntimeError('Formal job failed '+variant+' exit '+str(rc))
                report=read(a.output/'runs'/variant/'report.json')
                if report['status']!='COMPLETE_FORMAL_M3':raise ValueError('Wrong job completion '+variant)
                completed.append(variant);save()
            time.sleep(3)
        state['status']='COMPLETE_EIGHT_FORMAL_M3_JOBS';save()
    except Exception as exc:
        for variant,proc in active.items():
            proc.terminate()
        state.update(status='FAILED_STOPPED_NO_RETRY',error=repr(exc),traceback=traceback.format_exc());save();raise


if __name__=='__main__':main()
