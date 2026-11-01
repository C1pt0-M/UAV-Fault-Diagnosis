"""Run only the authorized frozen source-development jobs in fresh remote root."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib
import json
import os
import subprocess
import sys
import tarfile
import traceback

BASE=Path('/root/autodl-tmp/causal_source_v1_20261031')
PYTHON='/root/miniconda3/bin/python'
ARCHIVE_SHA='d835d736eee79a3d45c7d111d5f6e2e2890b4bc961184ebda373d3f444c433e4'
ARCHIVE_BYTES=956239401
EXPECTED_PROTOCOL='40cf51c9c711f06f1b35117f04b7d02fa7b6fd6c8b2a0606822c992320f8a9c0'
STATUS=BASE/'remote_status.json'

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for b in iter(lambda:stream.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def write(path,value):
    partial=path.with_name(path.name+'.partial')
    partial.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    partial.replace(path)

def update(phase,**fields):
    report.update(phase=phase,updated_at=datetime.now(timezone.utc).isoformat(),**fields)
    write(STATUS,report)

report={'phase':'STARTING','pid':os.getpid(),'started_at':datetime.now(timezone.utc).isoformat(),
        'source_only':True,'test_or_Real_payloads_present':False,'formal_training_started':False}
try:
    if STATUS.exists() or (BASE/'unpacked').exists() or (BASE/'runs').exists():
        raise FileExistsError('Refuse an existing launch or run; no automatic rerun')
    update('VERIFY_ARCHIVE')
    archive=BASE/'source_bundle.verified.tar.gz'
    if archive.stat().st_size!=ARCHIVE_BYTES or sha(archive)!=ARCHIVE_SHA:
        raise ValueError('Incomplete or changed transport archive; no extraction/training')
    out=BASE/'unpacked';out.mkdir(exist_ok=False)
    with tarfile.open(archive,'r:gz') as tar:
        members=tar.getmembers()
        if len(members)!=28 or any(m.name.startswith('/') or '..' in Path(m.name).parts or not m.isfile() for m in members):
            raise ValueError('Archive not expected file-only safe bundle')
        payloads=[m.name for m in members if m.name.endswith('.pt')]
        if set(payloads)!={'data/cache/train.pt','data/cache/val.pt'}:
            raise ValueError('Forbidden tensor or missing source input in bundle')
        tar.extractall(out,filter='data')
    subprocess.run(['sha256sum','--check','--strict','SHA256SUMS'],cwd=out,check=True,stdout=(BASE/'bundle_verification.txt').open('x'))
    manifest=json.loads((out/'bundle_manifest.json').read_text(encoding='utf-8'))
    for name,identity in manifest['identities'].items():
        path=out/name
        if path.stat().st_size!=identity['bytes'] or sha(path)!=identity['sha256']:
            raise ValueError('Manifest member identity mismatch: '+name)
    code=out/'code';data=out/'data';protocol=code/'configs/experiments/causal_source_development_v1.json'
    if sha(protocol)!=EXPECTED_PROTOCOL:raise ValueError('Execution protocol identity mismatch')
    # Only source allowlist is present; final-test/Real bytes never uploaded.
    if sorted(p.name for p in (data/'cache').iterdir())!=['train.pt','val.pt']:
        raise ValueError('Unexpected held-out inputs in remote development bundle')
    env=dict(os.environ,CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
    env['PYTHONPATH']=str(code/'packages/uav_fault')
    update('REMOTE_SOURCE_PREFLIGHT',archive_sha256=ARCHIVE_SHA,execution_protocol_sha256=EXPECTED_PROTOCOL)
    with (BASE/'preflight.log').open('x') as log:
        subprocess.run([PYTHON,'-u','scripts/research/train_causal_source_v1.py','preflight',
            '--protocol',str(protocol),'--data',str(data),'--output',str(BASE/'preflight')],cwd=code,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    # Synthetic device forward/backward/finite gradient check; no optimizer and no real validation scoring.
    synthetic='''import json,torch,numpy as np
from pathlib import Path
from uav_fault.models.cnn.lstm import OnlineKGMSTCA,RFLY_SEQ_COLUMNS_V2
from uav_fault.models.cnn.focal_loss import FocalLoss
torch.set_num_threads(4);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
torch.backends.cudnn.deterministic=True;torch.backends.cudnn.allow_tf32=True
torch.backends.cuda.matmul.allow_tf32=False
assert torch.cuda.is_available()
m=OnlineKGMSTCA(in_channels=26,hidden_size=64,num_layers=1,num_classes=2,dropout=.2,columns=list(RFLY_SEQ_COLUMNS_V2)).float().cuda()
x=torch.randn(32,26,150,device='cuda');y=torch.arange(32,device='cuda')%2
loss=FocalLoss(gamma=2)(m(x),y);assert torch.isfinite(loss)
loss.backward();norm=torch.nn.utils.clip_grad_norm_(m.parameters(),1.0,error_if_nonfinite=True)
torch.cuda.synchronize()
print(json.dumps(dict(status='PASS_SYNTHETIC_CUDA_FORWARD_BACKWARD_NOT_TRAINING',gpu=torch.cuda.get_device_name(0),torch=str(torch.__version__),numpy=np.__version__,torch_cuda=torch.version.cuda,loss=float(loss),grad_norm=float(norm),optimizer_updates=0,max_cuda_memory_bytes=torch.cuda.max_memory_allocated())))
'''
    update('SYNTHETIC_GPU_CHECK')
    test=subprocess.run([PYTHON,'-c',synthetic],cwd=code,env=env,text=True,capture_output=True,check=True)
    (BASE/'cuda_execution_check.json').write_text(test.stdout,encoding='utf-8')
    if test.stderr:(BASE/'cuda_execution_check.stderr').write_text(test.stderr,encoding='utf-8')
    update('FORMAL_TWO_ARM_SEQUENCE_RUNNING',formal_training_started=True,run_root=str(BASE/'runs'))
    with (BASE/'training_sequence.log').open('x') as log:
        command=['bash','scripts/experiments/run_causal_source_v1.sh',str(data),str(BASE/'runs')]
        subprocess.run(command,cwd=code,env=dict(env,PYTHON=PYTHON),stdout=log,stderr=subprocess.STDOUT,check=True)
    records={}
    for arm in ['uniform_replacement','balanced_source_normal']:
        row=json.loads((BASE/'runs'/arm/'report.json').read_text(encoding='utf-8'))
        if row['status']!='COMPLETE_FINITE_SOURCE_RETRAIN' or row['cpu_check'] or row['protocol_sha256']!=EXPECTED_PROTOCOL:
            raise ValueError('Formal registered job completion mismatch')
        if row['validation_windows']!=34710 or row['source_train_windows']!=162069:
            raise ValueError('Source validation population mismatch')
        if not row['checkpoint_reload_matches'] or not row['no_final_test_or_Real_reads']:
            raise ValueError('Formal checkpoint/scope check failed')
        records[arm]=row
    if records['uniform_replacement']['initial_state_digest']!=records['balanced_source_normal']['initial_state_digest']:
        raise ValueError('Paired initialization differs')
    summary={'status':'COMPLETE_TWO_FROZEN_SOURCE_DEVELOPMENT_RUNS','seed':42,'execution_protocol_sha256':EXPECTED_PROTOCOL,
        'same_initial_state':True,'test_or_Real_scored':False,'deployment_modified':False,
        'arms':{arm:{key:row[key] for key in ['epochs','best_epoch','gradient_steps','validation_windows','best_checkpoint_sha256','source_validation','elapsed_s']} for arm,row in records.items()},
        'interpretation':'Source validation only; both arms reported, one seed not significance or real-domain evidence. No automatic deployment or third run.'}
    write(BASE/'paired_source_summary.json',summary)
    update('COMPLETED_TWO_ARM_FORMAL_SEQUENCE',summary=str(BASE/'paired_source_summary.json'))
except Exception as exc:
    report.update(phase='FAILED_STOPPED_NO_AUTOMATIC_RETRY',error=repr(exc),traceback=traceback.format_exc(),updated_at=datetime.now(timezone.utc).isoformat())
    write(STATUS,report)
    raise
