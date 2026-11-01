"""Execute only the two authorized corrected-budget runs; no retries."""
from pathlib import Path
from datetime import datetime, timezone
import json
import hashlib
import os
import subprocess
import sys
import traceback

BASE = Path('/root/autodl-tmp/causal_source_longbudget_v1_20261101')
ORIGINAL = Path('/root/autodl-tmp/causal_source_v1_20261031')
CODE = ORIGINAL / 'unpacked/code'
DATA = ORIGINAL / 'unpacked/data'
PROTOCOL = BASE / 'corrected_longbudget_protocol.json'
PYTHON = '/root/miniconda3/bin/python'
EXPECTED = '2d92fc2ffa2dc28050000721e44fb5509a45e67eae9269ede59df4d557757acd'
STATUS = BASE / 'remote_status.json'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def save(path, record):
    partial = path.with_name(path.name + '.partial')
    partial.write_text(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    partial.replace(path)


def update(phase, **fields):
    status.update(phase=phase, updated_at=datetime.now(timezone.utc).isoformat(), **fields)
    save(STATUS, status)


status = {'phase': 'STARTING', 'pid': os.getpid(), 'started_at': datetime.now(timezone.utc).isoformat(),
    'source_only': True, 'new_training_jobs_limit': 2, 'automatic_retries': False,
    'test_or_Real_evaluated': False, 'deployment_modified': False}
try:
    if STATUS.exists() or (BASE / 'runs').exists():
        raise FileExistsError('Preserve all prior executions; no automatic restart')
    update('VERIFY_NEW_PROTOCOL_AND_ORIGINAL_ASSETS')
    if sha(PROTOCOL) != EXPECTED or PROTOCOL.with_suffix('.json.sha256').read_text().strip() != EXPECTED:
        raise ValueError('Corrected protocol hash mismatch')
    p = json.loads(PROTOCOL.read_text())
    old_path = CODE / 'configs/experiments/causal_source_development_v1.json'
    old_hash = sha(old_path)
    old = json.loads(old_path.read_text())
    if p['budget_revision']['baseline_protocol_sha256'] != old_hash:
        raise ValueError('Not the original execution protocol')
    for k in ['data', 'data_identities', 'code_identities', 'sampling', 'loss', 'model', 'formal_environment', 'seeds', 'arms']:
        if p[k] != old[k]:
            raise ValueError('Unregistered difference: ' + k)
    if p['training']['max_epochs'] != 100 or p['training']['warmup_epochs'] != 5:
        raise ValueError('Incorrect original long training horizon')
    if p['validation']['early_stop'] != dict(old['validation']['early_stop'], min_epochs=30, patience=20):
        raise ValueError('Unregistered early stopping changes')
    if sorted(f.name for f in (DATA / 'cache').iterdir()) != ['train.pt', 'val.pt']:
        raise ValueError('Unexpected payloads; final tests must not be uploaded')
    env = dict(os.environ, CUBLAS_WORKSPACE_CONFIG=':4096:8', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
        PYTHONPATH=str(CODE / 'packages/uav_fault'))
    trainer = [PYTHON, '-u', str(CODE / 'scripts/research/train_causal_source_v1.py')]
    update('CORRECTED_BUDGET_SOURCE_PREFLIGHT')
    with (BASE / 'preflight.log').open('x') as log:
        subprocess.run(trainer + ['preflight', '--protocol', str(PROTOCOL), '--data', str(DATA), '--output', str(BASE / 'preflight')],
            cwd=CODE, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    (BASE / 'runs').mkdir(exist_ok=False)
    records = {}
    for arm in p['arms']:
        update('FORMAL_CORRECTED_BUDGET_TRAINING', current_arm=arm, completed_arms=list(records))
        with (BASE / (arm + '.log')).open('x') as log:
            subprocess.run(trainer + ['run', '--protocol', str(PROTOCOL), '--data', str(DATA), '--output', str(BASE / 'runs' / arm),
                '--arm', arm, '--seed', '42'], cwd=CODE, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        r = json.loads((BASE / 'runs' / arm / 'report.json').read_text())
        previous = json.loads((ORIGINAL / 'runs' / arm / 'report.json').read_text())
        if r['status'] != 'COMPLETE_FINITE_SOURCE_RETRAIN' or r['cpu_check'] or r['protocol_sha256'] != EXPECTED:
            raise ValueError('Formal job identity/completion mismatch')
        if r['initial_state_digest'] != previous['initial_state_digest']:
            raise ValueError('Random initialization differs from corresponding short run')
        records[arm] = r
        if sha(old_path) != old_hash:
            raise ValueError('Original protocol changed')
    save(BASE / 'paired_source_summary.json', {'status': 'COMPLETE_TWO_CORRECTED_BUDGET_RUNS', 'seed': 42,
        'protocol_sha256': EXPECTED, 'arms': records, 'same_initialization_as_short_budget': True,
        'test_or_Real_evaluated': False, 'deployment_modified': False})
    update('COMPLETED_TWO_ARM_FORMAL_SEQUENCE', completed_arms=list(records), current_arm=None)
    with (BASE / 'verification.log').open('x') as log:
        subprocess.run([PYTHON, '-u', str(BASE / 'verify_and_package.py')], cwd=BASE, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    save(BASE / 'postrun_verification_status.json', {'status': 'PASS_RESULTS_VERIFIED', 'finished_at': datetime.now(timezone.utc).isoformat(),
        'archive_receipt': str(BASE / 'verified_source_results_receipt.json'), 'no_additional_training': True})
except Exception as exc:
    update('FAILED_STOPPED_NO_AUTOMATIC_RETRY', error=repr(exc), traceback=traceback.format_exc())
    raise
