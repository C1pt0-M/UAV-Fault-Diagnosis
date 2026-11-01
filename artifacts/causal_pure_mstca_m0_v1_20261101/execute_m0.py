"""Run exactly the two explicitly authorized M0 jobs, no automatic retries."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import os
import subprocess
import sys
import traceback

BASE = Path('/root/autodl-tmp/causal_pure_mstca_m0_v1_20261102')
CODE = BASE / 'code'
DATA = Path('/root/autodl-tmp/causal_source_v1_20261031/unpacked/data')
ANCHOR = Path('/root/autodl-tmp/causal_source_longbudget_v1_20261101/runs/uniform_replacement/initial.pt')
PROTOCOL = CODE / 'configs/experiments/causal_pure_mstca_m0_v1.json'
EXPECTED = '6ef47aead15fea51e4c7d4b53289383ecd038d67b82e3cae8c07ddcd8b450fe6'
PYTHON = '/root/miniconda3/bin/python'
STATUS = BASE / 'remote_status.json'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def save(path, value):
    partial = path.with_name(path.name + '.partial')
    partial.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    partial.replace(path)


def main():
    # An exclusive marker also protects against duplicate confirmation/launch.
    with (BASE / 'execution_started.lock').open('x') as stream:
        stream.write(str(os.getpid()))
    if STATUS.exists() or (BASE / 'runs').exists():
        raise FileExistsError('Never overwrite or retry prior M0 execution')
    status = {'phase': 'STARTING_AUTHORIZED_M0_SEQUENCE', 'pid': os.getpid(),
        'started_at': datetime.now(timezone.utc).isoformat(), 'ssh_alias': 'autodl_UAV3',
        'authorized_formal_jobs': 2, 'automatic_retries': False, 'test_or_Real_evaluated': False,
        'deployment_modified': False, 'completed_arms': [], 'current_arm': None}
    def update(phase, **fields):
        status.update(phase=phase, updated_at=datetime.now(timezone.utc).isoformat(), **fields)
        save(STATUS, status)
    try:
        update('VERIFY_AUTHORIZATION_AND_REGISTERED_M0')
        auth = json.loads((BASE / 'authorization.json').read_text())
        if auth['authorized_action'] != 'RUN_EXACTLY_TWO_FORMAL_M0_SOURCE_JOBS' or auth['protocol_sha256'] != EXPECTED:
            raise ValueError('Missing or inconsistent formal authorization')
        ready = json.loads((BASE / 'remote_preparation_verification.json').read_text())
        if ready['status'] != 'PASS_REMOTE_M0_PREPARATION_ONLY_AWAITING_TRAINING_AUTHORIZATION':
            raise ValueError('Missing matching-environment preparation')
        if sha(PROTOCOL) != EXPECTED:
            raise ValueError('Do not alter frozen M0 preparation/training protocol')
        config = json.loads(PROTOCOL.read_text())
        if auth['arms'] != config['arms'] or auth['seeds'] != [42]:
            raise ValueError('Unregistered jobs')
        if sorted(p.name for p in (DATA / 'cache').iterdir()) != ['train.pt', 'val.pt']:
            raise ValueError('Source-only payload required')
        sys.path.insert(0, str(CODE / 'scripts/research'))
        import train_causal_mstca_m0 as m0
        m0.load_protocol(PROTOCOL)
        for key, expected in config['formal_environment'].items():
            if m0.source.environment()[key] != expected:
                raise ValueError('Formal environment changed: ' + key)
        if not m0.torch.cuda.is_available():
            raise RuntimeError('No CUDA; never CPU-fallback')
        for name, expected in config['code_identities'].items():
            if sha(CODE / name) != expected:
                raise ValueError('Pinned code changed: ' + name)
        env = dict(os.environ, PYTHONPATH=str(CODE / 'packages/uav_fault'),
            CUBLAS_WORKSPACE_CONFIG=':4096:8', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
        (BASE / 'runs').mkdir(exist_ok=False)
        records = {}
        for arm in config['arms']:
            update('FORMAL_M0_TRAINING', current_arm=arm, completed_arms=list(records))
            command = [PYTHON, '-u', str(CODE / 'scripts/research/train_causal_mstca_m0.py'),
                'run', '--protocol', str(PROTOCOL), '--data', str(DATA),
                '--output', str(BASE / 'runs' / arm), '--initial-anchor', str(ANCHOR),
                '--arm', arm, '--seed', '42', '--confirm-formal-run']
            with (BASE / (arm + '.log')).open('x') as log:
                subprocess.run(command, cwd=CODE, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
            report = json.loads((BASE / 'runs' / arm / 'report.json').read_text())
            if report['status'] != 'COMPLETE_FINITE_SOURCE_RETRAIN' or report['model'] != 'OnlineMSTCA':
                raise ValueError('M0 run did not complete correctly')
            if report['protocol_sha256'] != EXPECTED or report['initial_state_digest'] != config['initialization_anchor']['m0_mapped_state_digest']:
                raise ValueError('M0 protocol or initialization differs')
            records[arm] = report
        save(BASE / 'paired_m0_summary.json', {'status': 'COMPLETE_TWO_AUTHORIZED_M0_SOURCE_RUNS',
            'protocol_sha256': EXPECTED, 'seed': 42, 'arms': records, 'test_or_Real_evaluated': False})
        update('COMPLETED_TWO_ARM_FORMAL_SEQUENCE', current_arm=None, completed_arms=list(records))
        with (BASE / 'verification.log').open('x') as log:
            subprocess.run([PYTHON, '-u', str(BASE / 'verify_and_package_m0.py')], cwd=BASE,
                env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        update('COMPLETE_TWO_M0_JOBS_VERIFIED_AND_PACKAGED', current_arm=None)
    except Exception as exc:
        update('FAILED_STOPPED_NO_AUTOMATIC_RETRY', error=repr(exc), traceback=traceback.format_exc())
        raise


if __name__ == '__main__':
    main()
