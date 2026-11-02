"""Complete existing four M1 results; adds no optimizer or training invocation."""
from pathlib import Path
from datetime import datetime, timezone
import json
import os
import subprocess
import traceback
BASE = Path('/root/autodl-tmp/causal_m1_continuous_v1_20261102')
CODE = BASE / 'code'
PY = '/root/miniconda3/bin/python'
ENV = dict(os.environ, PYTHONPATH=str(CODE/'packages/uav_fault'), OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', CUBLAS_WORKSPACE_CONFIG=':4096:8')


def main():
    with (BASE/'postprocessing_recovery_started.lock').open('x') as f:
        f.write(str(os.getpid()))
    original = json.loads((BASE/'remote_status.json').read_text())
    assert original['phase'] == 'ALL_FORMAL_JOBS_COMPLETE_VERIFY_PENDING'
    assert len(original['completed_jobs']) == 4
    state = {'status': 'VERIFYING_EXISTING_TRAINING', 'started_at': datetime.now(timezone.utc).isoformat(), 'training_calls': 0, 'original_failure_preserved': True}
    def save():
        temporary = BASE/'recovered_postrun_status.partial.json'
        temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding='utf-8')
        temporary.replace(BASE/'recovered_postrun_status.json')
    def run(name, command):
        state['status'] = name
        save()
        with (BASE/(name.lower()+'.log')).open('x') as log:
            subprocess.run([PY, '-u', *map(str, command)], cwd=CODE, env=ENV, stdout=log, stderr=subprocess.STDOUT, check=True)
    try:
        run('VERIFYING_EXISTING_TRAINING', [CODE/'scripts/research/verify_m1_training_results.py', '--protocol', CODE/'configs/experiments/causal_m1_continuous150_v1.json', '--data', '/root/autodl-tmp/causal_source_v1_20261031/unpacked/data', '--features', BASE/'features', '--runs', BASE/'runs', '--output', BASE/'verified_training'])
        common = ['--plan', BASE/'replay_plan.json', '--data', BASE/'replay_base', '--features', BASE/'features', '--runs', BASE/'runs']
        run('CONTINUOUS_FIXED18_REPLAY', [CODE/'scripts/research/replay_m1_continuous.py', *common, '--output', BASE/'continuous_replay', '--device', 'cuda:0'])
        run('NATURAL_M0_AVAILABILITY_REPLAY', [CODE/'scripts/research/replay_m1_natural_availability.py', *common, '--main-replay', BASE/'continuous_replay', '--output', BASE/'natural_availability_replay', '--device', 'cuda:0'])
        state.update(status='COMPLETE_TRAINING_VERIFICATION_AND_BOTH_REPLAYS', completed_at=datetime.now(timezone.utc).isoformat())
        save()
    except Exception as exc:
        state.update(status='FAILED_POSTPROCESSING_RECOVERY', error=repr(exc), traceback=traceback.format_exc())
        save()
        raise


if __name__ == '__main__':
    main()
