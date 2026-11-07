"""T preflight. --local never connects remotely, runs CUDA, or opens val files."""
from __future__ import annotations

import argparse
from pathlib import Path
import os
import socket
import subprocess
import sys
from types import SimpleNamespace
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/research'))
import train_t_selection as train
from train_t_selection import s
import numpy as np


def check_cpu_smoke(root=None):
    root = train.SMOKE_ROOT if root is None else Path(root)
    if not root.resolve().is_relative_to(train.SMOKE_ROOT.resolve()):
        raise ValueError('CPU smoke must stay under T smoke root')
    report = s.read_json(root / 'report.json')
    if (report.get('status') != 'COMPLETE_CPU_SMOKE_NOT_FORMAL'
            or report.get('protocol_sha256') != train.PROTOCOL_SHA
            or report['epochs'] != 2 or report['seed'] != 42 or report['fold'] != 0):
        raise ValueError('Missing successful local two-epoch smoke')
    if report['used_window_counts']['fit'] > 2048:
        raise ValueError('Smoke exceeds registered fit limit')
    if report['implementation_files'] != train.implementation_manifest()['files']:
        raise ValueError('CPU smoke does not match current frozen T implementation')
    s.read_json(root / 'selections.json')
    for role in ('inner', 'heldout'):
        with np.load(root / f'{role}_predictions.npz', allow_pickle=False) as data:
            if data['probabilities'].dtype != np.float32 or data['probabilities'].shape != (2, report['used_window_counts'][role]):
                raise ValueError('Smoke prediction shape/dtype mismatch')
    return {'status': 'PASS', 'root': str(root.resolve()), 'report_sha256': s.sha(root / 'report.json')}


def preflight(args):
    output = Path(args.output)
    if output.exists():
        raise FileExistsError('Fresh preflight output required')
    if args.local and not output.resolve().is_relative_to(train.OUTPUT_ROOT.resolve()):
        raise ValueError('Local preflight must stay under T artifact root')
    actual = train.environment()
    if not args.local:
        train.require_formal_environment(actual)
    _, pins = train.load_protocol(args.protocol)
    output.mkdir(parents=True, exist_ok=False)
    report = {'status': 'RUNNING', 'formal_clearance': False, 'local': args.local,
        'protocol_sha256': train.PROTOCOL_SHA, 'code_sha256': pins, 'environment': actual,
        'hostname': socket.gethostname(), 'remote_connection': 'OPERATOR_SIDE_ONLY_NOT_RUN',
        'nvidia_smi': 'NOT_RUN', 'gpu_determinism': 'NOT_RUN'}
    try:
        report['data_identities'] = train.verify_data(args.data, local=args.local)
        report['fold_sizes'] = s.fold_sizes(s.read_ledger(args.data), s.read_json(s.DEFAULT_PROTOCOL))
        s.write_json(output / 'implementation_manifest.json', train.implementation_manifest())
        manifest_sha = s.sha(output / 'implementation_manifest.json')
        report['implementation_manifest_sha256'] = manifest_sha
        tests = output / 'tests'
        tests.mkdir()
        scope = tests / 'scope'
        scope.mkdir()
        env = dict(os.environ, PYTEST_DISABLE_PLUGIN_AUTOLOAD='1', PYTHONDONTWRITEBYTECODE='1')
        if args.local:
            env['CUDA_VISIBLE_DEVICES'] = ''
        # Path-only test isolation activates the inherited basetemp guards
        # without writing any S artifacts or changing any frozen source file.
        bootstrap = (
            'import sys; from pathlib import Path; '
            f'sys.path.insert(0, {str(ROOT / "scripts/research")!r}); '
            'import train_s_ablation, build_pure_m0_anchor; '
            f'train_s_ablation.SMOKE_ROOT = Path({str(scope)!r}); '
            f'build_pure_m0_anchor.OUTPUT_ROOT = Path({str(scope)!r}); '
            'import pytest; raise SystemExit(pytest.main(sys.argv[1:]))'
        )
        command = [sys.executable, '-I', '-B', '-c', bootstrap, '-q', '-p', 'no:cacheprovider',
            '--basetemp=' + str(scope / 'tmp'), '--junitxml=' + str(tests / 'junit.xml'),
            str(ROOT / 'tests/test_mstca_ablation_factory.py'), str(ROOT / 'tests/test_s_ablation.py'),
            str(ROOT / 'tests/test_t_selection.py')]
        result = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True)
        with (tests / 'output.txt').open('x', encoding='utf-8') as stream:
            stream.write(result.stdout + '\n' + result.stderr)
        if result.returncode:
            raise RuntimeError('Factory/T/golden tests failed: STOP, never regenerate goldens')
        cases = list(ET.parse(tests / 'junit.xml').getroot().iter('testcase'))
        skipped = [case for case in cases if case.find('skipped') is not None]
        if any(case.find('skipped').get('type') != 'pytest.xfail' for case in skipped):
            raise RuntimeError('Preflight requires zero skipped tests, including basetemp cases')
        report['pytest_summary'] = {'passed': len(cases) - len(skipped), 'xfailed': len(skipped), 'skipped': 0}
        golden = [case for case in cases if case.get('name', '').startswith(('test_full_state_digest_golden[', 'test_new_golden['))]
        if len(golden) != 14 or any(case.find(tag) is not None for case in golden for tag in ('failure', 'error', 'skipped')):
            raise RuntimeError('All 12 old and 2 new goldens must actually pass')
        report['golden_tests_passed'] = 14
        smoke_root = getattr(args, 'cpu_smoke_root', train.SMOKE_ROOT)
        report['cpu_smoke'] = check_cpu_smoke(smoke_root)
        if args.local:
            report.update(status='LOCAL_GATES_PASS_FORMAL_NOT_AUTHORIZED', formal_gates_not_run=[
                'registered CUDA environment', 'three val identity hashes', 'committed CPU smoke',
                'formal host nvidia-smi', 'two GPU determinism probes'])
        else:
            # This is a repository-state assertion only: NEVER commit from this tool.
            files = [str((Path(smoke_root) / name).relative_to(ROOT)) for name in
                     ('report.json', 'selections.json', 'inner_predictions.npz', 'heldout_predictions.npz')]
            tracked = subprocess.run(['git', 'ls-files', '--error-unmatch', '--', *files],
                                     cwd=ROOT, capture_output=True, text=True)
            clean = subprocess.run(['git', 'diff', '--exit-code', 'HEAD', '--', *files],
                                   cwd=ROOT, capture_output=True, text=True)
            if tracked.returncode or clean.returncode:
                raise RuntimeError('Frozen preflight requires unchanged committed smoke; tool never commits')
            evidence = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=30)
            report['nvidia_smi'] = {'returncode': evidence.returncode, 'stdout': evidence.stdout, 'stderr': evidence.stderr}
            if evidence.returncode:
                raise RuntimeError('nvidia-smi host evidence failed')
            query = subprocess.run(['nvidia-smi', '--query-gpu=name,driver_version', '--format=csv,noheader'],
                                   capture_output=True, text=True, timeout=30)
            report['gpu'] = {'returncode': query.returncode, 'model_driver_csv': query.stdout, 'stderr': query.stderr}
            if query.returncode:
                raise RuntimeError('nvidia-smi GPU query failed')
            digests = []
            for repeat in (1, 2):
                run_args = SimpleNamespace(protocol=args.protocol, data=args.data, seed=42, fold=0,
                    phase='cv', cpu_smoke=False, output=output / f'gpu_probe_{repeat}',
                    implementation_manifest=output / 'implementation_manifest.json',
                    implementation_manifest_sha256=manifest_sha)
                run = train.run(run_args, probe_epochs=2)
                digests.append(run['checkpoint_state_digests'])
            report['gpu_determinism'] = {'state_digests': digests, 'identical': digests[0] == digests[1]}
            if digests[0] != digests[1]:
                raise RuntimeError('GPU determinism mismatch: STOP')
            report.update(status='ALL_PREFLIGHT_GATES_PASS', formal_clearance=True)
        s.write_json(output / 'preflight.json', report)
        return report
    except Exception as exc:
        report.update(status='BLOCKED', error=repr(exc), formal_clearance=False)
        s.write_json(output / 'preflight.json', report)
        raise


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--protocol', type=Path, default=train.DEFAULT_PROTOCOL)
    result.add_argument('--data', type=Path, default=Path('D:/UAV-Data/causal_input_v1'))
    result.add_argument('--output', type=Path, required=True)
    result.add_argument('--local', action='store_true')
    result.add_argument('--cpu-smoke-root', type=Path, default=train.SMOKE_ROOT,
                        help='Successful smoke directory under the T smoke root; old reports are never overwritten')
    return result


if __name__ == '__main__':
    print(preflight(parser().parse_args())['status'])
