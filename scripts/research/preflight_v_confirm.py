"""V gates; --local is CPU/STAND_IN only. No GPU determinism probe in V."""
from __future__ import annotations

import argparse
from pathlib import Path
import os
import socket
import subprocess
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/research'))
import train_v_confirm as train
from train_v_confirm import s, t
import numpy as np


def check_cpu_smoke(root):
    root = Path(root)
    if not root.resolve().is_relative_to(train.SMOKE_ROOT.resolve()):
        raise ValueError('V smoke root must remain under V artifacts')
    report = s.read_json(root / 'report.json')
    if (report.get('status') != 'COMPLETE_CPU_SMOKE_STAND_IN_NOT_FORMAL'
            or report.get('prediction_scope') != 'STAND_IN' or report.get('protocol_sha256') != train.PROTOCOL_SHA
            or report.get('seed') != 42 or report.get('epochs') != 2 or report.get('val_prediction_passes') != 0
            or report.get('used_window_counts') != {'fit': 512, 'inner': 256}
            or report.get('confirmation_window_count') != 256):
        raise ValueError('Missing valid 2-epoch 512/256/256 STAND_IN smoke')
    if report['implementation_files'] != train.implementation_manifest()['files']:
        raise ValueError('V smoke implementation mismatch')
    selected = s.read_json(root / 'selections.json')
    if (selected != train.selections([row['inner_class_balanced_ce'] for row in report['history']])
            or set(report['prediction_artifacts']) != {str(selected['V0']['epoch'])}
            or report['stand_in_prediction_passes'] != 1):
        raise ValueError('V smoke selection or distinct-checkpoint pass mismatch')
    with np.load(root / 'inner_logits.npz', allow_pickle=False) as data:
        if data['logits'].shape != (2, 256, 2) or data['logits'].dtype != np.float32:
            raise ValueError('Wrong smoke inner logits')
    for item in report['prediction_artifacts'].values():
        path = root / item['file']
        if s.sha(path) != item['sha256']:
            raise ValueError('Smoke STAND_IN artifact hash mismatch')
        with np.load(path, allow_pickle=False) as data:
            if str(data['population_scope']) != 'STAND_IN' or data['logits'].shape != (256, 2) or data['logits'].dtype != np.float32:
                raise ValueError('Wrong smoke STAND_IN logits')
    return {'status': 'PASS', 'root': str(root.resolve()), 'report_sha256': s.sha(root / 'report.json')}


def preflight(args):
    output = Path(args.output)
    if output.exists():
        raise FileExistsError('Fresh V preflight required')
    if args.local and not output.resolve().is_relative_to(train.OUTPUT_ROOT.resolve()):
        raise ValueError('Local preflight must stay under V artifact root')
    protocol, pins = train.load_protocol(args.protocol)
    gate_args = argparse.Namespace(t_phase1_evaluation=args.t_phase1_evaluation, cpu_smoke=args.local)
    origin = train.origin_gate(train.origin_path(gate_args), protocol['origin']['t_phase1_evaluation_sha256'])
    actual = t.environment()
    if not args.local:
        t.require_formal_environment(actual)
    output.mkdir(parents=True, exist_ok=False)
    report = {'status': 'RUNNING', 'local': args.local, 'formal_clearance': False,
        'protocol_sha256': train.PROTOCOL_SHA, 'code_sha256': pins, 'origin': origin, 'environment': actual,
        'hostname': socket.gethostname(), 'remote_connection': 'NOT_RUN_OPERATOR_SIDE_ONLY',
        'gpu_probe': 'NOT_REQUIRED_BY_V_PROTOCOL_FIRST_FORMAL_RUN_HAS_REPRODUCTION_GATE'}
    try:
        report['data_identities'] = t.verify_data(args.data, local=args.local)
        frame = s.read_ledger(args.data)
        report['fold_sizes'] = s.fold_sizes(frame, s.read_json(s.DEFAULT_PROTOCOL))
        positions, _ = s.split_ledger(frame, 0, 'val')
        report['formal_fit_inner_counts'] = {k: len(positions[k]) for k in ('fit', 'inner')}
        manifest = train.implementation_manifest()
        s.write_json(output / 'implementation_manifest.json', manifest)
        report['implementation_manifest_sha256'] = s.sha(output / 'implementation_manifest.json')
        tests = output / 'tests'
        scope = tests / 'scope'
        scope.mkdir(parents=True, exist_ok=False)
        bootstrap = (
            'import sys; from pathlib import Path; '
            f'sys.path.insert(0, {str(ROOT / "scripts/research")!r}); '
            'import train_s_ablation, build_pure_m0_anchor; '
            f'train_s_ablation.SMOKE_ROOT = Path({str(scope)!r}); '
            f'build_pure_m0_anchor.OUTPUT_ROOT = Path({str(scope)!r}); '
            'import pytest; raise SystemExit(pytest.main(sys.argv[1:]))'
        )
        command = [sys.executable, '-I', '-B', '-c', bootstrap, '-q', '-p', 'no:cacheprovider',
            '--basetemp=' + str(scope / 'tmp'), '--junitxml=' + str(tests / 'junit.xml')]
        command += [str(ROOT / 'tests' / name) for name in
            ('test_mstca_ablation_factory.py', 'test_s_ablation.py', 'test_t_selection.py', 'test_v_confirm.py')]
        env = dict(os.environ, PYTEST_DISABLE_PLUGIN_AUTOLOAD='1', PYTHONDONTWRITEBYTECODE='1')
        if args.local:
            env['CUDA_VISIBLE_DEVICES'] = ''
        result = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True)
        with (tests / 'output.txt').open('x', encoding='utf-8') as stream:
            stream.write(result.stdout + '\n' + result.stderr)
        if result.returncode:
            raise RuntimeError('V/frozen regressions failed: STOP')
        cases = list(ET.parse(tests / 'junit.xml').getroot().iter('testcase'))
        skipped = [case for case in cases if case.find('skipped') is not None]
        if any(case.find('skipped').get('type') != 'pytest.xfail' for case in skipped):
            raise RuntimeError('V requires zero skipped tests')
        goldens = [case for case in cases if case.get('name', '').startswith(('test_full_state_digest_golden[', 'test_new_golden['))]
        if len(goldens) != 14 or any(case.find(tag) is not None for case in goldens for tag in ('skipped', 'failure', 'error')):
            raise RuntimeError('All unchanged 14 goldens must pass; never regenerate')
        report['pytest_summary'] = {'passed': len(cases) - len(skipped), 'xfailed': len(skipped), 'skipped': 0, 'failed': 0}
        report['golden_tests_passed'] = 14
        report['cpu_smoke'] = check_cpu_smoke(args.cpu_smoke_root)
        if args.local:
            report.update(status='LOCAL_GATES_PASS_FORMAL_NOT_AUTHORIZED', formal_gates_not_run=[
                'registered CUDA environment', 'three real val identity hashes', 'committed V STAND_IN smoke',
                'first formal seed reproduction against S0 val'])
        else:
            smoke = Path(args.cpu_smoke_root)
            r = s.read_json(smoke / 'report.json')
            names = ['report.json', 'selections.json', 'inner_logits.npz'] + [entry['file'] for entry in r['prediction_artifacts'].values()]
            files = [str((smoke / name).relative_to(ROOT)) for name in names]
            tracked = subprocess.run(['git', 'ls-files', '--error-unmatch', '--', *files], cwd=ROOT, capture_output=True)
            clean = subprocess.run(['git', 'diff', '--exit-code', 'HEAD', '--', *files], cwd=ROOT, capture_output=True)
            if tracked.returncode or clean.returncode:
                raise RuntimeError('Formal V requires unchanged committed smoke; this tool never commits')
            report.update(status='ALL_PREFLIGHT_GATES_PASS', formal_clearance=True)
        if train.load_protocol(args.protocol)[1] != pins or train.implementation_manifest() != manifest:
            raise ValueError('V implementation or inherited code drift during preflight')
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
    result.add_argument('--cpu-smoke-root', type=Path, required=True)
    result.add_argument('--t-phase1-evaluation', type=Path)
    result.add_argument('--local', action='store_true')
    return result


if __name__ == '__main__':
    print(preflight(parser().parse_args())['status'])
