"""Verify staged M0 preparation on matching server; no optimizer or real inference.

Run in the registered remote workspace, not on local data paths.
Does not install pytest or change source/protocol/environment.
"""
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import sys

BASE = Path('/root/autodl-tmp/causal_pure_mstca_m0_v1_20261102')
CODE = BASE / 'code'
sys.path.insert(0, str(CODE / 'packages/uav_fault'))
sys.path.insert(0, str(CODE / 'scripts/research'))
import train_causal_mstca_m0 as m0
import torch
from types import SimpleNamespace


def main():
    target = BASE / 'remote_preparation_verification.json'
    if target.exists():
        raise FileExistsError('Preserve prior preparation verification')
    protocol = CODE / 'configs/experiments/causal_pure_mstca_m0_v1.json'
    p = m0.load_protocol(protocol)
    report = json.loads((BASE / 'source_preflight/preflight.json').read_text())
    init = report['initialization']
    assert report['source_train_windows'] == 162069 and report['source_val_windows'] == 34710
    assert report['source_train_cases'] == 2570 and report['source_val_cases'] == 552
    assert report['formal_environment_matches'] and init['legacy_fresh_seed_reproduction_checked_in_matching_environment']
    assert init['common_encoder_and_head_exact_anchor_match'] and init['parameters'] == 105938
    assert init['common_state_tensors'] == 41
    assert report['train_val_case_and_simulation_pair_isolated'] and not report['test_or_Real_data_read']
    assert not (BASE / 'runs').exists() and not report['formal_training_started']
    for relative, expected in p['code_identities'].items():
        assert m0.source.sha(CODE / relative) == expected, relative
    assert m0.source.sha(CODE / p['baseline']['protocol_path']) == p['baseline']['protocol_sha256']
    anchor = Path('/root/autodl-tmp/causal_source_longbudget_v1_20261101/runs/uniform_replacement/initial.pt')
    # Prevent any feature extraction through the legacy class during this check.
    original_relation = m0.OnlineKGMSTCA._relation_features
    def forbidden(*args, **kwargs):
        raise AssertionError('M0 must not calculate legacy9 features')
    m0.OnlineKGMSTCA._relation_features = forbidden
    try:
        first, receipt = m0.build_model(p, 42, anchor)
        second, other_receipt = m0.build_model(p, 42, anchor)
        assert receipt == other_receipt
        assert all(torch.equal(value, second.state_dict()[key]) for key, value in first.state_dict().items())
        assert list(first._modules) == ['encoder', 'head']
        assert all(not key.startswith(('relation_', 'gate.')) for key in first.state_dict())
        first.eval()
        before = m0.state_digest(first.state_dict())
        with torch.inference_mode():
            output = first(torch.zeros(2, 26, 150))
        assert output.shape == (2, 2) and torch.isfinite(output).all()
        assert m0.state_digest(first.state_dict()) == before
    finally:
        m0.OnlineKGMSTCA._relation_features = original_relation
    # Exercise the preparation-only guard without creating a formal run.
    unused = BASE / 'unauthorized_run_must_not_exist'
    try:
        m0.run(SimpleNamespace(confirm_formal_run=False, cpu_check=False, output=unused), p)
    except RuntimeError as exc:
        assert 'M0_PREPARATION_ONLY' in str(exc)
    else:
        raise AssertionError('Preparation unexpectedly authorized formal training')
    assert not unused.exists()
    result = {'status': 'PASS_REMOTE_M0_PREPARATION_ONLY_AWAITING_TRAINING_AUTHORIZATION',
        'checked_at': datetime.now(timezone.utc).isoformat(), 'ssh_alias': 'autodl_UAV3',
        'remote_root': str(BASE), 'protocol_sha256': m0.source.sha(protocol),
        'environment': report['environment'], 'train_windows': 162069, 'val_windows': 34710,
        'parameters': 105938, 'common_state_tensors': 41, 'initialization': init,
        'case_and_pair_isolation_pass': True, 'all_frozen_code_pins_pass': True,
        'two_M0_arms_start_exactly_equal': True, 'synthetic_cpu_forward_shape': [2, 2],
        'synthetic_forward_state_unchanged': True, 'legacy_relation_feature_function_called': False,
        'preparation_guard_rejects_formal_run': True, 'pytest_installed': importlib.util.find_spec('pytest') is not None,
        'remote_pytest_result': 'NOT_RUN_MISSING_DEPENDENCY; earlier invocation reported no module named pytest',
        'remote_independent_assertions_pass': True, 'no_dependencies_installed': True,
        'optimizer_updates': 0, 'real_sample_model_inference': False, 'test_or_Real_read': False,
        'training_started': False, 'old_experiments_modified': False,
        'frozen_protocol_compute_note': 'Original autodl_UAV2/authentication status retained as preparation history; execution uses autodl_UAV3.',
        'raw_preflight_status_note': 'Pinned trainer uses PASS_LOCAL... status label on all hosts; this receipt independently identifies matching remote environment.'}
    with target.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
