"""V regressions: synthetic CPU only, no real val, remote or GPU execution."""
from __future__ import annotations

from itertools import product
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts/research'))
import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn
from torch.utils.data import TensorDataset
import train_v_confirm as train
import evaluate_v_confirm as evaluate
import preflight_v_confirm as preflight
from train_v_confirm import s, t, m


@pytest.fixture(scope='module', autouse=True)
def cpu_settings():
    old = torch.get_num_threads()
    deterministic = torch.are_deterministic_algorithms_enabled()
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    yield
    torch.set_num_threads(old)
    torch.use_deterministic_algorithms(deterministic)


def decide(ratios=None, ci=None, delta=None, drops=None, ap=None):
    return evaluate.criteria([.9] * 5 if ratios is None else ratios, [.8, .95] if ci is None else ci,
        {'GPS': 0.} if delta is None else delta, {'GPS': 0.} if drops is None else drops,
        [-.01, .01] if ap is None else ap)


@pytest.mark.parametrize('ratios,ci,expected', [
    ([.9] * 5, [.8, .99], True), ([.9, .9, .9, .9, 1.1], [.8, .99], True),
    ([.9, .9, .9, 1., 1.], [.8, .99], False), ([.9] * 5, [.8, 1.], False),
    ([.9] * 5, [.8, 1.01], False), ([1.] * 5, [.8, .99], False),
    ([.9, .9, .9, .9, 10.], [.8, .99], False), ([None, .9, .9, .9, .9], [.8, .99], False),
])
def test_c1_all_branches_strict_bounds_and_four_of_five(ratios, ci, expected):
    assert decide(ratios=ratios, ci=ci)['criteria']['C1_false_alarms'] is expected


@pytest.mark.parametrize('interval', [None, [float('nan'), .9]])
def test_c1_undefined_interval_fails(interval):
    result = evaluate.criteria([.9] * 5, interval, {'GPS': 0}, {'GPS': 0}, [0, 1])
    assert not result['criteria']['C1_false_alarms']


def test_c2_tied_kth_normal_scores_report_actual_fp():
    y = np.array([0, 0, 0, 0, 1, 1])
    score = np.array([.9, .8, .8, .2, .95, .3])
    value = evaluate.matched_fp_threshold(y, score, 2)
    assert value == {'threshold': .8, 'target_fp': 2, 'realised_fp': 3}


def test_c2_weighted_kth_equivalent_to_repeated_windows():
    y = np.array([0, 0, 1, 1])
    p = np.array([.9, .8, .95, .3])
    w = np.array([2, 3, 1, 1])
    a = evaluate.matched_fp_threshold(y, p, 3, w)
    b = evaluate.matched_fp_threshold(np.repeat(y, w), np.repeat(p, w), 3)
    assert a == b == {'threshold': .8, 'target_fp': 3, 'realised_fp': 5}


@pytest.mark.parametrize('k', [0, None, -1])
def test_c2_zero_or_undefined_k_is_not_satisfied(k):
    value = evaluate.matched_fp_threshold([0, 1], [.5, .7], k)
    assert value['threshold'] is None and value['realised_fp'] is None
    assert not decide(delta={'GPS': None})['criteria']['C2_matched_fp_class_recall']


@pytest.mark.parametrize('delta,expected', [(-1., True), (-1.000001, False), (None, False)])
def test_c2_point_estimate_one_pp_boundary(delta, expected):
    assert decide(delta={'GPS': delta})['criteria']['C2_matched_fp_class_recall'] is expected


@pytest.mark.parametrize('cls,drop,expected', [('GPS', 2., True), ('GPS', 2.000001, False),
    ('accelerometer', 2.000001, False), ('barometer', 2., True), ('motor', None, False)])
def test_c3_cap_applies_to_every_class_exact_boundary(cls, drop, expected):
    assert decide(drops={cls: drop})['criteria']['C3_r97_class_cap'] is expected


@pytest.mark.parametrize('interval,expected', [([-.1, 0], True), ([-.1, .01], True),
    ([-.1, -.000001], False), (None, False), ([0, float('inf')], False)])
def test_c4_upper_bound_zero_is_allowed(interval, expected):
    value = evaluate.criteria([.9] * 5, [.8, .99], {'GPS': 0}, {'GPS': 0}, interval)
    assert value['criteria']['C4_ap'] is expected


@pytest.mark.parametrize('flags', list(product((False, True), repeat=4)))
def test_verdict_every_failure_combination(flags):
    c1, c2, c3, c4 = flags
    value = decide(ratios=[.9] * 5 if c1 else [1.] * 5,
        delta={'GPS': 0 if c2 else -2}, drops={'GPS': 0 if c3 else 3}, ap=[-.1, 0 if c4 else -.01])
    assert value['criteria'] == dict(zip(evaluate.CRITERIA, flags))
    assert value['failed_criteria'] == [name for name, good in zip(evaluate.CRITERIA, flags) if not good]
    assert value['verdict'] == ('CONFIRMED_ADOPT_FIXED_100' if all(flags) else 'NOT_CONFIRMED_KEEP_T0')


def test_float64_softmax_path_bitwise_matches_s_and_not_float32_softmax():
    model = nn.Linear(3, 2)
    x = torch.randn(1031, 3, generator=torch.Generator().manual_seed(8))
    model.eval()
    expected, raw = [], []
    with torch.inference_mode():
        for start in range(0, len(x), 512):
            logits = model(x[start:start + 512])
            expected.append(logits.double().softmax(1)[:, 1].numpy())
            raw.append(logits.numpy())
    logits = np.concatenate(raw)
    actual = train.logits_to_probability(logits)
    assert actual.dtype == np.float64 and np.array_equal(actual, np.concatenate(expected))
    assert not np.array_equal(actual, torch.from_numpy(logits).softmax(1)[:, 1].numpy())


@pytest.mark.parametrize('bad', [np.zeros((2, 2), dtype=np.float64), np.zeros((2, 3), dtype=np.float32),
    np.array([[np.inf, 0]], dtype=np.float32)])
def test_logits_boundary_validation(bad):
    with pytest.raises(ValueError, match='logits'):
        train.logits_to_probability(bad)


def frame_for_scores(labels, fault_classes=None):
    labels = np.asarray(labels, dtype=np.int64)
    fault_classes = ['GPS' if y else 'normal' for y in labels] if fault_classes is None else fault_classes
    return pd.DataFrame({'case_uid': [f'SIL/cruise/{c}/{i}' for i, c in enumerate(fault_classes)],
        'qualified_y': labels, 'normal_origin': np.where(labels == 1, 'fault_or_unknown', 'normal_flight'),
        'domain': 'SIL', 'window_end_row': 1000, 'original_cache_index': np.arange(len(labels), dtype=np.int64),
        'association_group': [f'{i // 2:05}' for i in range(len(labels))]})


def test_descriptive_ratios_include_r95_without_pseudocount():
    metrics = {a: {str(seed): {'primary': {'fp': 20 if a == 'V0' else 10},
        'fp_at_r95': {'fp': 8 if a == 'V0' else 2}} for seed in t.SEEDS} for a in evaluate.ARMS}
    assert all(value == {'FP_R97_V1_over_V0': .5, 'FP_R95_V1_over_V0': .25}
               for value in evaluate.per_seed_fp_ratios(metrics).values())
    metrics['V0']['42']['fp_at_r95']['fp'] = 0
    assert evaluate.per_seed_fp_ratios(metrics)['42']['FP_R95_V1_over_V0'] is None


def test_inner_logits_persist_before_next_epoch_even_if_that_epoch_fails(tmp_path, monkeypatch):
    model = nn.Linear(2, 2)
    inner = TensorDataset(torch.zeros(3, 2), torch.tensor([0, 1, 0]))
    logits = np.arange(6, dtype=np.float32).reshape(3, 2)
    def epoch(*args):
        if args[-1] == 2:
            saved = np.load(tmp_path / 'inner_logits_progress.npy')
            assert np.array_equal(saved[0], logits)
            assert s.read_json(tmp_path / 'report.json')['epochs'] == 1
            raise RuntimeError('simulated epoch-2 infrastructure failure')
        return 1e-4, .2, 1
    monkeypatch.setattr(t, 'train_epoch', epoch)
    monkeypatch.setattr(s, 'inner_ce', lambda *a: .3)
    monkeypatch.setattr(train, 'infer_logits', lambda *a: logits)
    report = {'history': [], 'io_events': []}
    with pytest.raises(RuntimeError, match='simulated epoch-2'):
        train.train_trajectory(model, None, inner, tmp_path, 2, 'cpu', 42, report)
    assert np.array_equal(np.load(tmp_path / 'inner_logits_progress.npy')[0], logits)


def test_ce_from_logits_finite_with_saturated_probability_endpoints():
    frame = frame_for_scores([0, 1])
    logits = np.array([[-1000, 1000], [1000, -1000]], dtype=np.float32)
    assert np.array_equal(train.logits_to_probability(logits), [1., 0.])
    metrics = evaluate.metrics_from_logits(frame, logits)
    assert metrics['class_balanced_ce'] == 2000.
    assert metrics['brier_binary'] == 1.


@pytest.mark.parametrize('gps_tp,expected,drop', [(490, True, 2.), (489, False, 2.2)])
def test_c3_integer_count_aggregation_does_not_lose_exact_two_pp(gps_tp, expected, drop):
    frame = frame_for_scores([0] * 20 + [1] * 1000, ['normal'] * 20 + ['GPS'] * 500 + ['accelerometer'] * 500)
    p0 = np.r_[np.full(10, .99), np.full(10, .2), np.full(500, .9), np.full(470, .9), np.full(30, .1)]
    accel_tp = 970 - gps_tp
    p1 = np.r_[p0[:20], np.full(gps_tp, .9), np.full(500 - gps_tp, .1),
               np.full(accel_tp, .9), np.full(500 - accel_tp, .1)]
    scores = {'V0': {seed: p0 for seed in t.SEEDS}, 'V1': {seed: p1 for seed in t.SEEDS}}
    point = evaluate.measure(evaluate.prepare(frame, scores))
    assert point['C3_seed_mean_recall_drop_pp']['GPS'] == drop
    assert decide(drops=point['C3_seed_mean_recall_drop_pp'])['criteria']['C3_r97_class_cap'] is expected


def test_bootstrap_one_shared_plan_all_eleven_vectors_and_no_c2_ci_veto(monkeypatch):
    frame = frame_for_scores([0, 1] * 3)
    p = np.array([.75, .5, .875, .625, 1., .75])
    scores = {a: {seed: p.copy() for seed in t.SEEDS} for a in evaluate.ARMS}
    prepared = evaluate.prepare(frame, scores)
    calls = []
    original = m.MetricCache.basic
    def record(self, weights=None, recall=.97):
        calls.append((id(weights), weights.copy()))
        return original(self, weights, recall)
    monkeypatch.setattr(m.MetricCache, 'basic', record)
    result = evaluate.paired_bootstrap(prepared, frame.association_group.to_numpy(), replicates=8)
    assert len(calls) == 8 * 11
    for i in range(0, len(calls), 11):
        assert len({call[0] for call in calls[i:i + 11]}) == 1
        assert all(np.array_equal(calls[i][1], call[1]) for call in calls[i:i + 11])
    assert result['aggregate_ratio_ci'] == [1., 1.] and result['aggregate_delta_ap_ci'] == [0., 0.]
    assert 'C2_class_delta_ci_pp' in result
    assert decide()['verdict'] == 'CONFIRMED_ADOPT_FIXED_100'  # C2 uses its point, not a CI veto.


def test_undefined_bootstrap_draws_not_dropped_redrawn_or_pseudocounted():
    frame = frame_for_scores([0, 1])
    frame['association_group'] = ['normal', 'fault']
    scores = {a: {seed: np.array([.9, .5]) for seed in t.SEEDS} for a in evaluate.ARMS}
    result = evaluate.paired_bootstrap(evaluate.prepare(frame, scores), frame.association_group.to_numpy(), replicates=40)
    assert result['replicates'] == 40 and result['undefined_ratio_replicates'] > 0 and result['undefined_ap_replicates'] > 0
    assert result['aggregate_ratio_ci'] is None and result['aggregate_delta_ap_ci'] is None


@pytest.mark.parametrize('change', ['digest', 'epoch', 'fp'])
def test_reproduction_three_checks_fail_independently(tmp_path, change):
    s.write_json(tmp_path / 'S0_seed42_fold0/report.json', {'status': 'COMPLETE', 'phase': 'val',
        'arm': 'S0', 'seed': 42, 'fold': 0, 'protocol_sha256': s.PROTOCOL_SHA,
        'best_epoch': 2, 'checkpoint_state_digests': {'best': 'd2'}})
    history = [{'state_digest': 'd1'}, {'state_digest': 'd2'}]
    selected = {'V0': {'epoch': 2}}
    assert train.reproduction_gate(42, history, selected, 785, tmp_path)['status'] == 'PASS'
    if change == 'digest':
        history[1]['state_digest'] = 'wrong'
    elif change == 'epoch':
        selected['V0']['epoch'] = 1
    with pytest.raises(RuntimeError, match={'digest': 'digest', 'epoch': 'selected epoch', 'fp': 'FP@R97'}[change]):
        train.reproduction_gate(42, history, selected, 786 if change == 'fp' else 785, tmp_path)


@pytest.mark.parametrize('v0,available,expected', [(7, True, 2), (100, True, 1), (2, False, 1)])
def test_selected_checkpoint_dedup_no_proxy_for_epoch100(tmp_path, monkeypatch, v0, available, expected):
    model = nn.Linear(2, 2)
    data = TensorDataset(torch.ones(4, 2), torch.tensor([0, 1, 0, 1]))
    data.ids, data.y = torch.arange(4), data.tensors[1]
    (tmp_path / 'checkpoints').mkdir()
    for epoch in {v0, 100} if available else {v0}:
        s.save_state(tmp_path / f'checkpoints/epoch_{epoch}.pt', model.state_dict())
    selected = {'V0': {'epoch': v0}, 'V1': {'epoch': 100, 'available': available}}
    calls = []
    actual = train.infer_logits
    def count(*args):
        calls.append(1)
        return actual(*args)
    monkeypatch.setattr(train, 'infer_logits', count)
    files = train.predict_selected(model, t.inference_loader(data, 42), data, selected, tmp_path, 'cpu', stand_in=not available)
    assert len(calls) == len(files) == expected
    if not available:
        assert selected['V1'] == {'epoch': 100, 'available': False} and '100' not in files
        assert list(files.values())[0]['file'].startswith('stand_in_')


def test_t_origin_hash_mismatch_refuses_before_any_data(tmp_path, monkeypatch):
    path = tmp_path / 'not_T.json'
    path.write_text('{}', encoding='utf-8')
    monkeypatch.setattr(t, 'verify_data', lambda *a, **kw: pytest.fail('No data before T origin SHA gate'))
    args = train.parser().parse_args(['--seed', '42', '--output', str(tmp_path / 'run'),
                                     '--t-phase1-evaluation', str(path), '--cpu-smoke'])
    with pytest.raises(ValueError, match='SHA mismatch'):
        train.run(args)
    assert not args.output.exists()


def test_protocol_sixteen_pins_and_unchanged_fourteen_goldens():
    p, pins = train.load_protocol()
    assert len(pins) == 16 and pins == p['code_identities']['files']
    assert len(t.golden_digests()) == 14
    assert train.origin_gate(train.LOCAL_ORIGIN, p['origin']['t_phase1_evaluation_sha256'])['t_adoption_unchanged'] == 'T0'


def test_synthetic_training_run_reads_only_train_uses_disjoint_stand_in(tmp_path, monkeypatch):
    n = 1800
    labels = np.tile([0, 0, 1], n // 3)
    frame = pd.DataFrame({'association_group': [f'{i // 30:05}' for i in range(n)],
        'qualified_y': labels, 'original_cache_index': np.arange(n, dtype=np.int64),
        'normal_origin': np.tile(['normal_flight', 'normal_part_of_fault_case', 'fault_or_unknown'], n // 3)})
    calls = []
    def ledger(root, split='train'):
        assert split == 'train', 'Real val must never be opened by local mode'
        calls.append(split)
        return frame
    class FakeCache(TensorDataset):
        def __init__(self, root, fr, indices, split='train'):
            assert split == 'train'
            calls.append('cache_train')
            self.x = torch.zeros(len(indices), 26, 150)
            self.y = torch.tensor(fr.iloc[indices].qualified_y.to_numpy(), dtype=torch.long)
            self.ids = torch.tensor(fr.iloc[indices].original_cache_index.to_numpy(), dtype=torch.long)
            super().__init__(self.x, self.y)
    monkeypatch.setattr(s, 'read_ledger', ledger)
    monkeypatch.setattr(s, 'CacheWindows', FakeCache)
    monkeypatch.setattr(s, 'fold_sizes', lambda *a: {})
    monkeypatch.setattr(t, 'verify_data', lambda *a, **kw: {'matched_sha256': {}, 'skipped_sha256': sorted(t.VAL_FILES)})
    monkeypatch.setattr(train, 'SMOKE_ROOT', tmp_path)
    args = train.parser().parse_args(['--seed', '42', '--output', str(tmp_path / 'run'), '--cpu-smoke'])
    r = train.run(args)
    assert r['status'] == 'COMPLETE_CPU_SMOKE_STAND_IN_NOT_FORMAL'
    assert r['epochs'] == 2 and r['optimizer_steps'] == 2 and r['val_prediction_passes'] == 0
    assert calls == ['train', 'cache_train', 'cache_train', 'cache_train']
    pop = s.read_json(args.output / 'population.json')
    assert not set(pop['STAND_IN']) & (set(pop['fit']) | set(pop['inner']))
    assert not set(pop['fit']) & set(pop['inner'])
    with np.load(args.output / 'inner_logits.npz') as saved:
        assert saved['logits'].dtype == np.float32 and saved['logits'].shape[0] == 2 and saved['logits'].shape[2] == 2
        assert np.array_equal(saved['logits'], np.load(args.output / r['inner_logits_progress_file']))
    for item in r['prediction_artifacts'].values():
        with np.load(args.output / item['file']) as saved:
            assert str(saved['population_scope']) == 'STAND_IN' and saved['logits'].shape == (256, 2)


def test_atomic_reports_retain_io_retries_and_do_not_change_training_rng(tmp_path, monkeypatch):
    replace = Path.replace
    attempts = []
    def locked(self, other):
        attempts.append(1)
        if len(attempts) <= 2:
            raise PermissionError('simulated Windows file lock')
        return replace(self, other)
    monkeypatch.setattr(Path, 'replace', locked)
    monkeypatch.setattr(train.time, 'sleep', lambda *a: None)
    torch_state = torch.random.get_rng_state().clone()
    np_state = np.random.get_state()
    value = {'status': 'RUNNING', 'io_events': []}
    train.write_report(tmp_path / 'report.json', value)
    assert len(s.read_json(tmp_path / 'report.json')['io_events']) == 2
    assert torch.equal(torch_state, torch.random.get_rng_state())
    assert np.array_equal(np_state[1], np.random.get_state()[1])


def test_formal_v_slot_is_five_ordered_seeds(tmp_path, monkeypatch):
    monkeypatch.setattr(train, 'OUTPUT_ROOT', tmp_path)
    args = SimpleNamespace(seed=43, output=tmp_path / 'val/seed43')
    with pytest.raises(RuntimeError, match='Preceding'):
        train.require_formal_slot(args, 'manifest')
    s.write_json(tmp_path / 'val/seed42/report.json', {'status': 'COMPLETE', 'protocol_sha256': train.PROTOCOL_SHA,
        'seed': 42, 'epochs': 100, 'implementation_manifest_sha256': 'manifest'})
    train.require_formal_slot(args, 'manifest')


def test_local_preflight_cannot_touch_gpu_before_environment_refusal(tmp_path, monkeypatch):
    monkeypatch.setattr(t, 'environment', lambda: {'torch': '2.8.0+cpu'})
    monkeypatch.setattr(preflight.subprocess, 'run', lambda *a, **kw: pytest.fail('No GPU or remote process allowed'))
    args = preflight.parser().parse_args(['--output', str(tmp_path / 'out'), '--cpu-smoke-root', str(tmp_path / 'smoke'),
        '--t-phase1-evaluation', str(train.LOCAL_ORIGIN)])
    with pytest.raises(RuntimeError, match='Formal environment'):
        preflight.preflight(args)
    assert not args.output.exists()


def test_preflight_smoke_requires_partial_selection_and_stand_in_scope(tmp_path, monkeypatch):
    monkeypatch.setattr(train, 'SMOKE_ROOT', tmp_path / 'smoke')
    root = train.SMOKE_ROOT / 'fresh'
    root.mkdir(parents=True)
    history = [{'inner_class_balanced_ce': 1.}, {'inner_class_balanced_ce': .9}]
    selected = train.selections([1., .9])
    s.write_json(root / 'selections.json', selected)
    with (root / 'inner_logits.npz').open('xb') as stream:
        np.savez_compressed(stream, logits=np.zeros((2, 256, 2), dtype=np.float32))
    path = root / 'stand_in_logits_epoch_2.npz'
    with path.open('xb') as stream:
        np.savez_compressed(stream, logits=np.zeros((256, 2), dtype=np.float32), population_scope=np.asarray('STAND_IN'))
    report = {'status': 'COMPLETE_CPU_SMOKE_STAND_IN_NOT_FORMAL', 'prediction_scope': 'STAND_IN',
        'protocol_sha256': train.PROTOCOL_SHA, 'seed': 42, 'epochs': 2, 'val_prediction_passes': 0,
        'stand_in_prediction_passes': 1, 'confirmation_window_count': 256,
        'used_window_counts': {'fit': 512, 'inner': 256}, 'history': history,
        'implementation_files': train.implementation_manifest()['files'],
        'prediction_artifacts': {'2': {'file': path.name, 'sha256': s.sha(path)}}}
    s.write_json(root / 'report.json', report)
    assert preflight.check_cpu_smoke(root)['status'] == 'PASS'
    report['prediction_scope'] = 'SOURCE_VAL_PREVIOUSLY_USED_VALIDATION'
    s.write_json(root / 'report.json', report, update=True)
    with pytest.raises(ValueError, match='STAND_IN smoke'):
        preflight.check_cpu_smoke(root)


def test_evaluator_cpu_refusal_before_any_val_ledger(tmp_path, monkeypatch):
    monkeypatch.setattr(s, 'read_ledger', lambda *a, **kw: pytest.fail('Local formal evaluator must refuse before val'))
    args = evaluate.parser().parse_args(['--runs', str(tmp_path / 'unused'), '--output', str(tmp_path / 'new'),
        '--t-phase1-evaluation', str(train.LOCAL_ORIGIN)])
    with pytest.raises(RuntimeError, match='Formal environment'):
        evaluate.evaluate(args)
    assert not args.output.exists()


def test_five_formal_shaped_synthetic_runs_validate_logits_and_reject_stand_in(tmp_path, monkeypatch):
    frame = frame_for_scores([0, 1, 0, 1])
    manifest = train.implementation_manifest()
    origin_sha = '7250c38832602a2b61dbd90488ba6890718064626da58040dfe6e9ccf968dfb7'
    expected_data = s.read_json(s.BASELINE)['data_identities']
    gates = []
    monkeypatch.setattr(train, 'reproduction_gate', lambda seed, *args: gates.append(seed))
    for seed in t.SEEDS:
        folder = tmp_path / f'seed{seed}'
        (folder / 'checkpoints').mkdir(parents=True)
        history = [{'epoch': epoch, 'inner_class_balanced_ce': 1., 'state_digest': f'd{epoch}'} for epoch in range(1, 101)]
        for epoch in range(1, 101):
            (folder / f'checkpoints/epoch_{epoch}.pt').write_bytes(b'synthetic fixture, not actual model weights')
        s.write_json(folder / 'population.json', {'inner': [100, 101]})
        with (folder / 'inner_logits.npz').open('xb') as stream:
            np.savez_compressed(stream, logits=np.zeros((100, 2, 2), dtype=np.float32),
                labels=np.array([0, 1], dtype=np.int64), original_cache_index=np.array([100, 101], dtype=np.int64),
                epochs=np.arange(1, 101, dtype=np.int64), population_scope=np.asarray('SOURCE_TRAIN_INNER'))
        choice = train.selections([1.] * 100)
        s.write_json(folder / 'selections.json', choice)
        artifacts = {}
        for epoch in (1, 100):
            path = folder / f'val_logits_epoch_{epoch}.npz'
            with path.open('xb') as stream:
                np.savez_compressed(stream, logits=np.tile(np.array([[0., 1.]], dtype=np.float32), (4, 1)),
                    labels=frame.qualified_y.to_numpy(dtype=np.int64), original_cache_index=frame.original_cache_index.to_numpy(dtype=np.int64),
                    epochs=np.array([epoch], dtype=np.int64), population_scope=np.asarray('SOURCE_VAL_PREVIOUSLY_USED_VALIDATION'))
            artifacts[str(epoch)] = {'file': path.name, 'sha256': s.sha(path)}
        s.write_json(folder / 'report.json', {'status': 'COMPLETE', 'phase': 'val', 'seed': seed, 'epochs': 100,
            'cpu_smoke': False, 'prediction_scope': 'SOURCE_VAL_PREVIOUSLY_USED_VALIDATION',
            'protocol_sha256': train.PROTOCOL_SHA, 'history': history,
            'environment': {'torch': '2.5.1+cu124', 'numpy': '2.1.3', 'torch_cuda': '12.4', 'cuda_available': True,
                'cublas_workspace_config': ':4096:8', 'aten_cpu_capability': 'default'},
            'initial_digest': t.golden_digests()[('mstca', seed)], 'implementation_files': manifest['files'],
            'implementation_manifest_sha256': 'same synthetic manifest', 'origin': {'sha256': origin_sha},
            'data_identities': {'matched_sha256': expected_data, 'skipped_sha256': []},
            'used_window_counts': {'inner': 2}, 'inner_logits_sha256': s.sha(folder / 'inner_logits.npz'),
            'prediction_artifacts': artifacts, 'val_prediction_passes': 2,
            'checkpoint_state_digests': {str(epoch): f'd{epoch}' for epoch in range(1, 101)}})
    scores, logits, reports, selected = evaluate.load_runs(tmp_path, frame, origin_sha, device='cpu')
    assert len(reports) == len(selected) == 5 and gates == [42, 43, 44]
    assert all(logits[a][seed].shape == (4, 2) and scores[a][seed].shape == (4,) for a in evaluate.ARMS for seed in t.SEEDS)
    path = tmp_path / 'seed42/val_logits_epoch_100.npz'
    with np.load(path) as saved:
        fields = {key: saved[key] for key in saved.files}
    fields['population_scope'] = np.asarray('STAND_IN')
    with path.open('wb') as stream:
        np.savez_compressed(stream, **fields)
    report_path = path.parent / 'report.json'
    report = s.read_json(report_path)
    report['prediction_artifacts']['100']['sha256'] = s.sha(path)
    s.write_json(report_path, report, update=True)
    with pytest.raises(ValueError, match='STAND_IN'):
        evaluate.load_runs(tmp_path, frame, origin_sha, device='cpu')
