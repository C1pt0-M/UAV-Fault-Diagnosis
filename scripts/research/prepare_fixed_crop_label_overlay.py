"""Prepare an audited label overlay, never a training-ready replacement cache.

Source CSV/X/y/norm and frozen protocols remain unchanged. No model is loaded.
Includes historical RflySim suffix and records unresolved input-quality issues.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from audit_source_event_boundaries import source_tasks, LINKS, MANIFEST
from audit_rflymad_raw_sources import digest

ROOT = Path(__file__).resolve().parents[2]
PROPOSAL = ROOT / 'artifacts/fixed_crop_label_policy_proposal_20261031'
QUALITY = ROOT / 'artifacts/source_event_boundaries_review_20261031'
BASELINE = ROOT / 'configs/experiments/normal_source_sampling_train_v1.json'
CACHE = Path('D:/UAV-Data/online_cache_v2')
KEYS = ['split', 'cache_index', 'case_uid', 'window_start_row', 'window_end_row']


def parse_bool(series):
    if series.isna().any():
        raise ValueError('Missing required boolean')
    if series.dtype == bool:
        return series.to_numpy()
    values = series.astype(str)
    if not values.isin(['True', 'False']).all():
        raise ValueError('Invalid boolean token')
    return values.eq('True').to_numpy()


def check_binary(values, name, allow_unknown=False):
    values = np.asarray(values)
    if values.ndim != 1 or not np.isfinite(values).all() or not np.equal(values, np.rint(values)).all():
        raise ValueError('Invalid integral labels: ' + name)
    if not np.isin(values, [-1, 0, 1] if allow_unknown else [0, 1]).all():
        raise ValueError('Invalid label space: ' + name)
    return values.astype(np.int64)


def prepare_frame(proposal, quality, links, cache_y, split, expected_extra):
    if proposal.duplicated(KEYS).any() or quality.duplicated(KEYS).any() or links.duplicated(KEYS).any():
        raise ValueError('Duplicate window identity')
    p = proposal.sort_values('cache_index').reset_index(drop=True)
    q = quality.sort_values('cache_index').reset_index(drop=True)
    link = links.sort_values('cache_index').reset_index(drop=True)
    if not p.split.eq(split).all() or not q.split.eq(split).all() or not link.split.eq(split).all():
        raise ValueError('Mixed partitions')
    for other in [q, link]:
        if len(p) != len(other) or not p[KEYS].equals(other[KEYS]):
            raise ValueError('Window identity/order mismatch')
    n = len(p)
    if not np.array_equal(p.cache_index, np.arange(n)):
        raise ValueError('Noncanonical source cache indices')
    y = check_binary(cache_y, 'historical cache')
    old = check_binary(p.existing_label, 'historical source')
    if not np.array_equal(old, y[:n]) or not np.array_equal(old, link.label):
        raise ValueError('Historical source labels differ from cache/links')
    if len(y) - n != expected_extra or (split != 'train' and expected_extra):
        raise ValueError('Unexpected appended data scope')
    if expected_extra and np.any(y[n:] != 0):
        raise ValueError('Non-normal historical RflySim suffix')
    labels = check_binary(p.proposed_label, 'proposed reference', True)
    gtd = check_binary(p.gtd_state_reference, 'GTD')
    command = check_binary(p.command_state_reference, 'command')
    agreed = gtd == command
    declared = parse_bool(p.source_references_agree)
    eligible = parse_bool(p.label_reference_eligible)
    if not np.array_equal(agreed, declared) or not np.array_equal(agreed, eligible):
        raise ValueError('Source agreement/mask mismatch')
    if not np.array_equal(labels, np.where(agreed, gtd, -1)):
        raise ValueError('Uncertain references assigned hard labels')
    if not np.array_equal(gtd, q.gtd_endpoint_reference) or not np.array_equal(command, q.command_endpoint_reference):
        raise ValueError('Label reference differs from quality audit')
    clock = parse_bool(p.input_clock_review_required)
    if not np.array_equal(clock, parse_bool(q.input_clock_requires_review)):
        raise ValueError('Clock review flag lost')
    family = link.case_level2_label.to_numpy(dtype=str)
    if not np.isin(family, ['normal', 'motor', 'sensor']).all():
        raise ValueError('Unknown source class')
    if ((family == 'normal') & (labels == 1)).any():
        raise ValueError('Unexpected activation in original-normal case')
    origins = np.where(labels == -1, 'uncertain_endpoint',
              np.where(labels == 1, 'fault_endpoint',
              np.where(family == 'normal', 'original_normal_case', 'normal_part_of_fault_case')))
    frame = p[KEYS + ['domain']].copy()
    frame['historical_label'] = old
    frame['reference_label'] = labels
    frame['source_family'] = family
    frame['flight_status'] = link.flight_status.to_numpy()
    frame['normal_origin_reference'] = origins
    frame['label_reference_defined'] = eligible
    frame['known_input_clock_review'] = clock
    frame['source_quality_scope'] = 'source_event_audit_not_all_channel_approval'
    frame['gap_over_0_1s_flag'] = q.sensor_gap_over_0_1s_intersects_window.to_numpy()
    frame['gap_over_0_1s_status_known'] = q.sensor_gap_over_0_1s_intersects_window.notna().to_numpy()
    # This is only a necessary exclusion condition; NOT training authorization.
    frame['pre_forward_candidate'] = eligible & ~clock
    if expected_extra:
        suffix = pd.DataFrame({'split': split, 'cache_index': np.arange(n, len(y)),
            'case_uid': pd.NA, 'window_start_row': pd.NA, 'window_end_row': pd.NA,
            'domain': 'RflySim', 'historical_label': 0, 'reference_label': 0,
            'source_family': 'normal', 'flight_status': pd.NA,
            'normal_origin_reference': 'rflysim_normal', 'label_reference_defined': True,
            'known_input_clock_review': False,
            'source_quality_scope': 'historical_suffix_unchanged_not_reaudited__false_is_no_new_flag',
            'gap_over_0_1s_flag': pd.NA, 'gap_over_0_1s_status_known': False,
            'pre_forward_candidate': True})
        frame = pd.concat([frame, suffix], ignore_index=True)
    if not np.array_equal(frame.cache_index, np.arange(len(y))):
        raise ValueError('Full-cache index integrity failed')
    return frame


def run(out):
    out = out.resolve()
    if out.exists():
        raise FileExistsError('Refuse to overwrite output: ' + str(out))
    for root in [Path('D:/RflyMAD').resolve(), Path('D:/UAV-Data').resolve()]:
        if out == root or root in out.parents:
            raise ValueError('Output must be outside dataset roots')
    tasks = source_tasks()  # verifies frozen source manifest and source links
    source_ids = {task[0]['case_uid'] for task in tasks}
    with MANIFEST.open(encoding='utf-8-sig') as stream:
        manifest = pd.read_csv(stream, dtype=str).fillna('')
    members = manifest[manifest.case_uid.isin(source_ids)]
    if len(members) != len(source_ids) or not members.split.isin(['train', 'val']).all():
        raise ValueError('Source membership mismatch')
    if members.groupby('simulation_pair_key').split.nunique().max() > 1:
        raise ValueError('Simulation-pair partition leakage')
    cfg = json.loads(BASELINE.read_text(encoding='utf-8'))['inputs']
    policy = json.loads((PROPOSAL / 'proposal.json').read_text(encoding='utf-8'))
    proposal_path = PROPOSAL / 'proposed_window_labels.csv'
    if digest(proposal_path) != policy['proposal_csv_sha256']:
        raise ValueError('Proposal identity mismatch')
    original_comparison = ROOT / 'artifacts/fixed_crop_label_validation_20261031/selected_window_comparison.csv'
    if digest(original_comparison) != policy['input_sha256']:
        raise ValueError('Diagnostic source identity mismatch')
    proposal = pd.read_csv(proposal_path)
    quality = pd.read_csv(QUALITY / 'all_selected_window_diagnostics.csv', low_memory=False)
    if len(proposal) != 200640 or len(quality) != 200640:
        raise ValueError('Unexpected source-window scope')
    identities = {str(proposal_path): digest(proposal_path), str(MANIFEST): digest(MANIFEST),
                  str(BASELINE): digest(BASELINE), str(QUALITY / 'all_selected_window_diagnostics.csv'): digest(QUALITY / 'all_selected_window_diagnostics.csv'),
                  str(original_comparison): digest(original_comparison)}
    norm = CACHE / 'norm_stats.json'
    if digest(norm) != cfg['norm_sha256']:
        raise ValueError('Frozen normalization changed')
    identities[str(norm)] = cfg['norm_sha256']
    prepared = {}
    for split in ['train', 'val']:
        cachepath = CACHE / (split + '.pt')
        if digest(cachepath) != cfg[split + '_cache_sha256']:
            raise ValueError('Frozen cache changed')
        identities[str(cachepath)] = cfg[split + '_cache_sha256']
        payload = torch.load(cachepath, map_location='cpu', weights_only=True, mmap=True)
        if tuple(payload['x'].shape) != (cfg[split + '_windows'], 26, 150) or payload['x'].dtype != torch.float32:
            raise ValueError('Unexpected input tensor shape/type')
        if payload['y'].dtype != torch.long or payload['y'].shape != (cfg[split + '_windows'],):
            raise ValueError('Unexpected target shape/type')
        linkpath = LINKS / (split + '_source_window_links.csv')
        if digest(linkpath) != cfg[split + '_links_sha256']:
            raise ValueError('Frozen source links changed')
        identities[str(linkpath)] = cfg[split + '_links_sha256']
        link = pd.read_csv(linkpath)
        if not link.groupby('case_uid').size().eq(64).all():
            raise ValueError('Source cap64 changed')
        prepared[split] = prepare_frame(proposal[proposal.split == split], quality[quality.split == split],
              link, payload['y'].numpy(), split, cfg['rflysim_normal_suffix_windows'] if split == 'train' else 0)
        del payload
    out.mkdir(parents=True, exist_ok=False)
    report = {'version': 'fixed_crop_label_overlay_review_v1',
              'created_at': datetime.now(timezone.utc).isoformat(), 'script_sha256': digest(__file__),
              'status': 'reviewed_overlay_prepared__training_blocked', 'training_ready': False,
              'scope': 'No source writes, new feature tensors, normalization fit, model load, training or final-test access. Official crop and all cache indices unchanged.',
              'required_before_training': [
                  'Approve separate handling of existing input gaps/all-channel validity; no arbitrary gap-based deletion.',
                  'Integrate an explicit pre-forward index view into a new versioned experiment, not the frozen original trainer.',
                  'Verify sampler, loss, validation, class weighting and early stopping all use eligible labels and identical old/new comparison coverage.',
                  'Retain 3680 historical RflySim suffix windows with original provenance; no fabricated per-case identities.',
                  'Retain continuous full-time replay for coverage and false alarms; this source-only cap64 overlay does not approve event/latency evaluation.'
              ],
              'critical_review_fixes': [
                  'Loss-only masking is insufficient for BatchNorm: uncertain and known clock-review rows must be removed BEFORE training forward if no influence is intended.',
                  'Current FocalLoss gathers targets directly and rejects -1. Current metrics/sampling also require binary labels; no direct import of proposal.',
                  'Regenerate label-based origin metadata, weights and sampling probabilities; obsolete phase/fault_fraction/onset fields deliberately omitted.',
                  'Pre-forward candidates are NOT an input-quality pass. Source clock anomalies are conservatively held out at window-index level without deleting cases.',
                  'Historical RflySim suffix remains in full-cache index space; no claimed fresh quality audit.',
                  'Any old/new training comparison must use the same pre-forward mask and frozen train-only normalization; no filling masked indices or claiming same supervised sample count.'
              ], 'input_identities': identities, 'splits': {}}
    for split, frame in prepared.items():
        frame.to_csv(out / (split + '_metadata.csv'), index=False)
        labels = frame.reference_label.to_numpy(dtype=np.int64)
        mask = frame.pre_forward_candidate.to_numpy(dtype=bool)
        np.savez_compressed(out / (split + '_overlay.npz'),
            cache_index=frame.cache_index.to_numpy(dtype=np.int64),
            historical_label=frame.historical_label.to_numpy(dtype=np.int64), reference_label=labels,
            label_reference_defined=frame.label_reference_defined.to_numpy(dtype=bool),
            known_input_clock_review=frame.known_input_clock_review.to_numpy(dtype=bool),
            pre_forward_candidate=mask)
        eligible = frame.loc[mask]
        if not set(eligible.reference_label.unique()).issubset({0, 1}):
            raise ValueError('Unknown label passed pre-forward necessary conditions')
        report['splits'][split] = {
            'full_cache_windows': len(frame), 'source_windows': int(frame.domain.isin(['SIL', 'HIL']).sum()),
            'historical_rflysim_suffix_windows': int(frame.domain.eq('RflySim').sum()),
            'unknown_label_windows': int((labels == -1).sum()),
            'clock_review_windows': int(frame.known_input_clock_review.sum()),
            'pre_forward_candidate_windows': int(mask.sum()), 'pre_forward_held_out_windows': int((~mask).sum()),
            'candidate_binary_relabels': int((eligible.historical_label != eligible.reference_label).sum()),
            'candidate_label_counts': {str(k): int(v) for k, v in eligible.reference_label.value_counts().items()},
            'candidate_origin_counts': {str(k): int(v) for k, v in eligible.normal_origin_reference.value_counts().items()},
            'source_gap_flag_windows_preserved': int(frame.gap_over_0_1s_flag.eq(True).sum()),
            'quality_gate_status': 'pending__pre_forward_candidates_not_training_approval',
            'metadata_sha256': digest(out / (split + '_metadata.csv')),
            'overlay_sha256': digest(out / (split + '_overlay.npz'))}
    for path, expected in identities.items():
        if digest(Path(path)) != expected:
            raise ValueError('Input modified during preparation: ' + path)
    (out / 'review.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'status': report['status'], 'training_ready': False, 'splits': report['splits']}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    run(args.output)
