"""Materialize label-time repair v1 without overwriting old data or training.

Official crop/continuous values are immutable. Copies source CSVs with GTD state
references and explicit supervision flags; writes compact binary train/val caches
with excluded-index ledgers. Uses validated raw-event sidecars and verifies their
original identities. Original converter and frozen training protocols are untouched.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
import traceback

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/research'))
from audit_source_event_boundaries import source_tasks, MANIFEST
from audit_rflymad_raw_sources import digest

OLD_CACHE = Path('D:/UAV-Data/online_cache_v2')
LABELS = ROOT / 'artifacts/fixed_crop_label_validation_20261031'
OVERLAY = ROOT / 'artifacts/fixed_crop_label_review_v1_20261031'
VERSION = 'rflymad_online_binary_v2_labeltime_v1'
FLAGS = ['labeltime_supervision_valid', 'labeltime_input_clock_review']
STATE = 'UAVState_data_fault_state'


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, data):
    with path.open('x', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2, allow_nan=False)


def make_split_view(x, historical_y, overlay, block=512):
    """Copy only eligible rows; never modifies x/y or passes unknown labels."""
    indices = np.asarray(overlay['cache_index'])
    previous = np.asarray(overlay['historical_label'])
    labels = np.asarray(overlay['reference_label'])
    defined = np.asarray(overlay['label_reference_defined'])
    clock = np.asarray(overlay['known_input_clock_review'])
    mask = np.asarray(overlay['pre_forward_candidate'])
    n = len(historical_y)
    if x.dtype != torch.float32 or x.shape[0] != n or historical_y.dtype != torch.long:
        raise ValueError('Unexpected historical cache type/shape')
    if not np.array_equal(indices, np.arange(n)) or any(v.shape != (n,) for v in [previous, labels, defined, clock, mask]):
        raise ValueError('Full-cache index/overlay shape mismatch')
    if any(v.dtype != np.bool_ for v in [defined, clock, mask]):
        raise ValueError('Masks must be boolean')
    if not np.isin(previous, [0, 1]).all() or not np.array_equal(previous, historical_y.numpy()):
        raise ValueError('Historical labels differ from overlay')
    if not np.issubdtype(labels.dtype, np.integer) or not np.isin(labels, [-1, 0, 1]).all():
        raise ValueError('Invalid reference labels')
    if not np.array_equal(defined, labels >= 0) or not np.array_equal(mask, defined & ~clock):
        raise ValueError('Overlay mask/label contract mismatch')
    chosen = np.flatnonzero(mask).astype(np.int64)
    if not len(chosen) or not np.isin(labels[chosen], [0, 1]).all():
        raise ValueError('No valid binary targets')
    target = torch.empty((len(chosen), *x.shape[1:]), dtype=x.dtype)
    for start in range(0, len(chosen), block):
        raw = x[torch.from_numpy(chosen[start:start + block])].clone()
        if not torch.isfinite(raw).all():
            raise ValueError('Nonfinite input reached compact cache; no zero filling')
        target[start:start + len(raw)].copy_(raw)
    return {'x': target, 'y': torch.from_numpy(labels[chosen].copy()).long()}, chosen


def repair_csv(original, destination, label_file, record, input_clock_review):
    if destination.exists():
        raise FileExistsError('Refuse repaired CSV overwrite')
    if digest(original) != record['processed_csv_sha256'] or digest(label_file) != record['sidecar_sha256']:
        raise ValueError('Input CSV/reference identity mismatch')
    with np.load(label_file, allow_pickle=False) as d:
        t = d['timestamp_us']; old = d['existing_state']; ref = d['gtd_state_reference']
        command = d['command_state_reference']; cover = d['gtd_covered'] & d['command_covered']
        valid = d['source_references_agree']
        if len(t) != record['source_rows'] or not cover.all() or not np.isin(ref, [0, 1]).all():
            raise ValueError('Incomplete reference in unchanged crop')
        if not np.array_equal(valid, ref == command):
            raise ValueError('Reference validity disagreement')
        destination.parent.mkdir(parents=True, exist_ok=True)
        unchanged_hash = hashlib.sha256()
        count = changed = 0
        with original.open(encoding='utf-8-sig', newline='') as src, destination.open('x', encoding='utf-8', newline='') as dst:
            reader = csv.reader(src); writer = csv.writer(dst)
            header = next(reader)
            if len(set(header)) != len(header) or any(name in header for name in FLAGS):
                raise ValueError('Unexpected or already-repaired CSV header')
            state_idx, time_idx = header.index(STATE), header.index('Timestamp')
            writer.writerow(header + FLAGS)
            for i, row in enumerate(reader):
                if i >= len(t) or len(row) != len(header):
                    raise ValueError('Unexpected processed row shape/count')
                if float(row[time_idx]) != int(t[i]) or float(row[state_idx]) != int(old[i]):
                    raise ValueError('Original row time/state differs from reference')
                # Exact strings of ALL original fields except the targeted state.
                unchanged = row[:state_idx] + row[state_idx+1:]
                unchanged_hash.update(json.dumps(unchanged, ensure_ascii=False, separators=(',', ':')).encode('utf-8'))
                changed += int(int(old[i]) != int(ref[i]))
                row[state_idx] = str(int(ref[i]))
                writer.writerow(row + [str(int(valid[i])), str(int(input_clock_review))])
                count += 1
        if count != len(t):
            raise ValueError('Truncated CSV')
        after = hashlib.sha256()
        with destination.open(encoding='utf-8', newline='') as f:
            reader = csv.reader(f); new_header = next(reader)
            if new_header != header + FLAGS:
                raise ValueError('CSV schema changed beyond declared patch')
            checked = 0
            for i, row in enumerate(reader):
                if len(row) != len(header) + len(FLAGS) or int(row[state_idx]) != int(ref[i]):
                    raise ValueError('Repaired state not persisted')
                if row[-2:] != [str(int(valid[i])), str(int(input_clock_review))]:
                    raise ValueError('Supervision flags not persisted')
                base = row[:len(header)]
                after.update(json.dumps(base[:state_idx] + base[state_idx+1:], ensure_ascii=False, separators=(',', ':')).encode('utf-8'))
                checked += 1
        if checked != count or after.digest() != unchanged_hash.digest():
            raise ValueError('Non-label CSV field modified')
    return {'rows': count, 'changed_state_rows': changed, 'uncertain_state_rows': int((~valid).sum()),
            'new_csv_sha256': digest(destination), 'nonlabel_field_strings_sha256': after.hexdigest(),
            'nonlabel_fields_exactly_preserved': True, 'input_clock_review': bool(input_clock_review)}


def rebuild_window_meta(frame, result_indices, sidecar_root):
    result = frame.iloc[result_indices].copy().reset_index(drop=True)
    result = result.rename(columns={'cache_index': 'source_cache_index', 'reference_label': 'label',
                                    'normal_origin_reference': 'normal_origin', 'source_family': 'case_level2_label'})
    result.insert(0, 'cache_index', np.arange(len(result)))
    result['level2_label'] = np.where(result.label == 0, 'normal', result.case_level2_label)
    result['phase'] = 'historical_rflysim_normal'
    result['fault_fraction'] = np.nan
    result['fault_onset_row'] = -1
    result['endpoint_fault'] = result.label.astype(int)
    result['phase_basis'] = 'historical_suffix'
    for uid, group in result[result.domain.isin(['SIL', 'HIL'])].groupby('case_uid', sort=False):
        with np.load(sidecar_root / (uid.replace('/', '__') + '.npz'), allow_pickle=False) as d:
            state = d['gtd_state_reference']; first = np.flatnonzero(state == 1)
            onset = int(first[0]) if len(first) else -1
            for i, r in group.iterrows():
                s, e = int(r.window_start_row), int(r.window_end_row)
                seq = state[s:e]
                if len(seq) != 150 or int(seq[-1]) != int(r.label):
                    raise ValueError('Rebuilt metadata/window reference mismatch')
                phase = 'normal_flight' if onset < 0 else ('pre_fault' if e <= onset else ('stable_fault' if s >= onset and np.all(seq == 1) else 'transition'))
                result.at[i, 'phase'] = phase
                result.at[i, 'fault_fraction'] = float((seq == 1).mean())
                result.at[i, 'fault_onset_row'] = onset
                result.at[i, 'phase_basis'] = 'gtd_timestamp_reference'
    return result


def run(args):
    out = args.output.resolve()
    resume = bool(getattr(args, 'resume', False))
    if out.exists() and not resume:
        raise FileExistsError('New independent version required; output already exists')
    if resume and (not out.is_dir() or not (out/'RUNNING.json').exists() or (out/'repair_manifest.json').exists()):
        raise ValueError('Resume requires an incomplete recorded repair run')
    if resume and any((out/name).exists() for name in ['train.pt','val.pt','case_manifest_source_train_val.csv','FAILED.json']):
        raise ValueError('Only interrupted pre-cache CSV phase can resume')
    rawroot = Path('D:/RflyMAD').resolve()
    if out == rawroot or rawroot in out.parents or out == OLD_CACHE.resolve() or OLD_CACHE.resolve() in out.parents:
        raise ValueError('Never modify original raw/cache directory')
    tasks = source_tasks()
    audited = json.loads((OVERLAY / 'review.json').read_text(encoding='utf-8'))
    refs = {r['case_uid']: r for r in (json.loads(line) for line in (LABELS / 'case_results.jsonl').read_text(encoding='utf-8').splitlines())}
    if {t[0]['case_uid'] for t in tasks} != set(refs):
        raise ValueError('Source membership changed')
    for path, expected in audited['input_identities'].items():
        if digest(Path(path)) != expected:
            raise ValueError('Audited input identity changed: ' + path)
    overlays, metadata = {}, {}
    for split in ['train', 'val']:
        opath = OVERLAY / (split + '_overlay.npz'); mpath = OVERLAY / (split + '_metadata.csv')
        if digest(opath) != audited['splits'][split]['overlay_sha256'] or digest(mpath) != audited['splits'][split]['metadata_sha256']:
            raise ValueError('Reviewed overlay changed')
        with np.load(opath, allow_pickle=False) as d:
            overlays[split] = {k: d[k].copy() for k in d.files}
        metadata[split] = pd.read_csv(mpath, low_memory=False)
    if shutil.disk_usage(out.parent).free < 16000000000:
        raise ValueError('Insufficient free space for independent CSVs and caches')
    if not resume:
        out.mkdir(parents=True, exist_ok=False)
        write_json(out / 'RUNNING.json', {'started_at': now(), 'version': VERSION, 'script_sha256': digest(__file__),
            'scope': 'Fixed-crop source train/val label repair only; all retained inputs identical; no training or final-test changes.'})
    else:
        if json.loads((out/'RUNNING.json').read_text(encoding='utf-8'))['version'] != VERSION:
            raise ValueError('Repair version differs from interrupted run')
    try:
        csvresults = []
        pending = tasks
        if resume:
            ledger=out/'case_repair_results.jsonl'
            csvresults=[json.loads(s) for s in ledger.read_text(encoding='utf-8').splitlines()]
            done={r['case_uid']:r for r in csvresults}
            if len(done)!=len(csvresults) or not set(done).issubset(refs):
                raise ValueError('Invalid interrupted membership')
            expected_rows={t[0]['case_uid']:t[0] for t in tasks}
            for r in csvresults:
                if r['historical_csv_sha256']!=refs[r['case_uid']]['processed_csv_sha256'] or r['split']!=expected_rows[r['case_uid']]['split']:
                    raise ValueError('Resume membership/version mismatch')
                if digest(Path(r['final_output_path']))!=r['new_csv_sha256']:
                    raise ValueError('Previously completed repaired CSV changed')
            pending=[t for t in tasks if t[0]['case_uid'] not in done]
            leftovers=[]
            for task in pending:
                row=task[0];target=out/'cases'/row['domain']/Path(row['final_output_path']).name
                if target.exists():
                    saved=out/'interrupted_files'/row['domain']/target.name
                    saved.parent.mkdir(parents=True,exist_ok=True)
                    if saved.exists():raise FileExistsError('Interrupted snapshot already exists')
                    target.replace(saved);leftovers.append(str(saved))
            write_json(out/'RESUME.json',{'resumed_at':now(),'completed_cases_verified':len(csvresults),
                'remaining_cases':len(pending),'preserved_incomplete_outputs':leftovers,'script_sha256':digest(__file__)})
        def patch(task):
            row, old, _ = task; uid = row['case_uid']; r = refs[uid]
            raw = Path(row['source_path']); logs = list((raw / 'Log').glob('*.ulg'))
            if len(logs) != 1 or digest(raw/'TrueData/UAVState_data.xlsx') != r['original_gtd_sha256'] or digest(logs[0]) != r['original_ulog_sha256']:
                raise ValueError('Original GTD/ULog identity changed: ' + uid)
            label_file = LABELS/'label_references'/(uid.replace('/', '__')+'.npz')
            target = out/'cases'/row['domain']/Path(row['final_output_path']).name
            patched = repair_csv(Path(row['final_output_path']), target, label_file, r, r['input_clock_review_required'])
            return {**row, 'historical_final_output_path': row['final_output_path'],
                    'final_output_path': str(target), 'historical_csv_sha256': old['csv_sha256'],
                    **patched, 'output_sha256': patched['new_csv_sha256'],
                    'labeltime_version': VERSION, 'label_reference_source': 'verified_original_GTD_recorded_state',
                    'raw_gtd_sha256': r['original_gtd_sha256'], 'raw_ulog_sha256': r['original_ulog_sha256']}
        with (out/'case_repair_results.jsonl').open('a' if resume else 'x',encoding='utf-8') as stream:
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                for i, r in enumerate(pool.map(patch, pending), len(csvresults)+1):
                    csvresults.append(r); stream.write(json.dumps(r, ensure_ascii=False)+chr(10)); stream.flush()
                    if i % 300 == 0:
                        print('CSV_REPAIRED_AND_VERIFIED', i, '/', len(tasks), flush=True)
        if len(csvresults)!=len(tasks) or len({r['case_uid'] for r in csvresults})!=len(tasks):
            raise ValueError('Incomplete repaired source scope')
        pd.DataFrame(csvresults).to_csv(out/'case_manifest_source_train_val.csv',index=False)
        fixed_paths = {r['case_uid']: r['final_output_path'] for r in csvresults}
        splits = {}
        for split in ['train', 'val']:
            torch.set_num_threads(4)
            source = torch.load(OLD_CACHE/(split+'.pt'),map_location='cpu',weights_only=True,mmap=True)
            compact, chosen = make_split_view(source['x'],source['y'],overlays[split])
            target = out/(split+'.pt'); temporary = out/(split+'.pt.partial')
            torch.save(compact,temporary); temporary.replace(target); del compact
            actual = torch.load(target,map_location='cpu',weights_only=True,mmap=True)
            for start in range(0,len(chosen),512):
                indices = torch.from_numpy(chosen[start:start+512]); piece = actual['x'][start:start+len(indices)]
                if not torch.equal(piece,source['x'][indices]):
                    raise ValueError('Retained input X changed')
            if not torch.equal(actual['y'],torch.from_numpy(overlays[split]['reference_label'][chosen]).long()):
                raise ValueError('Saved repaired targets differ')
            if not set(actual['y'].unique().tolist()).issubset({0,1}):
                raise ValueError('Nonbinary label reached output cache')
            meta = rebuild_window_meta(metadata[split],chosen,LABELS/'label_references')
            meta['source_csv'] = meta.case_uid.map(fixed_paths)
            meta['labeltime_version'] = VERSION
            meta.to_csv(out/(split+'_window_links.csv'),index=False)
            excluded = metadata[split].loc[~overlays[split]['pre_forward_candidate']].copy()
            excluded['exclusion_reason'] = np.where(~excluded.label_reference_defined,'label_source_disagreement','original_inertial_clock_review')
            excluded.to_csv(out/(split+'_excluded_window_ledger.csv'),index=False)
            np.save(out/(split+'_source_cache_indices.npy'),chosen,allow_pickle=False)
            old_y=source['y'][torch.from_numpy(chosen)].clone()
            torch.save(old_y,out/(split+'_historical_y_same_indices.pt'))
            y=actual['y'].numpy(); labels=overlays[split]['reference_label']
            sources=len(meta[meta.domain.isin(['SIL','HIL'])]); extras=len(meta)-sources
            splits[split]={'original_windows':len(source['y']),'repaired_windows':len(y),
                'held_out_windows':len(excluded),'source_windows':sources,'rflysim_suffix_windows':extras,
                'corrected_retained_window_labels':int((old_y.numpy()!=y).sum()),
                'normal_windows':int((y==0).sum()),'fault_windows':int((y==1).sum()),
                'source_gap_diagnostic_flags_retained':int(meta.gap_over_0_1s_flag.eq(True).sum()),
                'retained_X_exactly_equal_to_original':True,'binary_targets_verified':True,
                'cache_sha256':digest(target),'metadata_sha256':digest(out/(split+'_window_links.csv')),
                'source_indices_sha256':digest(out/(split+'_source_cache_indices.npy'))}
            if split=='train' and extras!=3680:
                raise ValueError('Lost historical RflySim suffix')
            del source,actual,old_y
            print('CACHE_REPAIRED',split,json.dumps(splits[split]),flush=True)
        oldnorm=json.loads((OLD_CACHE/'norm_stats.json').read_text(encoding='utf-8'))
        norm={**oldnorm,'protocol_name':VERSION,'source_norm_stats':str(OLD_CACHE/'norm_stats.json'),
              'source_norm_sha256':digest(OLD_CACHE/'norm_stats.json'),
              'normalization_policy':'Historical train-fitted mean/std copied exactly for source-only controlled label comparison; held-out training samples may have contributed to historical fit.',
              'sampling':{**oldnorm.get('sampling',{}),'mode':'original_cap64_indices_then_uncertainty_and_clock_holdout_no_refill'}}
        write_json(out/'norm_stats.json',norm)
        # Post-build checks: old train/val caches, norms, source links, proposal remain unchanged.
        for path,expected in audited['input_identities'].items():
            if digest(Path(path))!=expected:raise ValueError('Original audited input changed: '+path)
        for task in tasks:
            if digest(Path(task[0]['final_output_path']))!=task[1]['csv_sha256']:
                raise ValueError('Original CSV changed after repair')
        snap=out/'source_snapshots';snap.mkdir()
        for path in [Path(__file__),ROOT/'tests/test_fixed_crop_label_repair.py']:
            shutil.copyfile(path,snap/path.name)
        report={'version':VERSION,'status':'COMPLETE_SOURCE_LABELTIME_REPAIR', 'completed_at':now(),
            'source_cases':len(csvresults),'repaired_csv_state_rows':sum(r['changed_state_rows'] for r in csvresults),
            'csv_fields_changed':[STATE,*FLAGS],'all_other_original_csv_fields_exactly_preserved':True,
            'source_cache_splits_repaired':['train','val'],'splits':splits,
            'labeltime_repair_complete':True,'binary_cache_loader_ready':True,
            'model_training_run':False,'strict_online_input_certified':False,'deployment_ready':False,
            'final_test_repaired':False,'raw_originals_overwritten':False,
            'normalization_numeric_parameters_unchanged':all(norm[k]==oldnorm[k] for k in ['columns','mean','std']),
            'input_hashes_before_after_equal':audited['input_identities'],
            'boundaries':[
                'Only audited 26-channel source train/val. Old 31-channel runtime and all Real/final tests unchanged.',
                '151 uncertain endpoint windows and 192 known raw-clock-review windows held out from compact caches, original case splits retained and mappings saved.',
                'All retained X values identical. Legacy interpolation and remaining sampling gaps retained and flagged for retrospective source development; NOT strict online quality approval.',
                'CSV stores GTD state and supervision validity, not a third class. Consumers must honor validity and input-clock flags.',
                'Other label-only command fields, including UAVState_data_cmd, are historical and NOT repaired or valid supervision inputs.',
                'Historical normalization held constant to isolate label effects. Old scores cannot be treated as repaired model results.',
                'No test cache exists here; do not silently mix old final-test labels with this version. Frozen original trainer protocol is not rewritten; new training needs explicit repaired-input configuration.',
                'Replacing a label is not proof of physical health or future performance; GTD is recorded-state reference, disagreement explicitly retained in ledgers.'
            ],'script_sha256':digest(__file__)}
        write_json(out/'repair_manifest.json',report)
        print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)
    except BaseException as exc:
        write_json(out/'FAILED.json',{'failed_at':now(),'error':repr(exc),'traceback':traceback.format_exc(),
            'instruction':'Partial output is not a completed version. Original files untouched; do not train.'})
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=3,choices=range(1,5))
    parser.add_argument('--resume',action='store_true',help='Resume only verified interrupted CSV stage; never overwrite completed output')
    args=parser.parse_args()
    run(args)
