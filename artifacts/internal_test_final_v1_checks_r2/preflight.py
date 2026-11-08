"""PI supplement 1: full imports + frozen models + two VAL frame parities.
Run with -X utf8 -E -P -B. Never reads test predictions/tasks/ledger/payloads.
"""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import importlib
import json
import os
import pickle
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
CHECKS = ROOT / 'artifacts/internal_test_final_v1_checks_r2'
DATA = Path('D:/UAV-Data/causal_input_v1')
CACHE = Path('D:/UAV-Data/ens_event_threshold_v1_scores')
sys.path[:0] = [str(ROOT / 'scripts/research'), str(ROOT / 'packages/uav_fault')]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(4 << 20), b''):
            h.update(b)
    return h.hexdigest()


def run():
    target = CHECKS / 'preflight.json'
    if target.exists():
        raise FileExistsError('Preserve preflight outcome; no silent rerun')
    state = {'status': 'RUNNING', 'time': datetime.now(timezone.utc).isoformat(),
             'python': sys.executable, 'flags': {'ignore_environment': sys.flags.ignore_environment,
             'safe_path': sys.flags.safe_path, 'isolated': sys.flags.isolated,
             'no_user_site': sys.flags.no_user_site, 'utf8_mode': sys.flags.utf8_mode,
             'dont_write_bytecode': sys.dont_write_bytecode}, 'imports': [], 'cases': [], 'blocked_accesses': []}
    start = time.perf_counter()
    try:
        assert sys.flags.ignore_environment and sys.flags.safe_path and not sys.flags.isolated
        assert not sys.flags.no_user_site and sys.flags.utf8_mode and sys.dont_write_bytecode
        cfg_path = ROOT / 'configs/experiments/internal_test_final_v1.json'
        cfg = json.loads(cfg_path.read_text(encoding='utf-8'))
        models = {k: {'path': ROOT / text.split(', sha256 ')[0],
                     'expected': re.search(r'sha256 ([0-9a-f]{64})', text).group(1)}
                  for k, text in cfg['frozen_models'].items()}
        allowed = {(DATA / 'norm_stats.json').resolve(), (CACHE / 'index.csv').resolve()}
        model_paths = {v['path'].resolve() for v in models.values()}
        old_test = (ROOT / 'artifacts/ens_internal_test_v1_20261105').resolve()
        def boundary(event, args):
            if event != 'open' or not isinstance(args[0], (str, bytes, os.PathLike)):
                return
            p = Path(os.fsdecode(args[0])).resolve()
            writing = any(c in (args[1] or '') for c in 'wax+') or args[2] & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND)
            if ((p.is_relative_to(DATA.parent.resolve()) and (p not in allowed or writing))
                    or (p in model_paths and writing) or (p.is_relative_to(old_test) and p not in model_paths)):
                state['blocked_accesses'].append(str(p))
                raise PermissionError('Preflight is VAL-only/read-only; never test or Real: ' + str(p))
        sys.addaudithook(boundary)
        names = ['numpy', 'pandas', 'torch', 'sklearn.ensemble', 'threadpoolctl',
                 'uav_fault.causal_input', 'uav_fault.models.cnn.lstm',
                 'uav_fault.online.detection_policy', 'uav_fault.online.event_aggregator',
                 'uav_fault.online.monitoring_session', 'uav_fault.online.fault_detector',
                 'replay_m1_continuous', 'replay_ens_events_val', 'score_ens_frames',
                 'ens_event_threshold', 'gbt_event_operating_point_v1', 'internal_test_final_v1']
        for name in names:
            module = importlib.import_module(name)
            state['imports'].append({'module': name, 'file': getattr(module, '__file__', None)})
        import numpy as np
        import pandas as pd
        import torch
        import internal_test_final_v1 as final
        from score_ens_frames import load_m0
        state['config_sha256'] = sha(cfg_path)
        state['models_before'] = {k: sha(v['path']) for k, v in models.items()}
        assert all(state['models_before'][k] == v['expected'] for k, v in models.items())
        torch.set_num_threads(16)
        m0 = load_m0(models['M0_baseline']['path'])
        with models['GBT']['path'].open('rb') as f:
            gbt = pickle.load(f)
        state['models_after_load'] = {k: sha(v['path']) for k, v in models.items()}
        assert state['models_after_load'] == state['models_before']
        idx = pd.read_csv(CACHE / 'index.csv', dtype={'case_uid': str, 'split': str, 'domain': str})
        selected = []
        for domain in ('SIL', 'HIL'):
            rows = idx[idx.split.eq('val') & idx.domain.eq(domain)].sort_values('case_uid')
            assert len(rows) and rows.case_uid.is_unique and rows.fold.eq(-1).all()
            uid = rows.iloc[0].case_uid
            assert len(uid.split('/')) == 4 and uid.split('/')[0] == domain
            assert not any(p in ('', '.', '..') for p in uid.split('/'))
            selected.append(uid)
            key = uid.replace('/', '__') + '.npz'
            allowed.update({(DATA / 'cases' / key).resolve(), (CACHE / key).resolve()})
        norm = json.loads((DATA / 'norm_stats.json').read_text(encoding='utf-8'))
        mu = np.asarray(norm['mean'], np.float32)[None, :, None]
        sd = np.asarray(norm['std'], np.float32)[None, :, None]
        assert norm['fit_split'] == 'source_train_only' and mu.shape == sd.shape == (1, 26, 1)
        assert np.isfinite(mu).all() and np.isfinite(sd).all() and (sd > 0).all()
        for uid in selected:
            key = uid.replace('/', '__') + '.npz'
            with np.load(DATA / 'cases' / key, allow_pickle=False) as z:
                times, rows, valid = z['timestamp_us'], z['canonical_rows'], z['row_input_valid']
            pm, pg = final.score_arrays(rows, valid, m0, gbt, mu, sd)
            with np.load(CACHE / key, allow_pickle=False) as z:
                assert np.array_equal(times, z['timestamp_us'])
                old_m0, old_gbt = z['m0'], z['gbt']
            mask = np.isfinite(pm)
            assert mask.any() and np.array_equal(mask, np.isfinite(pg))
            assert np.array_equal(mask, np.isfinite(old_m0)) and np.array_equal(mask, np.isfinite(old_gbt))
            dm = float(np.max(np.abs(pm[mask].astype(float) - old_m0[mask].astype(float))))
            dg = float(np.max(np.abs(pg[mask].astype(float) - old_gbt[mask].astype(float))))
            item = {'case_uid': uid, 'split': 'val', 'scored_frames': int(mask.sum()),
                    'M0_max_abs_difference': dm, 'M0_limit': .002, 'GBT_max_abs_difference': dg,
                    'GBT_limit': 1e-6, 'same_code': 'internal_test_final_v1.score_arrays',
                    'cached_score_sha256': sha(CACHE / key), 'pass': dm <= .002 and dg <= 1e-6}
            state['cases'].append(item)
            assert item['pass'], 'VAL full-frame parity failed; never proceed to test'
        state['models_after_scoring'] = {k: sha(v['path']) for k, v in models.items()}
        assert state['models_after_scoring'] == state['models_before'] and not state['blocked_accesses']
        state.update(status='PASS', test_data_read=False, no_training_or_fit=True, scoring_device='cpu',
                     torch_threads=16, M0_batch=2048, GBT_batch=4096, wall_time_s=time.perf_counter()-start)
    except Exception as exc:
        state.update(status='FAIL_STOP', error=repr(exc), wall_time_s=time.perf_counter()-start)
        raise
    finally:
        with target.open('x', encoding='utf-8') as f:
            json.dump(state, f, ensure_ascii=False, indent=2, allow_nan=False); f.write('\n')
        print(json.dumps(state, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    run()
