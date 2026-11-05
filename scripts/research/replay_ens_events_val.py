"""ens_event_replay_val_v1: continuous event replay of M0 and ENS on all source-val cases."""
import hashlib, json, pickle, sys, time
from pathlib import Path
import numpy as np, pandas as pd, torch
from numpy.lib.stride_tricks import sliding_window_view

ROOT = Path('D:/UAV-Fault-Diagnosis')
sys.path.insert(0, str(ROOT / 'packages/uav_fault')); sys.path.insert(0, str(ROOT / 'scripts/research'))
from replay_m1_continuous import aggregate_scores  # noqa: E402
from uav_fault.online.detection_policy import OnlineDetectionPolicy  # noqa: E402
from uav_fault.models.cnn.lstm import OnlineMSTCA  # noqa: E402

DATA = Path('D:/UAV-Data/causal_input_v1')
OUT = ROOT / 'artifacts/ens_event_replay_val_v1_20261106'
M0_PT = ROOT / 'artifacts/causal_pure_mstca_m0_v1_20261101/results/runs/balanced_source_normal/best.pt'
GBT_PKL = ROOT / 'artifacts/ens_internal_test_v1_20261105/gbt_full_train.pkl'
CFG = ROOT / 'configs/experiments/ens_event_replay_val_v1.json'
W = 150


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(4 << 20), b''):
            h.update(b)
    return h.hexdigest()


def stats(x):
    d = np.abs(np.diff(x, axis=2))
    return np.concatenate([x.mean(2), x.std(2), x.min(2), x.max(2), d.mean(2)], axis=1)


def case_windows(rows, valid):
    """Return end-row indices of valid windows and their raw float32 windows (N, 26, 150)."""
    out = rows.astype(np.float64).copy()
    out[:, 25] = np.sqrt((out[:, 11:14] ** 2).sum(1))
    bad = np.r_[0, np.cumsum(~valid)]
    starts = np.arange(0, len(valid) - W + 1)
    ok = (bad[starts + W] - bad[starts]) == 0
    starts = starts[ok]
    v = sliding_window_view(out, W, axis=0)[starts]  # (N, 26, 150)
    v = v.copy(); v[:, 14, :] -= v[:, 14, :1]
    return starts + W - 1, v.astype(np.float32)


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    cfg = json.loads(CFG.read_text(encoding='utf-8'))
    assert sha(M0_PT) == cfg['models']['M0'].split('sha256 ')[1].split(')')[0]
    assert sha(GBT_PKL) == cfg['models']['GBT'].split('sha256 ')[1].split(')')[0]
    st = json.load(open(DATA / 'norm_stats.json'))
    mu = np.asarray(st['mean'], np.float32)[None, :, None]; sd = np.asarray(st['std'], np.float32)[None, :, None]
    gbt = pickle.load(open(GBT_PKL, 'rb'))
    m0 = OnlineMSTCA(in_channels=26, hidden_size=64, num_layers=1, num_classes=2, dropout=0.2).float()
    m0.load_state_dict(torch.load(M0_PT, map_location='cpu', weights_only=True)); m0.eval()
    torch.set_num_threads(16)
    policy = OnlineDetectionPolicy(**{k: v for k, v in {'inference_every': 1, 'onset_th': .6, 'clear_th': .4, 'k_on': 3, 'k_off': 3, 'min_event_s': .5, 'merge_gap_s': .5}.items()})
    tasks = [t for t in json.load(open(DATA / 'tasks.json')) if t['split'] == 'val']
    assert len(tasks) == 555

    # Parity: cap64 windows vs cache val x, and M0 argmax vs saved predictions.
    cache = torch.load(DATA / 'cache/val.pt', weights_only=True, mmap=True)
    cpos = {int(v): i for i, v in enumerate(cache['original_cache_index'].numpy())}
    saved = np.load(ROOT / 'artifacts/causal_pure_mstca_m0_v1_20261101/results/runs/balanced_source_normal/source_validation_predictions.npz')
    spos = {int(v): i for i, v in enumerate(saved['original_indices'])}
    parity = {'windows': 0, 'max_abs_x_diff': 0.0, 'argmax_mismatch': 0}

    rows_out = []; t0 = time.time()
    for k, t in enumerate(tasks):
        uid = t['case_uid']; key = uid.replace('/', '__')
        with np.load(DATA / 'cases' / (key + '.npz')) as z:
            grid = z['timestamp_us'] / 1e6; rows = z['canonical_rows']; valid = z['row_input_valid']
            ref = z['reference_state']; lv = z['row_label_valid']
        ends, xr = case_windows(rows, valid)
        n = len(grid); pm0 = np.full(n, np.nan); pg = np.full(n, np.nan)
        if len(ends):
            xn = (xr - mu) / sd
            with torch.no_grad():
                p = np.concatenate([m0(torch.from_numpy(xn[i:i + 2048])).softmax(1)[:, 1].numpy() for i in range(0, len(xn), 2048)])
            g = np.concatenate([gbt.predict_proba(stats(xr[i:i + 4096]))[:, 1] for i in range(0, len(xr), 4096)])
            pm0[ends] = p; pg[ends] = g
            wpath = DATA / 'windows' / (key + '.npz')
            if wpath.exists():
                with np.load(wpath) as wz:
                    emap = {int(e): j for j, e in enumerate(ends)}
                    for s_, oi in zip(wz['window_start_row'], wz['original_cache_index']):
                        j = emap.get(int(s_) + W - 1); ci = cpos.get(int(oi))
                        if j is None or ci is None:
                            continue
                        parity['windows'] += 1
                        parity['max_abs_x_diff'] = max(parity['max_abs_x_diff'], float(np.abs(xr[j] - cache['x'][ci].numpy()).max()))
                        sp = saved['probabilities'][spos[int(oi)]]
                        parity['argmax_mismatch'] += int((p[j] > 1 - p[j]) != (sp[1] > sp[0]))
        onset_idx = np.flatnonzero(lv & (ref == 1))
        onset = float(grid[onset_idx[0]]) if len(onset_idx) else None
        normal_row = lv & (ref == 0)
        scored = np.isfinite(pm0)
        for arm, prob in [('M0', pm0), ('ENS', (pm0 + pg) / 2)]:
            ev = aggregate_scores(grid, prob, policy=policy)
            fin = ev['finalized_events']
            false = []
            for e in fin:
                m = (grid >= e['start_t']) & (grid <= e['end_t'])
                if m.any() and normal_row[m].all():
                    false.append(e)
            occ = np.zeros(n, bool)
            for e in false:
                occ |= (grid >= e['start_t']) & (grid <= e['end_t'])
            exposure = normal_row & scored
            r = {'case_uid': uid, 'domain': t['domain'], 'flight': uid.split('/')[1], 'cls': uid.split('/')[2], 'arm': arm,
                 'onset_s': onset, 'normal_exposure_s': float(exposure.sum() * 0.02), 'false_events': len(false),
                 'false_occupancy_s': float((occ & exposure).sum() * 0.02), 'scored_frames': int(scored.sum())}
            if onset is not None:
                conf = [c for c in ev['confirmations'] if c['candidate_start_t'] >= onset]
                post = [e for e in fin if e['start_t'] >= onset]
                r.update(detected=bool(conf), confirm_latency_s=conf[0]['confirmed_t'] - onset if conf else None,
                         final_emit_latency_s=post[0]['emitted_t'] - onset if post else None,
                         final_emit_reason=post[0]['emission_reason'] if post else None,
                         event_spans_onset=any(e['start_t'] < onset <= e['end_t'] for e in fin),
                         post_onset_scored=bool((scored & (grid >= onset)).any()))
            rows_out.append(r)
        if k % 50 == 0:
            print(k, uid, round(time.time() - t0), 's', flush=True)
    df = pd.DataFrame(rows_out)
    df.to_csv(OUT / 'per_case.csv', index=False)
    json.dump({'parity': parity, 'config_sha256': sha(CFG), 'cases': len(tasks), 'elapsed_s': time.time() - t0},
              open(OUT / 'run_receipt.json', 'w'), indent=2)
    print(json.dumps(parity))


if __name__ == '__main__':
    main()
