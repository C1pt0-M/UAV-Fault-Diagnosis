#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def _add_import_paths():
    repo = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo / 'src'))
    sys.path.insert(0, str(repo / 'src' / 'models'))
    sys.path.insert(0, str(repo / 'src' / 'online'))


def _parse_list(s, cast):
    return [cast(x.strip()) for x in str(s).split(',') if x.strip()]


def _times(df):
    if 'trueTime' in df.columns:
        return pd.to_numeric(df['trueTime'], errors='coerce').ffill().bfill().fillna(0).to_numpy(dtype=float)
    return np.arange(len(df), dtype=float) / 50.0


def _gt_mask(df):
    for col in ('UAVState_data_fault_state', 'TrueState_data_fault_state'):
        if col in df.columns:
            return (pd.to_numeric(df[col], errors='coerce').fillna(0).to_numpy() > 0).astype(np.int64)
    if '_rfly_ctrl_lxl_0_id' in df.columns:
        ids = pd.to_numeric(df['_rfly_ctrl_lxl_0_id'], errors='coerce').fillna(0).astype(np.int64).to_numpy()
        return (((ids >= 123450) & (ids <= 123459)) | ((ids >= 123540) & (ids <= 123549))).astype(np.int64)
    return np.zeros(len(df), dtype=np.int64)


def _spans(mask, times):
    out, start = [], None
    for i, active in enumerate(mask.astype(bool)):
        if active and start is None:
            start = i
        elif (not active) and start is not None:
            out.append((float(times[start]), float(times[i - 1])))
            start = None
    if start is not None and len(mask):
        out.append((float(times[start]), float(times[len(mask) - 1])))
    return out


def _load_model(model_dir, norm_stats_path, device):
    from cnn.cnn_lstm import CNNLSTMClassifier
    from cnn.textcnn import TextCNN1d
    from cnn.transformer import TransformerClassifier
    cfg = pd.read_json(model_dir / 'config.json', typ='series').to_dict()
    stats = pd.read_json(norm_stats_path, typ='series').to_dict()
    model_name = cfg['model']
    if model_name == 'textcnn':
        model = TextCNN1d(in_channels=cfg['in_channels'], num_classes=cfg['num_classes'], num_filters=128, kernel_sizes=(3, 5, 7, 11, 21), dropout=0.3)
    elif model_name == 'cnn_lstm':
        model = CNNLSTMClassifier(in_channels=cfg['in_channels'], cnn_channels=(64, 128, 128), lstm_hidden=128, lstm_layers=2, num_classes=cfg['num_classes'], dropout=0.3)
    elif model_name == 'transformer':
        model = TransformerClassifier(in_channels=cfg['in_channels'], d_model=128, nhead=4, num_layers=4, num_classes=cfg['num_classes'], patch_size=8, dropout=0.2)
    else:
        raise ValueError(f'unsupported model: {model_name}')
    state = torch.load(model_dir / 'best.pt', map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device).eval()
    mean = np.asarray(stats['mean'], dtype=np.float32)
    std = np.asarray(stats['std'], dtype=np.float32)
    std = np.where(std > 1e-8, std, 1.0)
    return model, mean, std, cfg


def _build_records(files, model, mean, std, seq_len, every, device):
    from data.build_sequence_cache import _extract_channels
    rows = []
    for p in files:
        df = pd.read_csv(p)
        if len(df) < seq_len:
            continue
        feats = _extract_channels(df).astype(np.float32)
        times = _times(df)
        gt = _spans(_gt_mask(df), times)
        pts = []
        with torch.no_grad():
            for end in range(seq_len, len(df) + 1, every):
                x = feats[end - seq_len:end].T
                x = (x - mean[:, None]) / std[:, None]
                logits = model(torch.from_numpy(x[None]).to(device))
                probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()
                pts.append((float(times[end - 1]), float(1.0 - probs[0]), int(end - 1)))
        rows.append({'file': p.name, 'gt': gt, 'pts': pts})
    return rows


def _score(records, params):
    from event_aggregator import FaultEventAggregator
    tp = tn = fp = fn = hits = misses = false_alarms = 0
    latencies = []
    for item in records:
        agg = FaultEventAggregator(**params)
        events = []
        for t, prob, idx in item['pts']:
            events.extend(agg.update(t=t, fault_prob=prob, frame_idx=idx))
        events.extend(agg.flush())
        gt = item['gt']
        gt_fault, pred_fault = bool(gt), bool(events)
        if gt_fault and pred_fault: tp += 1
        elif gt_fault and (not pred_fault): fn += 1
        elif (not gt_fault) and pred_fault: fp += 1
        else: tn += 1
        hit = any(e.end_t >= s and e.start_t <= e2 for e in events for s, e2 in gt)
        hits += int(hit)
        misses += int(gt_fault and not hit)
        false_alarms += int(pred_fault and not hit)
        if hit and events and gt:
            latencies.append(float(events[0].start_t - gt[0][0]))
    known = max(1, tp + tn + fp + fn)
    return {
        **params, 'files': len(records), 'accuracy': (tp + tn) / known,
        'precision': tp / max(1, tp + fp), 'recall': tp / max(1, tp + fn),
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn, 'hits': hits,
        'misses': misses, 'false_alarms': false_alarms,
        'latency_mean': float(np.mean(latencies)) if latencies else np.nan,
        'latency_median': float(np.median(latencies)) if latencies else np.nan,
    }


def main():
    _add_import_paths()
    p = argparse.ArgumentParser(description='Sweep Rfly event aggregation params on cached model fault_prob streams.')
    p.add_argument('--csv-dir', default='/home/jack/UVAdetect-main/raw_data')
    p.add_argument('--glob', default='Case_*.csv')
    p.add_argument('--model-dir', default='models/transformer')
    p.add_argument('--norm-stats', default='data/processed/seq_cache/norm_stats.json')
    p.add_argument('--device', default='cpu')
    p.add_argument('--window', type=int, default=2048)
    p.add_argument('--every', type=int, default=50)
    p.add_argument('--limit-files', type=int, default=0)
    p.add_argument('--onset-ths', default='0.50,0.60,0.70')
    p.add_argument('--clear-ths', default='0.30,0.40,0.50')
    p.add_argument('--k-ons', default='1,2,3')
    p.add_argument('--k-offs', default='1,2,3')
    p.add_argument('--min-event-ss', default='0.0,0.5,1.0')
    p.add_argument('--merge-gap-ss', default='0.0,0.5,1.0')
    p.add_argument('--out', default='logs/rfly_event_sweep.csv')
    args = p.parse_args()
    files = sorted(Path(args.csv_dir).glob(args.glob))
    if args.limit_files > 0: files = files[:args.limit_files]
    device = torch.device(args.device)
    model, mean, std, cfg = _load_model(Path(args.model_dir), Path(args.norm_stats), device)
    records = _build_records(files, model, mean, std, args.window, args.every, device)
    rows = []
    for vals in product(_parse_list(args.onset_ths, float), _parse_list(args.clear_ths, float), _parse_list(args.k_ons, int), _parse_list(args.k_offs, int), _parse_list(args.min_event_ss, float), _parse_list(args.merge_gap_ss, float)):
        onset, clear, k_on, k_off, min_s, gap_s = vals
        if clear > onset: continue
        rows.append(_score(records, {'onset_th': onset, 'clear_th': clear, 'k_on': k_on, 'k_off': k_off, 'min_event_s': min_s, 'merge_gap_s': gap_s}))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows).sort_values(['recall', 'precision', 'accuracy', 'latency_mean'], ascending=[False, False, False, True])
    df.to_csv(out, index=False)
    print(f"[OK] model={cfg['model']} files={len(records)} wrote {out}")
    print(df.head(10).to_string(index=False))


if __name__ == '__main__':
    raise SystemExit(main())
