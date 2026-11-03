"""One frozen, CPU-only comparison of conditional and unconditional N references.

Statistical reference fitting only; no classifier/regression/neural training,
model forward, old9 computation, server, GPU, Test/Real or rule search.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ.setdefault('OMP_NUM_THREADS', '2')
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import sys
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/research'))
from collect_m3_formal_results import observed_states
FEATURES = ['N1', 'N2', 'N3', 'N4']
FAULTS = ['GPS', 'accelerometer', 'barometer', 'gyroscope', 'magnetometer', 'motor']


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def case_weights(cases):
    s = pd.Series(np.asarray(cases))
    return 1.0 / s.map(s.value_counts()).to_numpy(dtype=float)


def weighted_quantile(values, weights, q):
    values, weights = np.asarray(values, dtype=float), np.asarray(weights, dtype=float)
    if values.ndim != 1 or values.shape != weights.shape or not len(values):
        raise ValueError('Nonempty equal-length quantile inputs required')
    if not np.isfinite(values).all() or not np.isfinite(weights).all() or not (weights > 0).all() or not 0 <= q <= 1:
        raise ValueError('Finite values, positive weights and a fixed quantile required')
    order = np.argsort(values, kind='stable')
    cumulative = np.cumsum(weights[order])
    index = min(int(np.searchsorted(cumulative, q * cumulative[-1], side='left')), len(values) - 1)
    return float(values[order[index]])


def fit_reference(values, cases, quantiles):
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or values.shape[1] != 4 or len(values) != len(cases):
        raise ValueError('Only four signed residual summaries allowed')
    weights = case_weights(cases)
    reference = np.array([[weighted_quantile(values[:, j], weights, q) for q in quantiles] for j in range(4)])
    if not np.isfinite(reference).all() or not ((reference[:, 0] < reference[:, 1]) & (reference[:, 1] < reference[:, 2])).all():
        raise ValueError('Degenerate reference widths; no epsilon fallback')
    return reference


def score_reference(values, reference):
    values, reference = np.asarray(values, dtype=float), np.asarray(reference, dtype=float)
    if values.ndim != 2 or values.shape[1] != 4 or reference.shape != (4, 3) or not np.isfinite(values).all():
        raise ValueError('Finite four-dimensional residuals/reference required')
    lower, center, upper = reference.T
    if not np.isfinite(reference).all() or not ((lower < center) & (center < upper)).all():
        raise ValueError('Positive two-sided reference widths required')
    delta = values - center
    per_feature = np.where(delta >= 0, delta / (upper - center), -delta / (center - lower))
    return per_feature.max(1), per_feature


def case_balanced_auc(labels, scores, cases):
    labels, scores, cases = np.asarray(labels), np.asarray(scores), np.asarray(cases)
    if not np.isfinite(scores).all() or not np.isin(labels, [0, 1]).all():
        raise ValueError('Finite scores and known binary labels required')
    if not (labels == 0).any() or not (labels == 1).any():
        return None
    weights = np.empty(len(labels))
    for label in [0, 1]:
        mask = labels == label
        weights[mask] = case_weights(cases[mask])
    order = np.argsort(scores, kind='stable')
    scores, labels, weights = scores[order], labels[order], weights[order]
    ends = np.r_[np.flatnonzero(scores[:-1] != scores[1:]) + 1, len(scores)]
    neg_before, numerator, begin = 0.0, 0.0, 0
    for end in ends:
        y, w = labels[begin:end], weights[begin:end]
        negative, positive = w[y == 0].sum(), w[y == 1].sum()
        numerator += positive * (neg_before + .5 * negative)
        neg_before += negative
        begin = end
    return float(numerator / (weights[labels == 1].sum() * weights[labels == 0].sum()))


def load_split(a, split, identities):
    # Do not follow any Test/Real path or evaluate any model.
    if split not in ['train', 'val']:
        raise ValueError('Source Train/Val only')
    parent = read(ROOT / 'configs/experiments/M3_preflight_v1.json')
    expected = parent['data']['source_split_identities'][split]
    frozen = read(ROOT / 'configs/experiments/causal_pure_mstca_m0_v1.json')['data_identities']
    bp, rp = a.data / 'cache' / (split + '.pt'), a.features / 'cache' / (split + '.pt')
    lp = a.data / 'ledgers' / (split + '_included_windows.csv')
    npth = a.data / 'norm_stats.json'
    for path, wanted in [(bp, expected['base_cache_sha256']), (rp, expected['residual_cache_sha256']),
                         (lp, frozen['ledgers/' + split + '_included_windows.csv']), (npth, frozen['norm_stats.json'])]:
        actual = sha(path)
        if actual != wanted:
            raise ValueError('Frozen input SHA differs: ' + str(path))
        identities[str(path)] = actual
    norm = read(npth)
    base = torch.load(bp, map_location='cpu', weights_only=True, mmap=True)
    residual = torch.load(rp, map_location='cpu', weights_only=True, mmap=True)
    ix = residual['base_compact_index'].numpy()
    frame = pd.read_csv(lp, keep_default_na=False, dtype={'association_group': str}).iloc[ix].reset_index(drop=True)
    if len(frame) != expected['windows'] or not frame.split.eq(split).all() or not frame.domain.isin(['SIL', 'HIL']).all():
        raise ValueError('Wrong source population')
    if not np.array_equal(frame.original_cache_index, residual['original_cache_index'].numpy()) or not np.array_equal(frame.qualified_y, residual['y'].numpy()):
        raise ValueError('Original ID/label/order differs')
    if not torch.equal(base['original_cache_index'][ix], residual['original_cache_index']) or not residual['full_window_valid'].all():
        raise ValueError('Not the frozen joint150 domain')
    for values, key in [(frame.original_cache_index, 'original_ids_sha256'), (frame.qualified_y, 'labels_sha256')]:
        if hashlib.sha256(np.asarray(values, dtype=np.int64).tobytes()).hexdigest() != expected[key]:
            raise ValueError('Frozen order digest differs')
    states = np.empty(len(frame), dtype=object)
    summaries = np.empty((len(frame), 4), dtype=float)
    for begin in range(0, len(frame), 512):
        end = min(len(frame), begin + 512)
        # State call receives only the current past150 physical input, no metadata.
        states[begin:end] = observed_states(base['x'][ix[begin:end]].numpy(), norm['columns'])
        values = residual['residual'][begin:end].numpy().astype(np.float64)
        if not np.isfinite(values).all() or values.shape[1:] != (4, 150):
            raise ValueError('Invalid original signed N data')
        summaries[begin:end] = np.median(values, axis=2)
    frame['observed_state'] = states
    frame['whole_known_normal'] = frame.qualified_y.eq(0) & frame.whole_window_label_valid.eq(True) & frame.fault_fraction.eq(0)
    frame['pure_fault'] = frame.qualified_y.eq(1) & frame.whole_window_label_valid.eq(True) & frame.fault_fraction.eq(1)
    for j, feature in enumerate(FEATURES):
        frame[feature] = summaries[:, j]
    return frame


def metrics(frame, minimum):
    normal = frame.qualified_y.eq(0).to_numpy(); fault = ~normal
    u, c = frame.unconditional_abnormal.to_numpy(dtype=bool), frame.conditional_abnormal.to_numpy(dtype=bool)
    row = dict(windows=len(frame), normal_windows=int(normal.sum()), fault_windows=int(fault.sum()),
               normal_cases=int(frame.loc[normal, 'case_uid'].nunique()), fault_cases=int(frame.loc[fault, 'case_uid'].nunique()),
               normal_association_groups=int(frame.loc[normal, 'association_group'].nunique()),
               fault_association_groups=int(frame.loc[fault, 'association_group'].nunique()),
               unconditional_FP=int(u[normal].sum()), conditional_FP=int(c[normal].sum()),
               unconditional_TP=int(u[fault].sum()), conditional_TP=int(c[fault].sum()),
               FN_to_TP=int((fault & ~u & c).sum()), TP_to_FN=int((fault & u & ~c).sum()),
               FP_to_TN=int((normal & u & ~c).sum()), TN_to_FP=int((normal & ~u & c).sum()))
    row['adequate_descriptive_cell'] = all(row[k] >= v for k, v in minimum.items())
    row['case_balanced_AUC_unconditional'] = case_balanced_auc(frame.qualified_y, frame.unconditional_score, frame.case_uid) if len(frame) else None
    row['case_balanced_AUC_conditional'] = case_balanced_auc(frame.qualified_y, frame.conditional_score, frame.case_uid) if len(frame) else None
    for name, mask in [('normal_FPR', normal), ('fault_Recall', fault)]:
        for method, flag in [('unconditional', u), ('conditional', c)]:
            row[method + '_' + name] = float(flag[mask].mean()) if mask.any() else None
            if mask.any():
                f = frame.loc[mask, ['case_uid', 'association_group']].copy()
                f['flag'] = flag[mask].astype(float)
                percase = f.groupby(['case_uid', 'association_group']).flag.mean().reset_index()
                row[method + '_' + name + '_case_balanced'] = float(percase.flag.mean())
                row[method + '_' + name + '_association_balanced'] = float(percase.groupby('association_group').flag.mean().mean())
            else:
                row[method + '_' + name + '_case_balanced'] = None
                row[method + '_' + name + '_association_balanced'] = None
    return row


def report_tables(frame, protocol):
    rows = []
    minimum = protocol['Val_descriptive_cell_minimum']
    def add(scope, keys, subset):
        rows.append(dict(scope=scope, **keys, **metrics(subset, minimum)))
    add('overall', {}, frame)
    add('pure_fault_plus_all_normal', {}, frame[frame.pure_fault | frame.qualified_y.eq(0)])
    for domain in ['SIL', 'HIL']:
        add('domain', {'domain': domain}, frame[frame.domain.eq(domain)])
    for state, part in frame.groupby('observed_state'):
        add('state', {'state': state}, part)
        for domain in ['SIL', 'HIL']:
            add('state_domain', {'state': state, 'domain': domain}, part[part.domain.eq(domain)])
    normal = frame.qualified_y.eq(0)
    for fault in FAULTS:
        population = normal | frame.fault_directory.eq(fault)
        add('fault_vs_same_normal_pool', {'fault': fault}, frame[population])
        for domain in ['SIL', 'HIL']:
            add('fault_domain', {'fault': fault, 'domain': domain}, frame[population & frame.domain.eq(domain)])
            for state in protocol['states']:
                part = frame[population & frame.domain.eq(domain) & frame.observed_state.eq(state)]
                add('fault_state_domain', {'fault': fault, 'domain': domain, 'state': state}, part)
    # Directory information is introduced only now for reporting, never references.
    hover = frame.flight_directory.eq('hover')
    add('hover_all', {}, frame[hover])
    for fault in ['GPS', 'gyroscope', 'magnetometer']:
        population = hover & (normal | frame.fault_directory.eq(fault))
        add('hover_fault', {'fault': fault}, frame[population])
        for domain in ['SIL', 'HIL']:
            for state in ['all_supported', 'low_motion_proxy']:
                part = frame[population & frame.domain.eq(domain)]
                if state != 'all_supported': part = part[part.observed_state.eq(state)]
                add('hover_fault_domain', {'fault': fault, 'domain': domain, 'state': state}, part)
    for origin, part in frame[normal].groupby('normal_origin'):
        add('normal_origin', {'origin': origin}, part)
    return pd.DataFrame(rows)


def plot_results(output):
    """One descriptive figure; no new score/rule or result selection."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image, ImageOps
    sys.path.insert(0, 'C:/Users/23119/.agents/skills/scipilot-figure-skill/scripts')
    from visual_qa import audit_layout
    destination = output / 'figures'
    if destination.exists(): raise FileExistsError('Preserve previous figures')
    destination.mkdir()
    frame = pd.read_csv(output / 'paired_Val_metrics.csv')
    order = ['climb_proxy', 'low_motion_proxy', 'maneuver_proxy', 'other_motion', 'translation_proxy']
    state = frame[frame.scope.eq('state')].set_index('state').loc[order]
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'svg.fonttype':'none',
                         'axes.spines.top':False,'axes.spines.right':False})
    fig, axes = plt.subplots(1, 3, figsize=(11, 4.2))
    fig.subplots_adjust(left=.065, right=.985, top=.83, bottom=.29, wspace=.29)
    panels = [('normal_FPR', 'Normal falsely flagged (%)', 100),
              ('fault_Recall', 'Fault flagged (%)', 100),
              ('AUC', 'Case-balanced ROC AUC', 1)]
    for j, (name, ylabel, multiplier) in enumerate(panels):
        ax = axes[j]
        for offset, method, color, marker in [(-.09, 'unconditional', '#0072B2', 'o'),
                                               (.09, 'conditional', '#D55E00', '^')]:
            column = 'case_balanced_AUC_' + method if name == 'AUC' else method + '_' + name
            ax.scatter(np.arange(5) + offset, state[column] * multiplier, color=color, marker=marker,
                       s=34, label=method.capitalize(), zorder=3)
        ax.set_xticks(range(5), ['Climb', 'Low\nmotion', 'Maneuver', 'Other', 'Translation'], fontsize=7)
        ax.set_ylabel(ylabel); ax.set_title(chr(97+j) + '  ' + ylabel, fontsize=9)
        if name == 'AUC':
            ax.set_ylim(0, 1.02); ax.axhline(.5, color='0.5', ls='--', lw=.7)
        else: ax.set_ylim(bottom=0)
        ax.grid(axis='y', alpha=.2)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=2, loc='lower center', bbox_to_anchor=(.5,.10), frameon=False)
    fig.suptitle('Frozen N1–N4 normal-reference comparison: state-specific trade-offs', y=.97)
    fig.text(.5,.04,'Same 32,165 supported Val windows; 216 descent fault windows unavailable. Fixed marginal 99% bounds, no threshold search.',
             ha='center', fontsize=7)
    fig.canvas.draw(); issues = audit_layout(fig)
    if any(severity == 'FAIL' for severity, _ in issues): raise ValueError(issues)
    fig.savefig(destination/'conditional_reference_tradeoffs.png', dpi=300)
    ImageOps.grayscale(Image.open(destination/'conditional_reference_tradeoffs.png')).save(destination/'conditional_reference_tradeoffs_grayscale.png')
    fig.savefig(destination/'conditional_reference_tradeoffs.svg'); plt.close(fig)
    write(destination/'visual_qa.json', {'programmatic_layout_issues':issues,
                                       'manual_image_review':'Not asserted; image attachment may be unavailable to this agent.'})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('D:/UAV-Data/causal_input_v1'))
    parser.add_argument('--features', type=Path, default=Path('D:/UAV-Data/causal_m1_continuous_v1'))
    parser.add_argument('--output', type=Path, required=True)
    a = parser.parse_args(); torch.set_num_threads(2)
    protocol_path = a.output / 'protocol.json'; protocol = read(protocol_path)
    if sha(protocol_path) != protocol_path.with_suffix('.json.sha256').read_text().strip():
        raise ValueError('Previously frozen statistical protocol differs')
    if protocol['features'] != FEATURES or protocol['quantiles'] != [.005, .5, .995] or protocol['anomaly_score_boundary'] != 1.0:
        raise ValueError('No alternate threshold/feature search authorized')
    if (a.output / 'normal_references.json').exists():
        raise FileExistsError('No repeat fit or result overwrite')
    identities = {str(protocol_path): sha(protocol_path), str(ROOT / 'scripts/research/collect_m3_formal_results.py'): sha(ROOT / 'scripts/research/collect_m3_formal_results.py')}
    train = load_split(a, 'train', identities)
    normal = train[train.whole_known_normal].copy()
    if len(normal) != 74700:
        raise ValueError('Expected exactly original74700 whole-known-normal Train windows')
    references = {'unconditional': fit_reference(normal[FEATURES], normal.case_uid, protocol['quantiles'])}
    support, distributions = [], []
    for state in protocol['states']:
        part = normal[normal.observed_state.eq(state)]
        n = dict(normal_windows=len(part), cases=int(part.case_uid.nunique()), association_groups=int(part.association_group.nunique()))
        domain_cases = {d: int(part.loc[part.domain.eq(d), 'case_uid'].nunique()) for d in ['SIL', 'HIL']}
        minimum = protocol['Train_state_reference_minimum']
        available = all(n[k] >= minimum[k] for k in n) and all(v >= minimum['cases_per_domain'] for v in domain_cases.values())
        why = 'PASS_fixed_minimum' if available else 'Insufficient_Train_normal_coverage_no_fallback'
        if available:
            try:
                references[state] = fit_reference(part[FEATURES], part.case_uid, protocol['quantiles'])
            except ValueError as exc:
                available = False; why = str(exc)
        support.append(dict(state=state, **n, SIL_normal_cases=domain_cases['SIL'], HIL_normal_cases=domain_cases['HIL'],
                            reference_available=available, reason=why))
    for state, reference in references.items():
        part = normal if state == 'unconditional' else normal[normal.observed_state.eq(state)]
        weights = case_weights(part.case_uid); weight = weights.sum()
        for j, name in enumerate(FEATURES):
            x = part[name].to_numpy()
            mean = float(np.sum(weights * x) / weight)
            distributions.append(dict(reference=state, feature=name, windows=len(part), cases=part.case_uid.nunique(),
                                      weighted_mean=mean, weighted_variance=float(np.sum(weights * (x - mean) ** 2) / weight),
                                      lower=float(reference[j, 0]), median=float(reference[j, 1]), upper=float(reference[j, 2])))
    pd.DataFrame(support).to_csv(a.output / 'Train_state_support.csv', index=False)
    pd.DataFrame(distributions).to_csv(a.output / 'Train_normal_reference_distributions.csv', index=False)
    # References are saved/frozen before Val is even loaded.
    reference_path = a.output / 'normal_references.json'
    write(reference_path, dict(protocol_sha256=sha(protocol_path), fitted_from='Train74700_known_normal_only',
                               saved_before_Val_load=datetime.now(timezone.utc).isoformat(),
                               references={k: v.tolist() for k, v in references.items()}))
    reference_hash = sha(reference_path)
    print('FROZEN_TRAIN_ONLY_REFERENCES_BEFORE_VAL', reference_hash, flush=True)
    val = load_split(a, 'val', identities)
    scores, per_feature = score_reference(val[FEATURES], references['unconditional'])
    val['unconditional_score'] = scores; val['unconditional_abnormal'] = scores > 1.0
    val['conditional_score'] = np.nan; val['conditional_abnormal'] = pd.Series([pd.NA] * len(val), dtype='boolean')
    val['conditional_available'] = False
    for state, reference in references.items():
        if state == 'unconditional': continue
        mask = val.observed_state.eq(state)
        if not mask.any(): continue
        score, _ = score_reference(val.loc[mask, FEATURES], reference)
        val.loc[mask, 'conditional_score'] = score
        val.loc[mask, 'conditional_abnormal'] = score > 1.0
        val.loc[mask, 'conditional_available'] = True
    val['flight_directory'] = val.case_uid.str.split('/').str[1]
    val['fault_directory'] = val.case_uid.str.split('/').str[2]
    comparable = val[val.conditional_available].copy()
    comparable['conditional_abnormal'] = comparable.conditional_abnormal.astype(bool)
    tables = report_tables(comparable, protocol)
    tables.to_csv(a.output / 'paired_Val_metrics.csv', index=False)
    val.to_csv(a.output / 'Val_window_anomaly_ledger.csv', index=False)
    coverage = []
    for (state, domain, label), part in val.groupby(['observed_state', 'domain', 'qualified_y']):
        coverage.append(dict(state=state, domain=domain, label=int(label), windows=len(part), cases=part.case_uid.nunique(),
                             association_groups=part.association_group.nunique(), reference_available=bool(part.conditional_available.all())))
    pd.DataFrame(coverage).to_csv(a.output / 'Val_state_coverage.csv', index=False)
    case_rows = []
    for (uid, label), part in comparable.groupby(['case_uid', 'qualified_y']):
        case_rows.append(dict(case_uid=uid, label=int(label), domain=part.domain.iloc[0],
                              association_group=part.association_group.iloc[0], windows=len(part),
                              unconditional_rate=float(part.unconditional_abnormal.mean()),
                              conditional_rate=float(part.conditional_abnormal.mean())))
    pd.DataFrame(case_rows).to_csv(a.output / 'Val_case_rates.csv', index=False)
    # Per-feature abnormal flags, explanatory only; not alternative selected rules.
    feature_rows = []
    for state, part in comparable.groupby('observed_state'):
        _, u = score_reference(part[FEATURES], references['unconditional'])
        _, c = score_reference(part[FEATURES], references[state])
        for label in [0, 1]:
            mask = part.qualified_y.eq(label).to_numpy()
            for j, name in enumerate(FEATURES):
                feature_rows.append(dict(state=state, label=label, feature=name, windows=int(mask.sum()),
                                         unconditional_abnormal=int((u[mask, j] > 1).sum()), conditional_abnormal=int((c[mask, j] > 1).sum())))
    pd.DataFrame(feature_rows).to_csv(a.output / 'per_feature_flag_counts.csv', index=False)
    normal_mask, fault_mask = val.qualified_y.eq(0), val.qualified_y.eq(1)
    all_val = dict(windows=len(val), normal_windows=int(normal_mask.sum()), fault_windows=int(fault_mask.sum()),
                   unconditional_FP=int(val.loc[normal_mask, 'unconditional_abnormal'].sum()),
                   unconditional_TP=int(val.loc[fault_mask, 'unconditional_abnormal'].sum()))
    write(a.output / 'unconditional_all_Val_supplement.json', all_val)
    if sha(reference_path) != reference_hash:
        raise ValueError('Reference mutated after Val load')
    overall = tables[tables.scope.eq('overall')].iloc[0]
    hard_rows = tables[tables.scope.isin(['overall', 'domain', 'fault_vs_same_normal_pool'])]
    preserve = bool((hard_rows.conditional_TP >= hard_rows.unconditional_TP).all())
    summary = dict(status='COMPLETE_FROZEN_CONDITIONED_N_STATISTICAL_VALIDATION',
                   protocol_sha256=sha(protocol_path), Train_reference_sha256=reference_hash, reference_unchanged_after_Val=True,
                   Train_known_normal_windows=74700, Val_original_windows=len(val), comparable_Val_windows=len(comparable),
                   unavailable_Val_windows=int((~val.conditional_available).sum()),
                   supported_states=[k for k in references if k != 'unconditional'],
                   input_sha256=identities, overall={k: v.item() if hasattr(v, 'item') else v for k, v in overall.items() if not pd.isna(v)},
                   preserves_unconditional_TP_overall_six_faults_and_two_domains=preserve,
                   reduces_normal_FP=bool(overall.conditional_FP < overall.unconditional_FP),
                   no_threshold_state_search_training_model_forward_Test_Real_server_GPU=True,
                   hypotheses_only_no_C0_baseline_change=True)
    write(a.output / 'completion.json', summary)
    plot_results(a.output)
    print(json.dumps({k: summary[k] for k in ['status', 'comparable_Val_windows', 'unavailable_Val_windows',
                      'reduces_normal_FP', 'preserves_unconditional_TP_overall_six_faults_and_two_domains', 'overall']}, ensure_ascii=False, indent=2))


if __name__ == '__main__': main()
