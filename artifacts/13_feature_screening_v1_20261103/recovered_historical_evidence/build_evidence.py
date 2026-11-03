"""Summarize existing13-candidate evidence; no training or raw data reads.

Run: python artifacts/candidate_feature_evidence_v1_20261101/build_evidence.py
Inputs are saved source-development JSON and existing implementation source only.
"""
from pathlib import Path
import csv
import hashlib
import json
import statistics

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / 'results'
BASE = ROOT / 'artifacts/kinematic_residual_discrimination_v1_20261101'
INPUTS = {}


def load(relative):
    path = ROOT / relative
    raw = path.read_bytes()
    INPUTS[relative] = hashlib.sha256(raw).hexdigest()
    return json.loads(raw.decode('utf-8'))


def save(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + chr(10), encoding='utf-8')


def csv_rows(name, rows):
    with (OUT / name).open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aggregate(values):
    available = [v for v in values if v is not None]
    return {'case_count': len(values), 'available_cases': len(available),
        'median': statistics.median(available) if available else None,
        'min': min(available) if available else None, 'max': max(available) if available else None}


# IDs retain the original9 formula order and the four signed residual components.
CANDIDATES = [
    ('O1', 'old_output_std', '输出槽标准差代理', 'std(z_out_1..4), population std', 'z空间统计量',
     '数学定义明确；不是物理转速或推力不平衡，各输出槽分别标准化后再求跨槽离散。',
     '量化对PWM量级／工况的条件依赖、与O2冗余及相对M0收益。'),
    ('O2', 'old_output_range', '输出槽极差代理', 'max(z_out)-min(z_out)', 'z空间统计量',
     '数学定义明确；对单槽极值敏感，不是独立物理约束。',
     '未量化极端槽异常和O1冗余；区分真实故障信息与背景差异。'),
    ('O3', 'old_ctrl_output_gap_mean', '控制—输出差均值代理', 'mean_i(abs(z_ctrl_i-z_out_i))', 'z空间统计量',
     '控制槽与电机输出非同语义、非一一映射；只能作统计代理，不是跟踪残差。',
     '检查配对索引和标准化所致背景编码；独立增益未测。'),
    ('O4', 'old_ctrl_output_gap_std', '控制—输出差标准差代理', 'std_i(abs(z_ctrl_i-z_out_i)), population std', 'z空间统计量',
     '数学定义明确，物理跟踪解释不成立；正常阶段变化不证明导致误报。',
     '解释已见大IQR位移并区分小参考尺度效应；因果责任与增量收益均未测。'),
    ('O5', 'old_effort', 'effort代理', 'mean(concat(z_ctrl,z_out))/max(z_acc_norm,0.1)', 'z空间比值代理',
     '分母截断不是物理标定；不能称推力／能量。当前26通道为该重建版本。',
     '截断命中率、尾部／尖峰及条件敏感性尚未系统量化；不靠改截断追分。'),
    ('O6', 'old_acc_deviation', '加速度模标准化偏离代理', 'abs(z_acc_norm)', 'z空间统计量',
     '不是abs(norm(a)-g)的物理残差；基础26通道已有acc_norm。',
     '区分正常机动与故障偏离，检查对既有通道的非线性表征增益。'),
    ('O7', 'old_gyro_norm', '角速度活动代理', 'norm(z_gyro_xyz)', 'z空间统计量',
     '不是rad/s物理模；正常旋转／机动也可变大。',
     '运动条件混杂、不同故障模式稳定性及相对M0增益未测。'),
    ('O8', 'old_vertical_consistency', '垂向差分代理', 'abs(diff(z_baro,prepend=first)-z_vel_z)', '混合z空间代理',
     '未除dt且符号／尺度不统一，不能称严谨垂向一致性；差分受ZOH和异步影响。',
     '分离异步／采样差分与真实信息；若修正公式必须新版本，不能悄改旧候选。'),
    ('O9', 'old_attitude_energy', '四元数分量统计代理', 'norm(z_q_xyz)', 'z空间统计量',
     '标准化后不是合法旋转的能量；q/-q、绝对姿态和参考航向需谨慎。',
     '量化符号／参考系依赖和正常机动响应，不能把正常漂移小当增益证据。'),
    ('N1', 'rh_signed_m', '有符号垂向积分残差', 'h_av(t)-h_av(t-0.2)+integral(vz_av(s),ds)', 'm',
     'NED速度加号及合成闭式解已核验；是因果发布流一致性，不是同步物理真值。',
     '跨实际运行状态、故障强度／模式与边界短暂响应；共享估计、异步及覆盖影响；训练统计未拟合。'),
    ('N2', 'rR_x_rad', '姿态积分残差x', 'Log(R_hat.T R_end).vee_x; R_hat=R_start product(exp(omega_body*dt))', 'rad',
     '旋转约定／非交换积分／q双覆盖已核验，不能按固定物理轴故障直接解释分量。',
     '跨工况和故障模式方向稳定性；时序／reset／正常峰值及分量组合；训练统计未拟合。'),
    ('N3', 'rR_y_rad', '姿态积分残差y', 'Log(R_hat.T R_end).vee_y; same fixed0.2s integration', 'rad',
     '与N2相同的姿态组计算，不把共享验证当成三次独立验证。',
     '跨工况和故障模式方向稳定性；时序／reset／正常峰值及分量组合；训练统计未拟合。'),
    ('N4', 'rR_z_rad', '姿态积分残差z', 'Log(R_hat.T R_end).vee_z; same fixed0.2s integration', 'rad',
     '与N2相同的姿态组计算，正反方向信息不能用范数替代。',
     'train／val磁力计案例方向差异来源未分离；不能事后翻向择优；增量收益未测。'),
]


def main():
    prefix = BASE.relative_to(ROOT).as_posix()
    summary = load(prefix + '/results/summary.json')
    stages = load(prefix + '/results/normal_case_stage_stability.json')
    responses = load(prefix + '/results/matched_fault_responses.json')
    reference = load(prefix + '/results/diagnostic_reference_train_normal.json')
    verification = load(prefix + '/independent_verification.json')
    feasibility = load('artifacts/kinematic_residual_validation_v1_20261101/completion.json')
    assert summary['status'] == 'COMPLETE_FIXED36_SOURCE_HOVER_FEATURE_RESPONSE_DIAGNOSIS'
    assert verification['status'] == 'PASS_INDEPENDENT_FIXED36_SOURCE_RESPONSE_CHECKS'
    assert summary['case_count'] == 36 and summary['common_eligible_windows'] == 2072
    assert not summary['residual_training_normalization_fitted']
    assert len(CANDIDATES) == 13 and len({c[1] for c in CANDIDATES}) == 13
    assert len(stages) == 180 and len(responses) == 360
    subtypes = ['motor', 'accelerometer', 'gyroscope', 'barometer', 'magnetometer', 'GPS']
    normal_rows, auc_rows, cards = [], [], []
    for identity, feature, name, formula, unit, interpretation, gap in CANDIDATES:
        card = {'id': identity, 'feature_key': feature, 'name': name, 'formula': formula, 'unit': unit,
            'formula_scope': interpretation, 'specific_gaps': gap,
            'incremental_value_vs_M0': 'NOT_EVALUATED', 'selection_status': 'CANDIDATE_NOT_SELECTED_OR_REJECTED',
            'normal_stage': {}, 'fault_response': {}}
        assert reference[feature]['windows'] == 340
        card['diagnostic_reference'] = reference[feature]
        for split in ['train', 'val']:
            rows = [r for r in stages if r['split'] == split and r['feature'] == feature]
            assert len(rows) == 6
            value = aggregate([r['shift_in_train_normal_iqr'] for r in rows])
            expected = next(r for r in summary['normal_stability_case_summary'] if r['split'] == split and r['feature'] == feature)
            assert value['available_cases'] == expected['cases_with_value']
            assert value['median'] == expected['case_value_median']
            normal_rows.append({'id': identity, 'feature': feature, 'split': split, **value,
                'quantity': 'case-paired late-minus-early window-feature medians / selected train-normal IQR'})
            card['normal_stage'][split] = value
            for subtype in subtypes:
                rows = [r for r in responses if r['split'] == split and r['subtype'] == subtype and r['feature'] == feature]
                assert len(rows) == 2 and {r['domain'] for r in rows} == {'SIL', 'HIL'}
                value = aggregate([r['matched_crop_quarter_auc'] for r in rows])
                expected = next(r for r in summary['matched_fault_auc_case_summary'] if r['split'] == split and r['subtype'] == subtype and r['feature'] == feature)
                assert value['median'] == expected['case_value_median'] and value['min'] == expected['case_value_min'] and value['max'] == expected['case_value_max']
                by_domain = {r['domain']: r for r in rows}
                result = {'id': identity, 'feature': feature, 'split': split, 'fault_directory': subtype,
                    'SIL_auc': by_domain['SIL']['matched_crop_quarter_auc'], 'HIL_auc': by_domain['HIL']['matched_crop_quarter_auc'],
                    'SIL_case_uid': by_domain['SIL']['case_uid'], 'HIL_case_uid': by_domain['HIL']['case_uid'],
                    'SIL_active_windows': by_domain['SIL']['matched_active_windows'], 'HIL_active_windows': by_domain['HIL']['matched_active_windows'],
                    'SIL_unmatched_active_windows': by_domain['SIL']['unmatched_active_windows'],
                    'HIL_unmatched_active_windows': by_domain['HIL']['unmatched_active_windows']}
                assert all(v is None or 0 <= v <= 1 for v in [result['SIL_auc'], result['HIL_auc']])
                auc_rows.append(result)
                card['fault_response'].setdefault(split, {})[subtype] = result
        cards.append(card)
    assert len(normal_rows) == 26 and len(auc_rows) == 156
    assert sum(c['incremental_value_vs_M0'] == 'NOT_EVALUATED' for c in cards) == 13
    # Minimal runnable checks: no mixing signed components with two magnitude summaries.
    assert 'rh_abs_m' not in {c['feature_key'] for c in cards} and 'rR_norm_rad' not in {c['feature_key'] for c in cards}
    assert aggregate([None, -2., 3.]) == {'case_count': 3, 'available_cases': 2, 'median': .5, 'min': -2., 'max': 3.}
    assert [(r['available_cases']) for r in normal_rows if r['split'] == 'train'] == [5] * 13
    assert [(r['available_cases']) for r in normal_rows if r['split'] == 'val'] == [6] * 13
    OUT.mkdir(exist_ok=False)
    save('feature_cards.json', cards)
    csv_rows('normal_stage_shifts.csv', normal_rows)
    csv_rows('matched_fault_auc.csv', auc_rows)
    fmt = lambda value: '缺失' if value is None else f'{value:.3f}'
    lines = ['# 13维已有数值证据（由已保存结果重汇总）', '',
        '不是新实验、筛选排名或最终测试。正向AUC为数值越大越故障，未翻符号。', '',
        '## 正常前后段位移', '',
        '以每窗140位置特征中位数为输入，先逐案例配对，再跨案例取中位数；单位为选定train正常IQR。',
        'train每项5/6案例有配对、val6/6；SIL/HIL关联且窗口重叠，不作独立样本显著性。', '',
        '|ID|特征|train中位数 [min,max]|val中位数 [min,max]|', '|---|---|---|---|']
    for card in cards:
        a, b = card['normal_stage']['train'], card['normal_stage']['val']
        lines.append('|'+card['id']+'|'+card['feature_key']+'|'+fmt(a['median'])+' ['+fmt(a['min'])+','+fmt(a['max'])+']|'+fmt(b['median'])+' ['+fmt(b['min'])+','+fmt(b['max'])+']|')
    for split in ['train', 'val']:
        lines += ['', '## '+split+'故障匹配AUC', '', '每格SIL / HIL，各一个固定案例，不是置信区间。列为故障目录，不控制全部注入强度／模式。', '',
            '|ID|motor|accelerometer|gyroscope|barometer|magnetometer|GPS|', '|---|---|---|---|---|---|---|']
        for card in cards:
            values = [card['fault_response'][split][k] for k in subtypes]
            lines.append('|'+card['id']+'|'+'|'.join(fmt(v['SIL_auc'])+' / '+fmt(v['HIL_auc']) for v in values)+'|')
    lines += ['', '全部数值为旧脚本每窗特征中位数的匹配排序表现，不是完整时序特征模型性能。共同有效窗2072/原准入2258；不据此淘汰单项。',
        'rh_abs_m与rR_norm_rad只留原诊断来源，不列入13维。原完整输出仍保留15项。', '']
    (OUT / 'numeric_appendix.md').write_text(chr(10).join(lines), encoding='utf-8')
    # Record executable source identities without loading a network or a model.
    for relative in ['packages/uav_fault/uav_fault/models/cnn/lstm.py', 'scripts/research/check_kinematic_feature_response.py',
            'scripts/research/validate_kinematic_residuals.py', 'artifacts/candidate_feature_evidence_v1_20261101/build_evidence.py']:
        INPUTS[relative] = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
    for relative, expected in INPUTS.items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == expected, relative
    files = {p.name: {'bytes': p.stat().st_size, 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p in OUT.iterdir() if p.is_file()}
    save('completion.json', {'status': 'COMPLETE_SAVED_EVIDENCE_ONLY13_CANDIDATE_REVIEW',
        'candidate_count': 13, 'normal_summary_rows': 26, 'fault_summary_rows': 156,
        'input_sha256': INPUTS, 'output_identities': files,
        'aggregation_verified_against_existing_summaries': True, 'minimal_checks_passed': True,
        'training_model_inference_or_GPU': False, 'raw_logs_or_data_cache_read': False,
        'final_test_or_Real_read': False, 'new_normalization_fitted': False,
        'incremental_M0_value_unknown_for_all13': True,
        'coverage': {'original_cap64': 2304, 'original_admitted': 2258, 'common': 2072,
            'new_residual_unavailable_among_admitted': 186},
        'prior_feasibility_status': feasibility['status']})
    print('PASS:13 cards,26 paired summaries,156 subtype/domain AUC rows; source summaries match exactly.')
    print('No new experiment, candidate selection, source-data read, training, inference or GPU.')


if __name__ == '__main__':
    main()
