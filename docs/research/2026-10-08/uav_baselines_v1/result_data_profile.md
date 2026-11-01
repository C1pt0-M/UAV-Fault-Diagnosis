# Data profile: D:\UAV-Fault-Diagnosis\docs\research\2026-10-08\uav_baselines_v1\case_metrics.csv

**Shape:** 3324 rows × 24 cols

## Columns

| Column | Type | n | missing | summary |
|---|---|---|---|---|
| `fault_events` | boolean | 3324 | 0 | 2 levels: 1(2892), 0(432); min_group_n=432 |
| `detected_events` | boolean | 3324 | 0 | 2 levels: 0(1765), 1(1559); min_group_n=1559 |
| `fault_case` | boolean | 3324 | 0 | 2 levels: 1(2892), 0(432); min_group_n=432 |
| `detected_case` | boolean | 3324 | 0 | 2 levels: 0(1765), 1(1559); min_group_n=1559 |
| `false_alarm_events` | ordinal | 3324 | 0 | 6 levels: 0(1458), 1(1100), 2(572), 3(151), 4(38) +1 more; min_group_n=5 |
| `prefault_alarm_case` | boolean | 3324 | 0 | 2 levels: 1(1681), 0(1643); min_group_n=1643 |
| `normal_decisions` | continuous | 3324 | 0 | mean=359, sd=161, range=[19, 666], skew=0.04 (approximately symmetric) |
| `normal_exposure_s` | continuous | 3324 | 0 | mean=35.9, sd=16.1, range=[1.9, 66.6], skew=0.04 (approximately symmetric) |
| `normal_alarm_decisions` | continuous | 3324 | 0 | mean=2.1, sd=3.65, range=[0, 39], skew=4.77 (highly skewed); outliers=140 (IQR) |
| `point_tp` | continuous | 3324 | 0 | mean=119, sd=169, range=[0, 597], skew=1.11 (highly skewed) |
| `point_fp` | continuous | 3324 | 0 | mean=2.1, sd=3.65, range=[0, 39], skew=4.77 (highly skewed); outliers=140 (IQR) |
| `point_fn` | continuous | 3324 | 0 | mean=252, sd=189, range=[0, 597], skew=-0.03 (approximately symmetric) |
| `point_tn` | continuous | 3324 | 0 | mean=357, sd=161, range=[11, 665], skew=0.03 (approximately symmetric) |
| `detected_delays_s` | text | 3324 | 0 |  |
| `case_uid` | text | 3324 | 0 |  |
| `domain` | categorical | 3324 | 0 | 2 levels: HIL(1665), SIL(1659); min_group_n=1659 |
| `flight_status` | categorical | 3324 | 0 | 6 levels: waypoint(774), circling(774), hover(768), velocity(396), acce(306) +1 more; min_group_n=306 |
| `level3_label` | categorical | 3324 | 0 | 7 levels: motor(1653), normal(432), magnetometer(252), GPS(252), accelerometer(249) +2 more; min_group_n=240 |
| `split` | categorical | 3324 | 0 | 2 levels: val(1665), test(1659); min_group_n=1659 |
| `simulation_pair_key` | continuous | 3324 | 0 | mean=2.04e+08, sd=1.62e+08, range=[1, 5.1e+08], skew=0.34 (approximately symmetric); -> log axis |
| `duration_s` | continuous | 3324 | 0 | mean=76, sd=25.4, range=[26.1, 115], skew=-0.02 (approximately symmetric) |
| `model` | categorical | 3324 | 0 | 3 levels: pca(1108), persistence(1108), gru(1108); min_group_n=1108 |
| `score_file` | text | 3324 | 0 |  |
| `top_fault_residual_features` | text | 3324 | 0 |  |

## Group structure
- Grouped by: `split`, `model`
- Number of groups: 6
- Group size: min=553, median=554, max=555

## Correlations (Pearson, sorted by |r|)
- `normal_decisions` ↔ `normal_exposure_s` : r = 1.000 (very strong)
- `normal_alarm_decisions` ↔ `point_fp` : r = 1.000 (very strong)
- `normal_decisions` ↔ `point_tn` : r = 1.000 (very strong)
- `normal_exposure_s` ↔ `point_tn` : r = 1.000 (very strong)
- `normal_decisions` ↔ `duration_s` : r = 0.713 (very strong)
- `normal_exposure_s` ↔ `duration_s` : r = 0.713 (very strong)
- `point_tn` ↔ `duration_s` : r = 0.712 (very strong)
- `point_tp` ↔ `point_fn` : r = -0.502 (strong)
- `point_tp` ↔ `duration_s` : r = 0.434 (moderate)
- `point_fn` ↔ `duration_s` : r = 0.349 (moderate)
- ... +26 more pairs

## Warnings
- 列 'false_alarm_events' 至少有一个类别 n<10 — 小样本必须展示原始数据点，不要只画均值柱状图。

## Chart suggestions (preliminary)
- 分类 vs 连续，样本量充足 → 箱线图 / 小提琴图，或带误差棒的柱状图（误差棒说明 SD/SEM/CI）
- ≥3 个连续变量 → 相关性热力图（['normal_decisions', 'normal_exposure_s', 'normal_alarm_decisions', 'point_tp', 'point_fp']）或 pairplot 散点矩阵
- 分类维度组合数 = 96768（fault_events, detected_events, fault_case, detected_case, false_alarm_events, prefault_alarm_case, domain, flight_status, level3_label, split, model 全交叉），**一张图塞不下**——建议按某一维拆成多面板，或选择子集。
- normal_alarm_decisions 高度偏态（skew=4.77）→ 考虑对数变换或小提琴图代替均值柱图
- point_tp 高度偏态（skew=1.11）→ 考虑对数变换或小提琴图代替均值柱图
- point_fp 高度偏态（skew=4.77）→ 考虑对数变换或小提琴图代替均值柱图
- simulation_pair_key 跨数个量级（1 ~ 5.1e+08）→ 用对数 y 轴

> 这是基于数据形态的**初步建议**。最终图型选择必须结合**论证目标**（你想说什么）—— 详见 `references/chart_selection.md`。