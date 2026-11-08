# PI 指令：GBT 事件级工作点 v1

> 冻结协议：`configs/experiments/gbt_event_operating_point_v1.json`（commit 3a9ac66，运行前冻结）。**以协议为准**；本文与协议冲突时按协议执行，并在偏差清单中注明。

## 0. 背景（一段话）

用户已决定：GBT 是源域最终检测模型，工作点按“逐类召回不低于基线”，后段 hover mode 4 GPS 不作硬约束。窗口级阈值已定为 0.3744（`源域最终模型与工作点_v1.md`）。但部署时报警由事件策略（连续回放 + `OnlineDetectionPolicy`）产生，而 GBT 单模型从未做过事件级评估。11-06 的经验是：窗口级更好不代表事件级更好（ENS 在默认 onset 0.6 下 val 误报事件 29 对 M0 的 12）。本轮只做事件级聚合，**零训练、零新推理**。

## 1. 禁止事项

- 不训练、不重新推理，只用已有逐帧分数。
- 不读内部 test、不读 Real。
- 不在 val 上选 onset；val 只用于确认。
- 看到结果后不改网格、策略参数、约束或选择规则。没有 onset 满足约束时，**停止并报告**，不放宽。
- 不修改 `scripts/research/case_relative_features_dev.py` 和 `tmp_control_compensation_audit*.log`，不重生成任何 golden。

## 2. 输入

| 项目 | 位置 |
|---|---|
| 逐帧分数 | `D:/UAV-Data/ens_event_threshold_v1_scores/`：3135 个 npz + `index.csv`（列 case_uid, split, fold, domain）；npz 键 `timestamp_us, m0, gbt, reference_state, row_label_valid` |
| 参照脚本 | `scripts/research/ens_event_threshold.py`（`case_events` / `summarize` / 配对时延，直接复用其事件定义） |
| 参照结果 | `artifacts/ens_event_threshold_v1_20261106/results.json`、`m0_sweep_supplement.json` |
| GPS 故障模式 | `artifacts/q2_gps_fault_modes.csv`（case_uid, mode, params） |

分数只在本地。本轮是 CPU 聚合，建议本地运行；如需在服务器跑，只能原样拷贝该目录，不得重新生成。

## 3. 步骤

1. **新脚本** `scripts/research/gbt_event_operating_point_v1.py`，以 `ens_event_threshold.py` 为模板：
   - arm 为 `GBT`（`prob = z['gbt']`）与 `M0`（`prob = z['m0']`），另保留 `ENS = (m0+gbt)/2` 仅用于复现门；
   - 策略固定为 `inference_every=1, clear_th=0.4, k_on=3, k_off=3, min_event_s=0.5, merge_gap_s=0.5`；
   - onset 网格：0.45, 0.50, …, 0.95（11 档），GBT 和 M0 都跑全网格；
   - 误报事件、detected、spans_onset、确认时延的定义与 `ens_event_threshold.py` 完全一致。
2. **复现门（先做，不过就停）**：train held-out 上 M0@0.60 = 113 个独立正常误报事件、1513 个 detected；ENS@0.75 = 81、1705。不一致即停止并报告差异。
3. **排除集**：hover 飞行、类别 GPS、`q2_gps_fault_modes.csv` 中 mode = 4 的故障案例。运行前核对：train 36 个、val 8 个；任何 GPS 案例查不到 mode 即停止。
4. **在 train held-out 上选 onset**（GBT 与 M0 各自套用同一规则；M0 的结果称为 M0-tuned）：
   - 约束 a：6 个类别（GPS 去掉排除集、加速度计、气压计、陀螺、磁力计、电机）的 detected 案例数，每类都 ≥ M0@0.60 同类数；
   - 约束 b：两者都检出的案例，配对确认时延中位数差（候选 − M0@0.60）≤ +0.5 s；
   - 目标：满足约束的 onset 中，独立正常误报事件/小时最低；并列时取 detected 总数多的，再并列取 onset 低的。
5. **判定**：
   - EV1（train）：GBT@选定 的独立正常误报事件/小时 < M0@0.60；
   - EV2（val）：GBT@选定 的独立正常误报事件数 ≤ M0@0.60；
   - EV3（val）：GBT@选定 的 detected 总数（去掉排除集）≥ M0@0.60 − 1% × 已评分故障案例数；单类缺口 > 2 个案例只标记，不判失败；
   - 三项全过为 PASS。
6. **次要输出**（全部描述性）：
   - GBT@选定 对 M0-tuned，train 与 val；
   - train held-out 上误报事件/小时比值（GBT@选定 / M0@0.60，及 / M0-tuned）的 association_group 聚类 bootstrap CI，2000 次，rng 20261106（association_group 从案例台账取，无法取得就停止报告，不得自造分组）；
   - 每类：detected、spans_onset、确认时延中位数与 P90；
   - 按飞行目录：独立正常误报事件、出现误报的案例数、报警占用率；
   - 故障案例中的故障前误报事件；
   - 排除集案例：按确认时刻在 t = 30 s 之前 / 之后分开报告 detected；
   - GBT、M0 全网格扫描表（train、val 两个总体）；
   - 仅报告：feature_groupcv_v1 的 M0 折分数与 T0 seed 42 held-out 概率在 cap64 窗口上是否一致（做不到就写“无法核对”及原因，不另行推理）。

## 4. 产物

- `artifacts/gbt_event_operating_point_v1/results.json`、`events_per_case.csv`、`sweep.csv`
- 运行日志
- 报告 `docs/research/2026-11-08/GBT事件级工作点结果_v1.md`：协议、复现门、选定 onset、EV1–EV3、次要结果、注意事项
- **不要**改结项报告；台账由我核对后更新。

## 5. 回复格式（必须）

1. 复现门数值（4 个数）；
2. GBT 与 M0-tuned 的选定 onset，以及每档是否满足约束 a/b（表）；
3. EV1–EV3 数值与 PASS/FAIL；
4. 次要结果要点；
5. **完整偏差清单**：凡与协议或本文不一致的地方（含实现选择、无法完成的项、额外做的分析），逐条列出；没有就写“无”；
6. 新增 / 修改的文件列表与 commit hash。
