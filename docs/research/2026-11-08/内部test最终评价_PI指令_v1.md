# PI 指令：内部 test 最终评价 v1

> 冻结协议：`configs/experiments/internal_test_final_v1.json`（用户 2026-11-08 明确同意后冻结）。**以协议为准**；本文与协议冲突时按协议执行，并在偏差清单中注明。
> 这是内部 test 的**第二次、也是最后一次**使用。只运行一次；任何失败都如实报告，不重跑指标。

## 0. 背景

源域最终检测模型已定：GBT（`gbt_full_train.pkl`）。窗口阈值 0.3744，事件 onset 0.95，都是在 train 上定的。本轮在内部 SIL/HIL test 上评一次冻结配置，对照部署基线 M0。零训练；窗口级复用已有预测，事件级需要用冻结模型对 test 做逐帧打分。

## 1. 禁止事项

- 只评 4 个工作点：GBT 窗口 0.3744、M0 窗口 argmax、GBT 事件 onset 0.95、M0 事件 onset 0.60。**不在 test 上扫任何 onset 或阈值**，也不计算其他工作点的结果（包括“顺便看看”）。
- 不训练、不重拟合 GBT，不改事件策略参数。
- 不读 Real；不读 train/val 以外的其他留出数据。
- 看到结果后不排除案例、不改判据、不重跑。
- 不修改 `scripts/research/case_relative_features_dev.py`、`tmp_control_compensation_audit*.log`；不重生成 golden；不改结项报告和台账。

## 2. 输入

| 项目 | 位置 / 校验 |
|---|---|
| GBT | `artifacts/ens_internal_test_v1_20261105/gbt_full_train.pkl`，sha256 `df468524…6774` |
| M0 | `artifacts/causal_pure_mstca_m0_v1_20261101/results/runs/balanced_source_normal/best.pt`，sha256 `ac8596cd…c6cb` |
| 已有 test 窗口预测 | `artifacts/ens_internal_test_v1_20261105/test_predictions.npz`，sha256 `45360dbf…e8f8`；键 `m0_probabilities, gbt_prob, labels, original_indices` |
| test 案例与窗口台账 | `D:/UAV-Data/causal_input_v1/tasks.json`（split = test，553 例）、`ledgers/test_included_windows.csv` |
| 逐帧打分模板 | `scripts/research/score_ens_frames.py`（val 分支：M0 B best.pt + 全量 GBT），窗口切分用 `replay_ens_events_val.case_windows` |
| 事件聚合模板 | `scripts/research/gbt_event_operating_point_v1.py`（定义、汇总、bootstrap 直接复用） |
| GPS 故障模式 | `artifacts/q2_gps_fault_modes.csv`（已覆盖 48 个 test GPS 案例） |
| 旧锁 | `artifacts/ens_internal_test_v1_20261105/TEST_EVALUATED.lock` 只读，不得删除或改写 |

## 3. 步骤

1. **启动检查**：
   - 两个模型和 `test_predictions.npz` 的 sha256 都要与上表一致；
   - `artifacts/internal_test_final_v1/` 不能已存在；
   - test 案例必须是 553 个，GPS 案例都能查到 mode，排除集（hover、GPS、mode 4）数出来；
   - 任一项不符就停止。
2. **逐帧打分**：对 553 个 test 案例，用两个冻结模型打分（与 `score_ens_frames.py` 的 val 分支相同，只把 split 换成 test）。输出放 `D:/UAV-Data/internal_test_final_v1_scores/`，不入库，sha 记入回执。
3. **一致性门（看任何指标之前）**：
   - 把逐帧分数在 test 窗口末端处的值和 `test_predictions.npz` 对齐（通过 `test_included_windows.csv` 的 case_uid + 窗口末行 / original index）；
   - GBT 最大绝对差 ≤ 1e-6，M0 ≤ 0.002；
   - 对不上、或有窗口无法对齐，立即停止并报告，不计算任何指标。
4. **写锁**：门通过后，先写 `artifacts/internal_test_final_v1/TEST_EVALUATED.lock`（含配置 sha、模型 sha、时间），再计算指标。脚本见到锁就拒绝运行。
5. **事件级（主判据）**：策略固定 `clear 0.4, k_on 3, k_off 3, min_event 0.5 s, merge 0.5 s, inference_every 1`；GBT onset 0.95，M0 onset 0.60。
   - **F1**：独立正常误报事件/有效曝光小时，GBT < M0；
   - **F2**：去掉排除集后的检出案例数，GBT ≥ M0 − 1% × 已评分故障案例数（分母取完整已评分故障案例数，并同时披露去排除后的分母）；单类缺口 > 2 例只标记；
   - 两项都满足为 **CONFIRMED**，否则为 **NOT_CONFIRMED**。
6. **窗口级（F3，次要）**：用 `test_predictions.npz`，GBT 阈值 `≥ 0.3744`，M0 用 argmax。6 类召回（GPS 去掉后段 hover mode 4 窗口，即 hover、窗口末行 × 0.02 > 30 s、mode 4），GBT ≥ M0 − 1 pp，逐类给出通过 / 失败。
7. **时延（F4，次要）**：每类给出确认时延中位数和 P90，GBT 对 M0；中位数慢超过 2 s 的类别标记。另给全部共同检出案例的配对差中位数和 P90。
8. **描述性**：
   - 窗口级正常 FP：GBT@0.3744 对 M0，并列出已知的 ENS 342 作参照；
   - association_group 聚类 bootstrap（2000 次，rng 20261106），给误报事件/小时比值和窗口 FP 比值的 CI；
   - SIL / HIL 分开；按飞行目录统计；hover 报警占用率；
   - 故障前误报事件；各类 spans_onset；
   - 排除集：检出按 30 s 前后分开，后段 mode 4 GPS 的窗口召回；
   - 案例级：出现误报事件的独立正常案例数，检出的故障案例数。

## 4. 产物

- `artifacts/internal_test_final_v1/`：`TEST_EVALUATED.lock`、`receipt.json`（含逐帧分数 sha）、`parity_gate.json`、`results.json`、`events_per_case.csv`、`run.log`、`execution_notes.md`
- 报告 `docs/research/2026-11-08/内部test最终评价结果_v1.md`
- 脚本 `scripts/research/internal_test_final_v1.py`（如需单测，测试只能用合成数据，不能读 test）
- docs/、artifacts/ 下的新文件用 `git add -f`；提交后 **push**

## 5. 回复格式（必须）

1. 启动检查与一致性门的数值（模型 sha、对齐窗口数、GBT 和 M0 的最大差）；
2. F1、F2 的数值与 CONFIRMED / NOT_CONFIRMED；
3. F3 逐类表、F4 逐类时延表；
4. 描述性结果要点；
5. **完整偏差清单**：凡与协议或本文不一致的地方（实现选择、失败与重试、无法完成的项、额外分析），逐条列出；没有就写“无”；
6. 新增文件列表与 commit hash（确认已 push）。
