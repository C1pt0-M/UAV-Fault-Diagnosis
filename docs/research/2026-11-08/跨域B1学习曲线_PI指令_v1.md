# PI 指令：跨域 B1 学习曲线 v1（Real 开发池）

> 冻结协议：`configs/experiments/real_v2_b1_learning_curve.json`。**以协议为准**；本文与协议冲突时按协议执行，并在偏差清单中注明。
> 划分与 B0 口径不变：`configs/data/real_v2_split.csv`（sha256 `9232ee35…90ee`）、`configs/experiments/real_v2_b0_diagnostics.json`。

## 0. 背景

B0（commit 9044e26）的结论：

- Real 内部有很强的会话差异：完整正常飞行与故障前段的 AUC 为 0.972；
- 只用 Real 训练的 GBT，在同一批故障案例内部区分故障与故障前段，AUC 为 0.975（motor 0.984，sensor 0.965）；冻结源 GBT 只有 0.757；
- 架构方另做了核对：逐案例配对 AUC 中位数为 1.00（motor）和 0.99（sensor）；故障后正常段、正常飞行后段没有被系统性误报，所以区分能力不只是“飞行时间”带来的。

用户已确认 B1 的数值：

- H2：完整正常飞行、故障前段两类正常的误报都 ≤ 5.5 次/h，motor、sensor 新确认检出都 ≥ 80%；
- S+R 用固定权重，两个域总权重相等；
- 事件 onset 在内层 CV 中选择。

## 1. 禁止事项

- 不打开封存部分，做法同 B0：用断言保证，并在回执中给出实际加载的文件数。
- 看到任何结果后，不改比例档、重复次数、权重、onset 网格、策略参数、选择规则或判据。
- 不在外层折上选 onset；外层折只用于评价。
- 不读内部 test，不跑 M0/MSTCA，不用 GPU，不训练 T1，不启动 B2。
- 不改标签，看到结果后不排除案例。
- 不修改 `scripts/research/case_relative_features_dev.py`、`tmp_control_compensation_audit*.log`；不重生成 golden；不改结项报告和台账。

## 2. 输入

| 项目 | 位置 / 校验 |
|---|---|
| 划分 | `configs/data/real_v2_split.csv`，sha256 一致才继续 |
| Real 窗口 | B0 同口径（台账 + `windows/*.npz`，按 `original_cache_index` 对齐） |
| Real 连续数据 | `cases/*.npz`（仅开发池），用 `replay_ens_events_val.case_windows` 逐帧切窗 |
| 源 train | `D:/UAV-Data/causal_input_v1/cache/train.pt` + `ledgers/train_included_windows.csv`，权重与 `run_test.py` 相同 |
| 冻结源 GBT（S 臂） | `gbt_full_train.pkl`，sha256 `df468524…6774` |
| 事件定义模板 | `scripts/research/ens_event_threshold.py` 的 `case_events` / `summarize`；N1、N2 按协议拆分 |
| B0 代码 | `scripts/research/real_v2_b0_diagnostics.py`（特征、对齐、bootstrap 直接复用） |

在本地运行（Real 只在本地）。解释器参数用 `-X utf8 -E -P -B`。CPU 线程数沿用 B0；不改 CPU dispatch。

## 3. 步骤

1. **启动检查**：
   - 划分和 S 模型的 sha256 一致；
   - 开发池去掉 quarantine 后的案例数与 B0 一致（315）；
   - 输出目录不能已存在。
2. **复现检查（先做）**：用 fraction 1.0 的 R 臂、外层 5 折重新得到 OOF 窗口概率，与 B0 `oof_predictions.csv` 的 `p_R` 比对。
   - 训练集和配方相同，预期一致；最大绝对差 > 1e-6 就停止并报告（容许多线程求和顺序带来的极小差异）。
   - 这一步同时验证特征和折的实现。
3. **学习曲线（窗口级）**：
   - 比例 0.10、0.25、0.50 各 5 个重复，1.00 一次；
   - 子集按协议的哈希规则逐层取组，同一重复内各比例嵌套，R 与 S+R 用同一个子集；
   - 每个“臂 × 比例 × 重复”给出合并外层 OOF 的 AP、故障对 pre_fault 的 AUC，以及召回 0.8 时各正常总体的 FPR；
   - **H1**：比例 0.25 时，ΔAP = AP(S+R) − AP(R) 在 5 个重复上取均值；对开发池关联组做聚类 bootstrap（2000 次，rng 20261108），每次重新计算重复均值 ΔAP；CI 下界 > 0 判为成立。其他比例只描述。
4. **事件级（比例 1.00，R 与 S+R）**：
   - **内层**：对每个外层折 k，在其余 4 折上做留一折 CV。每个内层模型用除 k、j 之外的 3 折训练，给 j 折打逐帧分；
   - **选 onset**：在内层 OOF 上扫 0.45–0.95 网格。可行条件是 motor、sensor 的检出都 ≥ 0.80；在可行点中取 (N1 + N2) 合并误报率最低的；并列时取检出多的，再并列取 onset 低的。没有可行点时，取 min(motor, sensor) 检出最大的 onset，并标记 `no_feasible_onset`；
   - **外层**：k 折案例用 4 折训练的外层模型打分（即第 3 步 1.00 档的模型），在选定 onset 下计算事件，最后汇总 5 折；
   - **H2**：对某一臂，N1 ≤ 5.5/h、N2 ≤ 5.5/h、motor 检出 ≥ 0.80、sensor 检出 ≥ 0.80 全部满足（按点值）即成立；至少一臂成立，则整条路线 H2 成立。四个量都给出关联组 bootstrap CI，仅作描述；
   - S 臂用 onset 0.95 跑同样的事件统计，仅作描述。
5. **描述性**：按协议 `descriptive` 列表逐项给出，包括逐案例配对 AUC，以及按飞行目录拆分的 N1/N2。

逐帧特征可以每个案例缓存一次，模型和逐帧分数不入库，sha256 写进回执。

## 4. 产物

- `artifacts/real_v2_b1_learning_curve/`：receipt.json、results.json、window_metrics.csv、onset_selection.csv、events_per_case.csv、run.log、execution_notes.md；
- 报告：`docs/research/2026-11-08/跨域B1学习曲线结果_v1.md`；
- 脚本：`scripts/research/real_v2_b1_learning_curve.py`，单测只用合成数据；
- docs/、artifacts/ 下的新文件用 `git add -f`；提交后 **push**。

## 5. 回复格式（必须）

1. 启动检查、复现检查（最大差）、实际加载的文件数（只含开发池）；
2. 学习曲线表：臂 × 比例的 AP、AUC(b)（重复均值 ± 范围）；H1 的 ΔAP [CI] 与判定；
3. 事件级：每折选定的 onset 及其可行性；R、S+R 的 N1、N2、motor/sensor 检出、时延（含 CI）；H2 逐臂判定；S@0.95 参照；
4. 描述性要点，特别是会话差异核查（逐案例配对 AUC、N1 与 N2 的差别）；
5. 按协议 `consequences` 给出的结论（不自动启动任何后续步骤）；
6. **完整偏差清单**：凡与协议或本文不一致的地方（实现选择、失败与重试、无法完成的项、额外分析），逐条列出；没有就写“无”；
7. 新增文件列表与 commit hash（确认已 push）。
