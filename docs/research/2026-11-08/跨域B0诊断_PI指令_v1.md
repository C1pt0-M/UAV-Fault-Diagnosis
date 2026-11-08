# PI 指令：跨域 B0 诊断 v1（Real 开发池）

> 冻结协议：`configs/experiments/real_v2_b0_diagnostics.json`。**以协议为准**；本文与协议冲突时按协议执行，并在偏差清单中注明。
> 划分文件：`configs/data/real_v2_split.csv`，sha256 `9232ee35…90ee`（只用 tasks.json 元数据生成，生成脚本 `scripts/research/make_real_v2_split.py`）。

## 0. 背景

源域主线已收尾（GBT；内部 test CONFIRMED 并永久关闭）。跨域阶段的用户决定（2026-11-08）：

1. Real 按关联组重新划分：约 70% 开发池、约 30% 封存为最终测试；
2. 旧角色中“test_fault 不得用于监督训练”的约束由本路线的新协议取代；
3. 先做 CPU 的 GBT 路线（B0、B1），T1 MSTCA 以后再定。

本轮 B0 是零风险诊断，回答三件事：

- **B0-1**：Real 内部是否有会话捷径，即独立正常飞行和故障案例的故障前段，能否被轻易区分；
- **B0-2**：只用 Real 标签，130 维窗口统计能否区分故障与正常，而且是否在同一次飞行内部也能区分；
- **B0-3**：标签和曝光核查。

本轮不选工作点，不做事件回放，不保留任何模型用于部署。

## 1. 禁止事项

- **不打开封存部分**：不加载 partition = sealed 案例的 `cases/` 或 `windows/` npz；从台账读入后立即丢弃封存行，不对它们计算任何量。脚本要用断言保证这一点，并在回执中列出实际加载的文件数。
- 不读内部 test；不改源 train/val 任何东西。
- 不搜索超参数或样本权重，不改特征，不在看到结果后改标签或排除案例。
- 不做事件回放，不选工作点，不跑 M0/MSTCA，不用 GPU，不训练 T1，不启动 B1。
- 不使用 `CausalCacheDataset` 读取 Real（它按旧角色约束设计），直接读 `windows/*.npz`。
- 不修改 `scripts/research/case_relative_features_dev.py`、`tmp_control_compensation_audit*.log`；不重生成 golden；不改结项报告和台账。

## 2. 输入

| 项目 | 位置 / 校验 |
|---|---|
| 划分 | `configs/data/real_v2_split.csv`，sha256 `9232ee35e7a59f4fac4534fc8e98fc454380b51e6e5d8439aaa0f769cd1090ee`；运行前核对，不符即停止 |
| 窗口台账 | `D:/UAV-Data/causal_input_v1/ledgers/real_included_windows.csv`（label = `qualified_y`；`normal_origin`、`phase`、`fault_onset_row` 等） |
| 窗口张量 | `D:/UAV-Data/causal_input_v1/windows/REAL__<...>.npz`，键 `x (n,26,150)`、`y`、`original_cache_index`、`window_start_row`；按 `original_cache_index` 与台账对齐，`y` 必须等于 `qualified_y` |
| 逐行数据（只用于 B0-3 曝光） | `D:/UAV-Data/causal_input_v1/cases/REAL__<...>.npz`，键 `row_input_valid`、`row_label_valid`；只读开发池案例 |
| 冻结源 GBT（S 臂） | `artifacts/ens_internal_test_v1_20261105/gbt_full_train.pkl`，sha256 `df468524…6774` |
| 特征与配方模板 | `artifacts/ens_internal_test_v1_20261105/run_test.py` 的 `stats()` 与 GBT 参数 |
| 复现门参照 | `artifacts/real_normal_adapt_v2_dev_20261106/real_dev_per_case.csv` 的 D0 行 |

运行位置：Real 数据只在本地，本轮 CPU 计算量小，在本地运行。解释器参数用 `-X utf8 -E -P -B`，不用 `-I`。

## 3. 步骤

1. **启动检查**：
   - 划分文件 sha256 一致；
   - 开发池 320 例，去掉 quarantine 后的案例数写入回执；
   - 冻结 GBT sha256 一致；
   - 输出目录 `artifacts/real_v2_b0_diagnostics/` 不能已存在。
2. **复现门（看任何 B0 指标之前）**：
   - 对开发池里 role_v1 为 adapt_normal 或 calibrate_normal 的案例，用 S 臂打分；
   - 每例的窗口数、S 概率 ≥ 0.5 的窗口数，都必须与 D0 行的 `windows`、`GBT_win_fp05` 完全一致；
   - 不一致就停止并报告。
3. **B0-1 会话捷径**：
   - 正类：开发池 normal_full 案例中 normal_origin = normal_flight 的窗口；负类：开发池故障案例中 phase = pre_fault 的窗口；
   - 类别平衡权重；按 `dev_fold` 做 5 折得到 OOF 概率；
   - 报告 AUC 和关联组聚类 bootstrap CI（2000 次，rng 20261108），并按飞行目录给出 AUC；
   - 敏感性：负类只取 window_end_row ≤ fault_onset_row − 250 的窗口（参考起点前至少 5 s）；
   - 描述：S 概率在两类上的中位数，以及 ≥ 0.3744、≥ 0.5 的比例。
4. **B0-2 可分性**：
   - R 臂：开发池窗口，label = qualified_y，权重与 gbt_full_train 相同（故障权重 1；正常总权重等于正常窗口数，在 normal_flight 与 normal_part_of_fault_case 之间各占一半），按 `dev_fold` 得到 OOF 概率；
   - S 臂：冻结源 GBT 直接打分；
   - 两臂都报告：
     - (a) 故障对全部正常的 AUC、AP；
     - (b) 故障对开发池故障案例 pre_fault 窗口的 AUC（**主数字**）；
     - (c) 故障对 normal_flight 窗口的 AUC，以及 (c) − (b)；
     - motor、sensor 分层的 (a)–(c)；
     - 所有 AUC 都给出关联组聚类 bootstrap CI；
   - 描述：在合并 OOF 上取使故障召回达到 0.5、0.8 的阈值，报告各正常总体的 FPR。正常总体分为：normal_full 飞行、补充正常段、pre_fault、post_fault_normal、qualified_y = 0 的 transition。注意补充正常段在台账里的 normal_origin 也是 normal_flight。
   - 按协议规则给出 B1 建议：
     - R 臂 (b) 合并 AUC ≥ 0.75 → GO；
     - CI 上界 < 0.65 → STOP；
     - 其余 → 交用户决定；
     - motor 或 sensor 的 (b) < 0.65 时，标记该层在 B1 中不受支持。
5. **B0-3 标签与曝光核查**：按协议逐项给出，写入 `label_audit.csv` 和 results.json。

## 4. 产物

- `artifacts/real_v2_b0_diagnostics/`：
  - `results.json`、`reproduction_gate.json`；
  - `oof_predictions.csv`（开发池窗口：case_uid、original_cache_index、dev_fold、family、phase、normal_origin、qualified_y、p_R、p_session、p_S）；
  - `label_audit.csv`、`run.log`、`execution_notes.md`。
- 报告：`docs/research/2026-11-08/跨域B0诊断结果_v1.md`。
- 脚本：`scripts/research/real_v2_b0_diagnostics.py`。如需单测，只能用合成数据。
- docs/、artifacts/ 下的新文件用 `git add -f`；提交后 **push**。

## 5. 回复格式（必须）

1. 启动检查与复现门：比对的案例数、是否全部一致、实际加载的 npz 文件数（应只含开发池）；
2. B0-1：AUC [CI]、敏感性 AUC、各飞行目录 AUC、S 概率描述；
3. B0-2：R、S 两臂的 (a)(b)(c) 表，含 motor/sensor 分层与 CI；FPR@召回表；B1 建议（GO / STOP / 交用户）；
4. B0-3 要点：故障前长度分布、无 pre_fault 窗口的案例数、曝光小时数、各折构成；
5. **完整偏差清单**：凡与协议或本文不一致的地方（实现选择、失败与重试、无法完成的项、额外分析），逐条列出；没有就写“无”；
6. 新增文件列表与 commit hash（确认已 push）。
