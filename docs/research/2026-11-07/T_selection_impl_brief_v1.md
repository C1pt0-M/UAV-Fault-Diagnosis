# T 组实现简报（v1）：给 PI

协议：`configs/experiments/t_selection_mstca_v1.json`。实现前先运行 `sha256sum` 校验 `.sha256` 锁。协议与本简报冲突时以协议为准，并报告冲突。
设计说明：`docs/research/2026-11-07/T组训练与选点协议_v1.md`。

## 本次范围

只做本地实现和本地 CPU smoke。不使用 GPU，不连远程，不读 val，不 commit。

## 要写的文件

### 1. `scripts/research/train_t_selection.py`

尽量复用 `scripts/research/train_s_ablation.py` 中已冻结的函数，可以 import，也可以逐字复制，但不得修改原文件。涉及的函数：`split_ledger`、`CacheWindows`、`class_balanced_ce`、`inner_ce`、`selection_step`、环境断言、数据 identity、golden 校验。

- **固定配置：** arm 固定为 mstca，seed 取 {42..46}，fold 取 {0,1,2}，phase 取 {cv, val}。
- **训练：** 训练配方、sampler、optimizer、余弦调度（T_max=100）都与 S runner 逐位相同；只是去掉早停，训满 100 轮。
- **每轮结束后（eval 模式）：**
  1. 保存 `checkpoints/epoch_<e>.pt` 和 state digest，digest 的计算方法与 S runner 相同；
  2. 计算 inner CE；
  3. 预测 inner 概率；
  4. 仅在 cv 阶段预测外层 held-out 概率。
- **随机数隔离：** 额外推理用独立的 DataLoader（shuffle=False，num_workers=0，显式 `torch.Generator`），不得消耗训练用的 RNG。
- **输出：**
  - `inner_predictions.npz`：形状 [100, n_inner]，float32，附带 window id 和 label；
  - `heldout_predictions.npz`：形状 [100, n_heldout]；
  - `selections.json`：T0、T1、T2 选出的轮次，以及 T0 回放的 stop epoch；
  - `report.json`：字段尽量与 S 报告对齐，另加 per-epoch history。
- **阶段 2（本次只实现，不运行）：** 训练集为 train 减去 inner，不做 held-out 预测。训练后只用 T0 和采纳规则选出的 checkpoint 各预测一次 val；采纳规则通过命令行参数传入，并且必须附带阶段 1 evaluation 的 SHA。在阶段 1 判定记录之前，拒绝运行阶段 2。
- **重现门控：** seed 42–44 的 cv run 结束时，对照 `artifacts/s_ablation_mstca_v1/cv/S0_seed<s>_fold<f>/report.json` 检查两项：S0 best_epoch 处的 digest，以及 T0 回放选出的轮次。任一不符，写入 FAILED 并退出，返回码非零。
- **正式运行前的检查：** 断言环境（含 `ATEN_CPU_CAPABILITY=default`），并校验协议锁、规范化 code pin（按协议 `code_identities.hash_rule`）和数据 identity。

### 2. `scripts/research/evaluate_t_selection.py`

- 读取 15 个 run，按协议构造 T0、T1、T2、T3 的 pooled held-out 预测。
- 用继承的 FP@R97 定义和 bootstrap（2000 次，rng seed 20261106，所有臂和 seed 共用同一套重采样）计算：
  - 比值和 CI；
  - delta AP；
  - 类召回；
  - 协议列出的全部次要指标；
  - 描述性的 oracle 曲线。
- 判定严格按协议 `decision_rules` 中的优先顺序，并输出采纳的规则。
- 检查 `evaluation_reproduction`：T0 在 seed 42–44 上的 FP 必须为 3595 / 3588 / 4610。
- 支持阶段 2 的确认。

### 3. `scripts/research/preflight_t_selection.py`

内容参照 `preflight_s_ablation.py`。确定性探针用 T runner 跑 seed 42 fold 0 两轮，从头跑两次，digest 必须相同。

### 4. `tests/test_t_selection.py`

至少覆盖：

- **T0 回放：**
  - 构造的 CE 序列能得到预期的 best 和 stop；
  - 没有 stop 时取到第 100 轮为止的最小值。
- **T2：** 平局取较晚的轮次。
- **T1：** 恒为 100。
- **T3：** 每个窗口取 5 个模型的平均。
- **判定：**
  - 优先顺序的每个分支各一例，包括 equivalent_seed_mixed；
  - AP 一致性；
  - 采纳规则的 3 种情形。
- **重现门控：** 不一致时失败。
- **RNG 隔离：** 用小型合成数据跑 3 轮，开启和关闭额外推理时，最终 digest 必须相同。
- **新 golden：** (mstca, 45) 和 (mstca, 46) 的初始化 golden 在本地 CPU 上生成一次，写入本测试文件。生成方法与 `tests/test_mstca_ablation_factory.py` 中现有 golden 的生成方法完全相同，需在回复中说明命令。已有的 12 个 golden 不得改动。

## 本地验收

- `pytest tests/test_mstca_ablation_factory.py tests/test_s_ablation.py tests/test_t_selection.py` 全部通过，S 组原有测试不回退。
- CPU smoke：seed 42 fold 0，2 轮，fit 不超过 2048 个窗口，inner 和 held-out 也做子采样。输出写到 `artifacts/t_selection_mstca_v1/smoke/`，并且必须有 `selections.json` 和两个 npz，形状为 [2, n]。
- 不得改动 S 组的任何文件、`scripts/research/case_relative_features_dev.py`，以及 `tmp_control_compensation_audit*.log`。

## 回复要求

1. 新增或修改的文件清单，以及规范化 SHA256。
2. pytest 汇总，含新 golden 的值和生成命令。
3. smoke 输出路径和关键数字。
4. **完整的偏离清单：** 每一次重试、失败、环境变通、对协议的解释，以及发现的歧义，都要列出。没有就写"无"。不得省略。
