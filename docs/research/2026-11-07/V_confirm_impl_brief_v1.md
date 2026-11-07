# V 组实现简报（v1）：给 PI

- 协议：`configs/experiments/v_confirm_fixed100_mstca_v1.json`，开始前先校验它的 `.sha256` 锁。
- 设计说明：`docs/research/2026-11-07/V组固定轮数val确认协议_v1.md`。
- 协议与本简报冲突时，以协议为准，并报告冲突。

## 本次范围

只做本地实现和本地 CPU smoke。不使用 GPU，不连服务器，不读真实 val，不 commit。

## 要写的文件

尽量 import 复用 `train_t_selection.py`、`train_s_ablation.py`、`evaluate_s_ablation.py`、`evaluate_t_selection.py` 中已冻结的函数，不得修改这些文件。

### 1. `scripts/research/train_v_confirm.py`

- seed 取 {42..46}，只有 val 阶段。训练集 = `split_ledger(frame, 0, 'val')` 给出的 fit，即 train 减去 inner。
- 训练配方与 T runner 逐位相同，训满 100 轮。
- 每轮：保存 checkpoint 和 state digest，计算 inner CE（复用 `inner_ce`），保存 inner logits。
- 训练结束后：
  1. 用复用的 T0 回放得到 V0 的轮次；
  2. V0 和第 100 轮的 checkpoint 各预测一次全部 34710 个 val 窗口（如果是同一轮只预测一次）；
  3. 保存 float32 logits [n, 2]，附带 original_cache_index 和 labels。
- **重现门控（seed 42–44）：** 对照 `artifacts/s_ablation_mstca_v1/val/S0_seed<s>_fold0/report.json`，检查三项：
  1. best_epoch 处的 checkpoint digest；
  2. V0 选出的轮次；
  3. V0 的 val FP@R97 等于 785 / 428 / 741。

  任一不符：写 FAILED，返回非零。
- **正式运行前的检查：**
  - 环境断言；
  - 协议锁；
  - 规范化 code pin（`code_identities.files` 共 16 项）；
  - 9 项数据 identity；
  - T 组阶段 1 的 evaluation SHA 等于 `origin.t_phase1_evaluation_sha256`，从 `artifacts/t_selection_mstca_v1/evaluation_cv/evaluation.json` 读取，本地副本在 `artifacts/t_selection_mstca_v1/server_r1/evaluation_cv/`；
  - 已封存的实现 manifest；
  - 正式 slot 顺序。
- val 在本地绝不读取。本地 smoke 用 train 侧的子样本代替 val，并在报告中明确标注为 STAND_IN。

### 2. `scripts/research/evaluate_v_confirm.py`

- **输入：** 5 个 run 的 val logits。概率 = float64 softmax 的第 1 列。
- **bootstrap：** 共用一套重采样计划（val 的 association_group 簇，2000 次，rng seed 20261106）。
- **判定：** 严格按协议计算 C1–C4，输出 verdict 和未满足的标准列表。
  - **C2：** k = V0 的 FP@R97；V1 的阈值 t_m = V1 正常窗口概率中第 k 大的值，判警条件是 ≥ t_m；记录实际 FP。
  - **C3：** 两个臂各自取自己的 t*。
- **描述性指标：** 输出协议 `secondary_descriptive` 中的全部内容，包括 V3 集成，以及与 S 组阶段 2 S0 best/last 的对照。

### 3. `scripts/research/preflight_v_confirm.py`

按协议 `preflight_gates` 实现。不需要 GPU 确定性探针，由第一个 run 的重现门控代替。

### 4. `tests/test_v_confirm.py`

至少覆盖：

- **C1：** 每个分支，包括 seed 数正好 4/5、上界恰好等于 1。
- **C2 匹配阈值：**
  - 有并列分数时的实际 FP；
  - k=0 时判为不满足。
- **C3：**
  - 正好 2.0 pp 时满足；
  - 2.0 pp 加一点点时不满足；
  - 非 GPS 类超过上限时不满足。
- **C4。**
- **综合 verdict：** 每种失败组合。
- **logits→概率：** 与 S 组 `model(x).double().softmax(1)` 的路径逐位一致。
- **重现门控：** 不一致时失败。
- **训练脚本：**
  - V0 与第 100 轮同一轮时只预测一次；
  - 拒绝在 T 组阶段 1 SHA 不符时运行；
  - 本地模式不打开任何 val 文件。

## 本地验收

- 四个测试文件（factory、S、T、V）全部通过，要求 0 failed、0 skipped（basetemp 的处理方式同前）。
- CPU smoke：seed 42，2 轮，fit 512 / inner 256 / val stand-in 256。输出到 `artifacts/v_confirm_fixed100_mstca_v1/smoke/` 下的新目录。

## 回复要求

1. 文件清单及规范化 SHA256；
2. pytest 汇总；
3. smoke 的路径和 selections；
4. 完整偏离清单：每一次失败、重试、变通、解释和歧义都要列出，没有就写"无"。
