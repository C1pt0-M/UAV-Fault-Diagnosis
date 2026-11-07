# W 组（阶段平衡加权）PI 指令 v1

协议：`configs/experiments/w_phase_balance_mstca_v1.json`（r1，以提交该文件的 commit 作为冻结记录）。
背景：`docs/research/2026-11-07/误报诊断分析_v1.md`。

**工作方式（按用户要求）：**
- 本轮直接在服务器 `ssh autodl_UAV` 上实现并跑完，一轮交付。
- 不做 SHA 锁、implementation manifest 或本地 smoke 回合。
- 服务器上的 python 用 `artifacts/s_ablation_mstca_v1/runtime_pytest_r1/bin/python`。
- 环境变量 `CUBLAS_WORKSPACE_CONFIG=:4096:8` 和 `ATEN_CPU_CAPABILITY=default` 要显式设置并 assert。

## 1. 要做什么

### 1.1 训练器 `scripts/research/train_w_phase_balance.py`

以 `scripts/research/train_t_selection.py` 的 CV 路径为基础，复用它的 import 和函数，不要复制修改冻结文件。唯一的实质改动是采样权重：

1. **基础权重：** `b = sampling_weights(labels, origins, True)`，与 T 相同。
2. **起始参考时间：** 只用 fit 集合，对每种飞行类型（`case_uid.split('/')[1]`），取有效起始的故障案例（`fault_onset_row >= 0` 的唯一 `case_uid`；ledger 用 -1 表示无起始，必须排除）的 `fault_onset_row * 0.02` 中位数。（r1.1 澄清）
3. **阶段：** `window_end_row * 0.02 > onset_reference[flight]` 为 post，否则为 pre。
4. **格子：** 共 12 个（6 种飞行类型 × 2 个阶段）。每个格子算 `N_c = b[正常].sum()`、`F_c = b[故障].sum()`、`r_c = F_c / N_c`（`N_c = 0` 时取 1）。
5. **乘子：** W1 下正常窗口乘以 `max(1, r_c)`，W2 下乘以 `max(1, sqrt(r_c))`。故障窗口保持 `b` 不变。不做重新归一化。
6. **采样器：** `WeightedRandomSampler` 的抽样数、generator 种子、loader 设置与 T 完全相同。
7. **其他不变：** 100 轮训练，每轮的 inner 和 held-out 预测，T0 规则 replay（未加权的 inner CE），全部与 T 相同。
8. **report.json** 写入：各飞行类型的 onset reference、12 个格子的 `N_c`、`F_c`、`r_c`、乘子、窗口数，以及 sampler 的有效样本量 `sum(w)^2 / sum(w^2)` 和各格子的采样概率。
9. **checkpoint：** 至少保留 T0 所选轮次和第 100 轮。磁盘够用的话可以全部保留。

参考值（r1.1，排除 -1 后，fold 0 fit 集合，PI 预算）：W1 的 post 格子乘子为 acce 3.6148、circling 19.6427、dece 2.5915、hover 3.9351、velocity 3.9778、waypoint 8.4783，pre 格子都是 1。W2 的乘子取这些值的平方根。有效样本量请自行计算并报告。你算出的数值应与此一致，不一致要在报告里说明原因。

### 1.2 评估器 `scripts/research/evaluate_w_phase_balance.py`

- 复用 `evaluate_v_confirm.py` 中的 `matched_fp_threshold`、`criteria` 逻辑、bootstrap 方案，以及 `evaluate_t_selection` / `s_ablation` 中的 FP@R97 定义和 strata。
- **W0** 读取现有 T CV 产物：`artifacts/t_selection_mstca_v1/cv/seed<s>_fold<f>/heldout_predictions.npz`，取 `selections.json` 中 T0 轮次的概率。
- **C2 和 C3 的类窗口：** 包括 late-hover 故障在内的全部故障窗口。这一点与 V 组不同，协议里写明了。V 式口径（排除 late_hover_fault）作为描述性指标另行报告。
- **判定：** 每个 arm 判 PASS 或 FAIL，再按协议的 adoption 规则机械得出 `KEEP_T0` / `ADOPT_W1` / `ADOPT_W2`。
- **描述性指标：** 按协议 `secondary_descriptive` 一项不落，其中最重要的是独立正常窗口按"飞行类型 × 时间段"的误报率。

### 1.3 测试 `tests/test_w_phase_balance.py`

- 所有乘子强制为 1 时，权重与 `sampling_weights(..., True)` 逐元素相等；
- 用一个玩具 ledger 测试格子划分和 onset reference；
- W1 和 W2 的乘子公式；
- 评估器在合成数据上的判定分支（PASS、FAIL 和 adoption 的三种情形）。

## 2. 服务器上的执行顺序

1. 用 bundle 同步代码，方式同前。
2. 运行 `pytest tests/test_mstca_ablation_factory.py tests/test_t_selection.py tests/test_w_phase_balance.py`。golden 不一致就 **STOP** 并报告，绝不重新生成 golden。
3. **W0 复现门：** 用新评估器读 T0 预测，seed 42、43、44 的 FP@R97 必须等于 3595、3588、4610，不等就 STOP。
4. 正式运行 30 个 run，顺序为 seed → fold → arm（W1, W2）。如果显存允许，可以两个 run 并行。用 `nohup` 或 tmux 挂起，避免 SSH 断开导致中断。
5. 跑评估，输出 `artifacts/w_phase_balance_mstca_v1/evaluation_cv/evaluation.json`。
6. 把新脚本、测试、所有 run 的 report.json / selections.json 和评估输出提交到仓库，`artifacts/` 用 `git add -f`。大文件（checkpoint、预测 npz）留在服务器上，不提交。

## 3. 禁止事项

- 读取 source val、internal test 或任何 Real 数据。
- 把时间、阶段或位置作为模型输入；用 held-out 预测做选点或加权。
- 看到结果后修改 arm、乘子、判定标准或 adoption 规则。
- 丢弃或替换 seed。
- 修改 S/T/V 组的冻结文件或产物。
- 重新生成 golden；未经架构师批准改变 CPU dispatch 设置。

## 4. 回报内容

1. 每个 arm 的 verdict、失败的标准，以及 adoption 结果。
2. 每个 seed 的 FP@R97：W0、W1、W2，以及比值。
3. C1–C4 的全部数值和 CI。
4. 各格子的乘子和有效样本量，与上面的参考值对照。
5. 独立正常窗口按"飞行类型 × 时间段"的误报率表（W0、W1、W2）。
6. 各 (arm, seed, fold) 选中的 T0 轮次。
7. **完整的偏差清单：** 列出所有与本指令或协议不一致的地方，包括实现上的选择、重跑、传输问题和环境差异。没有偏差也要明确写"无"。
8. commit hash。
