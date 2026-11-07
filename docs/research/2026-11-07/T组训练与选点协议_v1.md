# T 组：训练长度与选点协议（v1，冻结于任何正式运行之前）

日期：2026-11-07。机器可读协议：`configs/experiments/t_selection_mstca_v1.json`，SHA256 锁在同名 `.sha256` 文件中。
前置结论：`S组结构消融结果_阶段1_v1.md`、`S组结构消融结果_阶段2_v1.md`。

## 1. 动机

S 组说明，结构差异对误报的影响不超过 10%，而选点、训练长度和 seed 的影响在 20–80%：

- inner CE 选点经常选到早期轮次：36 个 run 中有 13 个的 best epoch ≤ 23。
- S0 用 last 权重时，FP@R97 比用 best 权重低 9–20%。
- 在 val 上，S0 的 seed 43 训练到第 94 轮，FP 为 428；其余 seed 约 750。

所以降低误报的第一步是固定训练与选点协议，而不是继续调整网络结构。

## 2. 基线

架构固定为 MSTCA（S0 / M0，含 SE 和多尺度），只改变训练长度和选点规则。保留完整 MSTCA 的理由：

1. 按冻结协议，它判为优于参数匹配 CNN，并在 val 上 CONFIRMED；
2. 与历史结果连续；
3. S1 和 S2 都是 TRADE_OFF，不是干净的改进；
4. 同时换架构会与选点效应混杂。

单尺度 k7（S2）的简化留到以后单独评估。

## 3. 设计

**从 S 协议 r2 继承的部分：** 初始化、数据与折、inner 划分（`s_ablation_inner_v1|`）、训练配方、确定性、分层、FP@R97 定义、bootstrap 设置和 undefined_policy。

**与 S 协议不同的部分：**

- 不早停，每个 run 都训练满 100 轮；
- 每轮保存 checkpoint，并预测 inner 和外层 held-out；
- 选点规则在训练结束后按冻结定义回放；
- seed 增加到 42–46，共 5 个，以减小 seed 方差的影响。

### 3.1 比较臂

|臂|规则|说明|
|---|---|---|
|T0|inner_ce_es|原 S 规则，在 inner CE 历史上回放 `selection_step`；参照臂，seed 42–44 必须等于 S0|
|T1|fixed_100|第 100 轮（余弦调度结束）的权重，不使用 inner|
|T2|inner_fp97|inner FP@R97 最小的轮次，范围 1..100，平局取较晚的轮次|
|T3|fixed_100_seed_ensemble|5 个 T1 模型的平均概率，次要比较，计算量是单模型的 5 倍|

### 3.2 运行中的门控

- seed 42–44 每个折中，T run 在 S0 best_epoch 的 checkpoint digest 必须等于 S0 报告里的 `checkpoint_state_digests.best`。
- 用 T0 规则回放 inner CE 历史，选出的轮次必须与 S0 相同。
- T0 的 pooled FP@R97 必须等于 3595 / 3588 / 4610。

任一不符就停下来报告。这同时检验了额外推理没有扰动训练的随机数。

### 3.3 判定

**比较：** 主比较是 T1 对 T0、T2 对 T0；次要比较是 T2 对 T1、T3 对 T1。

**比值：** 每个 seed 算 FP@R97(臂) / FP@R97(T0)，5 个 seed 取几何平均；CI 用关联组配对 bootstrap。

**优先顺序：** 依次检查下列条件，取第一个成立的：

1. 比值或 CI 无定义 → INCONCLUSIVE；
2. seed 不一致 → INCONCLUSIVE。seed 一致性要求 5 个 seed 中至少 4 个严格落在 1 的同一侧；如果 CI 落在边际 [1/1.10, 1.10] 内，仍可判为 EQUIVALENT；
3. 类约束不满足 → TRADE_OFF。类约束是：late_hover_fault 以外，任何故障类的召回下降不超过 1 pp；
4. BETTER：上界 < 1；
5. WORSE：下界 > 1；
6. EQUIVALENT：CI 落在 [1/1.10, 1.10] 内；
7. 其余为 INCONCLUSIVE。

**AP 一致性：** BETTER 或 WORSE 与 delta AP 的方向矛盾时，改判为 INCONCLUSIVE。

**采纳规则（由阶段 1 的判定机械决定）：**

- T1 和 T2 都是 BETTER：采纳 T1；只有当 T2/T1 的上界 < 1 时才采纳 T2。
- 只有一个是 BETTER：采纳它。
- 都不是 BETTER：T1 为 EQUIVALENT 时采纳 T1，否则保留 T0。

平局时优先 T1，因为固定轮数没有选点噪声，也不需要 inner 集。是否用集成属于部署决策，不影响采纳。

**描述性分析：** 按 held-out FP 随轮次的曲线找出 oracle 轮次，只用来说明还有多少提升空间，标注为"不可选"。

### 3.4 运行

**阶段 1：** 15 个 run，顺序为 seed 42→46、fold 0→2，约 4 GPU 小时。

**阶段 2：** 阶段 1 的判定及其 SHA 记录后才能开始。

- 5 个 run（每个 seed 一个），用 train 减去 inner 训练 100 轮。
- 只用 T0 选出的 checkpoint 和采纳规则选出的 checkpoint 各预测一次 val，约 1.5 GPU 小时。
- 确认标准：采纳规则判为 BETTER 时，val 几何平均比值需 < 1；判为 EQUIVALENT 时，val 比值需落在边际内。

## 4. 环境与禁止事项

**环境：** 与 S 组相同，并断言 `ATEN_CPU_CAPABILITY=default`。golden 不一致时停下报告，不得自行更改 CPU 分发设置。

**code pin：** 对换行规范化（CRLF→LF）后的字节取 SHA。

**禁止：**

- 使用 internal test 或 Real 数据；
- 在任何选点中使用 held-out 或 val；
- 看到结果后修改臂、规则、边际或 seed；
- 改动 S 组的冻结文件或产物；
- 在服务器上重新生成 golden。

## 5. 预期与风险

- 如果 T1 判为 BETTER，那么"训练满轮次"就是降低误报最简单的手段，后续实验统一用 fixed_100。
- inner 只有约 100 个组，用 inner FP@R97 选点（T2）可能噪声较大，因此 T2 不一定优于 T1。
- 存储估算：每个 run 保存 100 个 checkpoint，约 100 × 1–2 MB；每轮的 held-out 预测为 float32，100 × 约 54k 个窗口，每个 run 约 22 MB。可以接受。
