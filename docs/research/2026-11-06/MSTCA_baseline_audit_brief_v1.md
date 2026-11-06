# 任务书：MSTCA 基线（M0）结构与流水线审查 v1

委托方：架构设计（Claude）。执行方：PI agent。日期：2026-11-06。

## 背景（只需知道这些）

- 项目：无人机故障在线二分类（normal/fault），输入为 26 通道 × 150 帧（50 Hz，约 3 s）的因果历史窗口。
- 用户已决定**放弃 KG 分支**（9 维关系特征 + 门控）。接下来以纯 MSTCA（研究代号 M0，105938 参数）作为唯一神经基线。本任务的目的是**确保这个基线本身没有错误**，以便之后做结构消融。跨域和误报问题不在本任务范围内。
- 背景资料（按需查阅，不必通读）：`docs/research/科研技术路线总览_2026-11-05.md` 第 3.3–3.4、6.2、7.2 节；`docs/research/2026-11-06/KG-MSTCA结构审查与消融设计_v1.md`。

## 硬性边界

- **禁止**：GPU 训练；读取内部 test（`test` split）或任何 Real 数据；修改已有源码、配置、冻结协议、数据和权重；git commit/push。
- **允许**：阅读代码；在 CPU 上做小规模前向/反向检查（合成数据，或源 train/val 中最多几千个窗口）；在 `scripts/research/audit_mstca/` 下新建检查脚本；在 `artifacts/mstca_baseline_audit_v1_20261106/` 下写结果。
- 数据根目录 `D:/UAV-Data/causal_input_v1`（只读）。Python 运行时加 `-I`。

## 审查对象

|文件|关注点|
|---|---|
|`packages/uav_fault/uav_fault/models/cnn/lstm.py`：`SEBlock`、`_MSTCAFeatureEncoder`、`OnlineMSTCA`|结构本身|
|`scripts/research/train_causal_mstca_m0.py`|初始化与训练入口|
|`scripts/research/train_source_normal_sampling.py`|共享的采样、选点、早停、`state_digest`|
|`packages/uav_fault/uav_fault/data/causal_cache.py`|Dataset、标准化、mask|
|`packages/uav_fault/uav_fault/models/cnn/focal_loss.py`|损失|
|`configs/experiments/causal_pure_mstca_m0_v1.json`|冻结配置|
|`packages/uav_fault/uav_fault/online/fault_detector.py`|在线推理路径|
|`tests/test_causal_mstca_m0.py`、`tests/test_causal_cache.py`|现有测试覆盖|

## 需要逐项回答的问题

每项结论必须标注证据等级：**代码事实**（引用 `文件:行号`）/ **实测**（附脚本和输出）/ **推断**。

### A. 结构正确性

1. 逐层核对张量形状：26×150 → 三路卷积（24/24/16 通道）→ SE → MaxPool → … → 128 → Linear(2)。参数量是否确为 105938？MAC 是否为 6764928？
2. 时间长度 150→75→37：第二次 MaxPool 时最后一个时间步被丢弃。量化最新帧（窗口末端，对在线检测最重要）对输出的影响：对末端 k 帧（k=1,3,5,10）做扰动，比较输出变化与窗口开头同等扰动的变化（梯度或有限差分均可）。
3. 对称零填充：输入是 z-score，0 等于训练均值。窗口两端的填充是否造成边界伪影？末端帧的有效感受野是否比中部小？
4. SE 用的是对整个时间轴的均值，GlobalAvgPool 也会丢弃时间位置。说明模型对"故障出现在窗口哪一段"是否敏感，并用合成阶跃信号实测：同一阶跃分别放在窗口开头、中部、末尾。
5. 通道冗余：`acc_norm` 由 accel 推出；att_q 四元数有单位约束。这是否会造成问题？只需给出判断，不需要改。
6. BN：eval 模式是否被正确使用？Dropout 只在 head 上吗？是否存在 batch 大小为 1 时的问题（在线推理逐窗）？

### B. 初始化与可复现性（重点）

7. M0 的初始化是先构造 `OnlineKGMSTCA` 再把编码器和分类头映射过来（`train_causal_mstca_m0.py:62-93`）。请说明：
   - 当前映射的具体规则，以及哪些参数被映射、哪些被丢弃；
   - **放弃 KG 之后**，如果直接 `OnlineMSTCA(seed)` 重新构造，初始化是否与锚点不同（实测 digest）；
   - 为了让后续消融臂（去 SE、单尺度、CNN1D）也能"同起点、同规则"，应如何设计一个**不依赖 KG 类**的初始化方案，同时仍能复现现有 M0 锚点。只给出设计，不改代码。
8. 同 seed 在 CPU 上两次构造/前向是否逐位一致？说明有哪些非确定性来源（cuDNN、TF32、DataLoader workers）以及现有代码如何处理。

### C. 数据与训练流水线

9. 标准化：统计量来自哪里？通道顺序在训练 Dataset、冻结配置、`norm_stats.json` 和 `FaultDetector` 之间是否一致？请对同一个 val 窗口分别走训练路径和在线推理路径，比较输入张量和输出（若在线路径只支持旧 31 维资产，说明差异，不要强行运行）。
10. `baro_alt_relative` 逐窗相对化在训练和推理中是否以同样方式实现？
11. mask/未知标签（-1）是否在前向和 BN 之前被过滤？抽样核对一个 batch。
12. FocalLoss（gamma=2，alpha=None）的实现是否正确？与交叉熵在 gamma=0 时是否数值一致？
13. Balanced 采样：两种正常来源各占一半的权重是否按配置实现？抽样 10 万次检查经验比例。
14. 选点与早停：best 按完整源 val 的 Macro-F1 选。说明这对后续消融的公平性有什么影响（val 同时承担了选点和评估）。给出替代方案：固定轮数、train 内留组选点等，并说明利弊。
15. 预测规则：argmax、并列时判为正常、等价于阈值 0.5。核对实现。

### D. 现有测试覆盖

16. 列出现有测试覆盖了上面哪些项，哪些没有覆盖。

## 交付物

1. 报告 `docs/research/2026-11-06/MSTCA_baseline_audit_v1.md`（中文）：
   - 开头是一张总表：问题编号 / 结论（无问题、有问题、需关注）/ 严重程度（会使已有结果无效、影响公平消融、仅需记录）/ 证据等级。
   - 发现的问题按严重程度排序，每项给出：复现方法、影响范围（已有 M0/M2/M3/ENS 结果是否受影响）、建议修改（只写设计，不改代码）。
   - 最后一节：为后续消融（去 SE、单尺度 k7、CNN1D 参数匹配）给出统一的模型构造与初始化接口设计。
2. 检查脚本放在 `scripts/research/audit_mstca/`，输出放在 `artifacts/mstca_baseline_audit_v1_20261106/`。
3. 不确定的地方写"未核实"，不要猜。没有发现问题的项也要写明"已核对，无问题"及依据。
