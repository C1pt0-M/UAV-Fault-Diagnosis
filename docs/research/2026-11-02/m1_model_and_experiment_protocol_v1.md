# M1模型接入与公平实验口径v1

2026-11-02。用户要求保留MSTCA骨干、从M0加载共同预训练参数，并在窗口人口改变时配套M0-matched。本文件在任何M1正式GPU更新前明确设计，不改M0、M2代码／数据／结果。

## 模型

OnlineMSTCAKinematicM1为独立类，位于models/cnn/mstca_kinematic_m1.py。基础输入B×26×150，与N1～N4输入B×4×150一一同时间戳、同原始窗口ID。基础26通道仍用原冻结统计；四维先物理计算，另用source-train全窗已知正常、全残差有效的窗口拟合均值／总体标准差，重叠150帧贡献按协议显式保留。

骨干直接复用OnlineMSTCA.encoder及head，不复制／改写多尺度卷积实现。四维adapter为Conv1d(4,16,k3,p1)→ReLU→Conv1d(16,16,k3,p1)→ReLU→AdaptiveAvgPool1d(1)→Linear(16,128)。两层局部时序卷积是最小独立适配，避免将四维降成事先固定窗口常数；池化后输出与基础128维特征相加一次，再共用原分类头。没有旧九维公式、关系MSTCA编码器或门控，不把原始输入简单拼成30通道。

adapter参数：208＋784＋2176＝3168；M0 105938，M1共109106。记录开销，不将额外容量本身叫作物理创新。adapter只新增ReLU／卷积／线性，无新增BN；基础BN只见严格合格人口。输出projection的weight和bias初始化为0，其他新增层登记seed42默认初始化；在eval模式M1初始函数与公共M0精确相同。第一步投影有梯度，后续卷积层在投影更新后可学习，已用3步CPU合成更新验证。

## 公共预训练参数公平性

此前准备建议使用未训练锚点，但本轮用户明确“尽可能从M0加载共同的预训练参数”。因此新实验优先使用各采样组已完成M0的best.pt作为共同起点：Uniform的M0 best65、Balanced的M0 best73，SHA与state digest明确登记。

- M1与对应M0-matched都加载同一个arm的M0预训练encoder/head，包括BN buffer，逐张量精确校验；不是只给M1预训练优势。
- 原initial.pt随机锚点仍用于现有M0构造及源码验证，随后显式加载已核best；不把训练过的best冒称随机初始化。
- 新实验均使用新AdamW，不续用原optimizer，100轮上限、warmup5、batch512、lr3e-4、wd1e-4、Focal gamma2／alphaNone、clip1、min_lr1e-6、min30／patience20／delta0.002等按M0协议，参数初始化来源是明示新实验差别。
- 该结果解释为“共同M0预训练起点、同人口／预算继续学习时，四维adapter的增量”，不是从头训练的纯特征唯一效应。M0已使用全部源train、源val选best的历史暴露继续披露，不称新的盲验证。

如全量连续残差人口发生改变则本轮最多4个正式作业：M0_matched Uniform、M1 Uniform、M0_matched Balanced、M1 Balanced，各seed42一次；不新增seed／跨度／架构。两变体同训练／验证原ID、完整150帧、共同valid集合及重算的正常来源权重。原M0全覆盖结果仍保留，既有M2只作历史参照。

## 正式训练门禁

全量特征与manifest、逐项覆盖、仅train正常统计、独立数值核验、Dataset对齐与mask门禁、模型forward/backward、CPU真实数据小样本更新、连续buffer／事件基本回放全部通过才可启动。任何失败先停下，不以改阈值、补零／均值、删难例或改名单绕过。若严格全窗人口损失过大／明显类别偏差，应先报告处理方案，不自动放宽。

此次只冻结结构与解释口径；训练配置中的实际窗数、mean/std、manifest SHA及作业批准门禁将在全量交付后写入，不能现在凭局部比例虚构。原M0／M2资产hash保持不变。
