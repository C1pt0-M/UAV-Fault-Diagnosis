# 正常来源采样：CPU 小规模流程检查通过

日期：2026-10-30。先固定 [CPU检查协议](normal_source_sampling_cpu_smoke_v1.json)，再运行两组检查。

> 后续正式配置与GPU启动入口已冻结，见 [正式运行说明](../../../scripts/experiments/README_source_normal_sampling_v1.md)。本文“仍未完成”记录的是CPU阶段边界，不代表当前仍缺启动脚本；正式GPU训练依然尚未执行。

**结果：流程通过。两组从完全相同的随机初始化出发，各完成2个小轮次、8次梯度更新，损失／梯度有限，最佳权重可重载并逐元素复现保存的验证概率。没有验证模型效果改善，也没有开始正式GPU训练。**

## 检查了什么

- 复用现有KG-MSTCA结构（26通道、150帧、156491参数）和FocalLoss；不改网络、不加载历史训练权重。
- 只打开源训练`train.pt`和源验证`val.pt`，使用既有源归一化。没有打开Real适配／校准／测试张量、源`test.pt`或RflySim留出缓存。
- 不调用旧`train()`入口，因为该入口会自动加载内部／Real最终测试；本次只复用模型和损失，在独立的源开发冒烟入口运行。
- 两组均为有放回采样。均匀组保持每窗口等权；均衡组只调整SIL/HIL两类正常来源，避免同时将采样机制变更作为额外混杂因素。
- 每组batch=32、每小轮次4批、共2小轮次，合计256次窗口抽样／8次更新；没有数据增强或学习率调度，这是本次检查的显式设置，不是正式训练配置。
- 验证从555个源验证案例各均匀取4个已有窗口，共2220窗口；按正常来源、正常工况、原无故障工况和故障案例类别分层输出，含48个原无故障hover窗口。
- 每组保存历史、最佳及最后权重、最佳轮次元数据和预测；保存失败不得破坏之前的最佳权重，已有输出目录拒绝覆盖。

CPU端约39秒完成；服务器未连接，部署模型、冻结数据／协议未修改。

## 采样概率是否正确

原训练缓存正常87030、故障81770。均衡组给两类SIL/HIL正常各分配其合计83350个窗口的一半权重，即各41675；故障与RflySim正常每窗口仍为1。

| 概率，占所有训练抽样 | 均匀组 | 正常来源均衡组 |
|---|---:|---:|
| 故障 | 48.4419% | 48.4419% |
| 正常总计 | 51.5581% | 51.5581% |
| 原无故障正常 | 12.6635% | 24.6890% |
| 故障前正常 | 36.7145% | 24.6890% |
| RflySim正常 | 2.1801% | 2.1801% |

解析概率检查通过；每组另抽10万次检查可重复性与实际分布。**概率保持不变不等于每个有限批次的类别数量完全相同。**这次极小训练抽样中，均匀组故障138/256，均衡组124/256；不能拿这点训练量比较性能或收敛。

本次均匀组不是旧训练的`shuffle=True`无放回精确复现。两组共同使用相同的有放回机制，只检查重分配正常来源权重这条路径。

## 保存与独立复核

本地目录：`artifacts/normal_source_sampling_cpu_smoke_20261030/`。

- `report.json`：协议、输入／源码哈希、初始权重标识、理论与实际采样分布、训练状态和分层评估。
- `shared_initial.pt`：两组共同的随机初始化。
- 两个实验臂目录：`history.json`、`checkpoint_metadata.json`、`best.pt`、`last.pt`、`validation_predictions.npz`。
- `smoke_val_links.csv`：2220个验证窗口的原缓存索引及案例／来源字段。
- `independent_verification.json`：独立重算采样质量、复现抽样序列、核对参数更新、重载最佳权重重新前向后概率逐元素相同、分层误报重算。

训练采样器独立generator seed=53；10万次分布探针使用全局约定seed=42。模型／Dropout每组重置到相同seed=42，采样随机数与模型随机数分离。

30项相关最小检查通过，其中新增5项覆盖：采样概率与重复性、标签／来源及缺失分组拒绝、分层单类边界、权重写失败保留旧最佳、已有目录拒绝覆盖。不是系统全量验收。

```text
python scripts/research/smoke_normal_source_sampling.py --protocol docs/research/2026-10-30/normal_source_sampling_cpu_smoke_v1.json --cache D:/UAV-Data/online_cache_v2 --audit artifacts/source_training_coverage_audit_20261030_r2 --output <新目录>
python -m pytest -q tests/test_normal_source_sampling_smoke.py tests/test_source_training_coverage.py tests/test_feature_branch_sensitivity.py tests/test_check_bn_estimator_dev_v1.py tests/test_diagnose_normal_adaptation_v1.py tests/test_run_normal_adaptation_v1.py tests/test_run_normal_replay_v1.py
```

## 仍未完成

正式训练轮数、调度／早停、完整验证选模规则、重复种子及GPU启动入口尚未另行冻结和集成。本脚本刻意限制为CPU小测试，不能直接称为“已具备正式GPU实验的一键启动”。下一步先固定正式成对训练协议、准备运行包，再开GPU；不根据本次少量更新的分数选择方法，也不提前重测Real最终集。
