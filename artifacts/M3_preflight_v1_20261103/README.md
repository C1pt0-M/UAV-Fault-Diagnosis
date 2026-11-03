# M3训练前审查交付 v1

**本批完成本地技术协议审查与冻结，M3尚未具备正式开训条件。**总体及六类Recall容忍度、域约束角色和可选绝对下限均留待用户决定；正式训练授权0次。没有连接训练服务器、使用GPU、执行优化器更新或读取最终Test／Real。

报告：[M3_preflight_review_v1.md](../../docs/research/2026-11-03/M3_preflight_review_v1.md)。独立配置：[M3_preflight_v1.json](../../configs/experiments/M3_preflight_v1.json)，[SHA锁](../../configs/experiments/M3_preflight_v1.json.sha256)。配置的`ready_for_formal_training=false`、正式召回字段`null`具有实质含义，示例方案并未自动采纳。

## 当前权威证据

|文件|内容|
|---|---|
|[preparation_scope.json](preparation_scope.json)|本轮检查范围、0训练授权与禁止项|
|[source_scaling_final/normalization.json](source_scaling_final/normalization.json)|O1–O9新Train正常缩放＋保持原值的N1–N4统计；74700窗／11205000帧贡献|
|[source_scaling_final/normalization_table.csv](source_scaling_final/normalization_table.csv)|13维均值／std、单位和无效处理机器表|
|[source_scaling_final/data_preflight.json](source_scaling_final/data_preflight.json)|151833 Train／32381 Val的原ID、父缓存与脚本SHA、标签／case／pair检查和全13有限值检查|
|[model_checks_focal/model_preflight.json](model_checks_focal/model_preflight.json)|最终8组实际参数、公共未训练state、初始输出、原FocalLoss求导及实际N1／O6极值压力检查；0 optimizer step|
|[sampler_cpu_check.json](sampler_cpu_check.json)|8组本地CPU各100轮抽样的原ID摘要一致；非训练，跨正式PyTorch版本仍需再核对|
|[recall_context/saved_recall_context.csv](recall_context/saved_recall_context.csv)|历史保存源Val概率复算：原M0全域与共同ID、M0-matched、M1、两采样臂及SIL/HIL|
|[recall_context/recall_tolerance_window_budgets.csv](recall_context/recall_tolerance_window_budgets.csv)|多个百分点容忍度对应的净多漏窗上限，未采纳任何值|
|[recall_context/recall_sensitivity.json](recall_context/recall_sensitivity.json)|逐类约束方案与历史取舍，正式M3容忍度仍为null|
|[pytest_delivery.log](pytest_delivery.log)|最终相关41项测试通过，无优化器更新测试|
|[completion.json](completion.json)|最终完成范围、验证、未解决项和输入输出SHA|

`source_scaling/`是第一次CPU统计回执；后续仅为显式normalization SHA接口与最终代码身份重新验证，`source_scaling_final/`为当前权威。两次normalization.json／normalization_table.csv逐字节一致。`model_checks/`、`model_checks_final/`、`model_checks_extrema/`保留结构检查演进：依次加强纯C0梯度、实际极值，再用冻结FocalLoss完成最终检查。当前引用`model_checks_focal/`，旧回执不冒称覆盖后来新增检查。日志均保留，不覆盖历史结果。

## 复现范围

新代码为`packages/uav_fault/uav_fault/data/m3_features.py`、`models/cnn/mstca_m3.py`以及`prepare_m3_normalization.py`、`check_m3_models.py`、`analyze_m3_recall_constraints.py`。前者按原窗口重算O8首差0并继承全150／原生200ms N门禁；模型以独立13槽Adapter作一次128维相加，C0直接使用原OnlineMSTCA。没有新训练入口或自动GPU作业脚本。

从仓库根目录运行下面的只读／本地准备检查。复算目录必须不存在，脚本拒绝覆盖；先完成scaling后将其路径传给model检查。正式复现时应保留本批结果，在另外的不存在目录生成输出。

```bash
python scripts/research/prepare_m3_normalization.py --data D:/UAV-Data/causal_input_v1 --features D:/UAV-Data/causal_m1_continuous_v1 --output <new_scaling_directory>
python scripts/research/check_m3_models.py --data D:/UAV-Data/causal_input_v1 --features D:/UAV-Data/causal_m1_continuous_v1 --normalization <new_scaling_directory>/normalization.json --output <new_model_check_directory>
python scripts/research/analyze_m3_recall_constraints.py --data D:/UAV-Data/causal_input_v1 --features D:/UAV-Data/causal_m1_continuous_v1 --results artifacts/causal_m1_continuous_v1_20261102/results --original-m0 artifacts/causal_pure_mstca_m0_v1_20261101/results --output <new_recall_directory>
python -m pytest -q tests/test_m3_preflight.py tests/test_causal_mstca_m0.py tests/test_m1_continuous_dataset.py tests/test_continuous_kinematic_residuals.py tests/test_m1_continuous_replay.py tests/test_m1_natural_availability.py
```

sampler回执按现有WeightedRandomSampler的底层`torch.multinomial`、每组独立Generator seed53、同一冻结Train人口／weights生成100轮原ID摘要。它没有DataLoader、模型前向、优化器或训练循环；未来M3正式入口还须记录实际每轮draw身份和实现一致性。

当前本地PyTorch2.8.0+cpu／NumPy2.5.1，仅作结构与数据检查；原正式环境2.5.1+cu124／2.1.3仍须获准后复现未训练anchor、新Adapter与抽样。M3专用训练／评估／连续回放入口、冷启动和invalid-gap整合也尚待验收。原M1连续流通过不等于M3新增路径已通过。

原始日志、基础／残差张量缓存和权重留在原位置，本批没有复制或改写；最终Test／Real未访问。现有无关工作树修改保持原状。本批没有Git提交或推送。
