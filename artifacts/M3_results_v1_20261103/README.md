# M3八组实验结果 v1

**八组全部完成；CO的Val Macro-F1最高，但七个候选均未通过全部零召回下降约束，只有参照C0满足约束。**见[简要报告](../../docs/research/2026-11-03/M3_eight_group_results_v1.md)。

本次只收集和核验已有训练结果，不训练、不运行模型前向或连续回放，不调阈值，不读取Test／Real。所有best指标来自相同32381个source-Val原窗口ID的保存概率。

## 结果表

当前权威结果目录为 **results_verified_v1/**。本地其他试算目录不是交付结果，不上传。

- [总体、best Epoch和权重SHA](results_verified_v1/overall_summary.csv)
- [总体、六类及两域Recall硬约束](results_verified_v1/hard_constraint_check.csv)
- [六类条件Precision／Recall／F1](results_verified_v1/fault_category_metrics_same_normal_pool.csv)
- [SIL／HIL总体指标](results_verified_v1/domain_metrics.csv)、[六类×域指标](results_verified_v1/fault_by_domain_metrics.csv)
- [正常工况／观测运动状态／hover前后段FPR](results_verified_v1/normal_state_and_flight_fpr.csv)
- [相对C0双向窗口迁移](results_verified_v1/paired_changes_vs_C0.csv)、[指定组间直接配对](results_verified_v1/selected_pairwise_changes.csv)
- [513轮Train损失与Val指标](results_verified_v1/training_curves.csv)
- [参数与已有开销证据](results_verified_v1/model_overhead.csv)、[各组训练配置](results_verified_v1/run_configuration.csv)
- [64项关键结果文件索引与SHA](results_verified_v1/result_file_index.csv)
- [核验摘要](results_verified_v1/collection_summary.json)、[交付清单](delivery_manifest.json)

模型只输出正常／故障。逐故障P/R/F1采用“该类故障为正例、同一正常池为负例”，其它故障不算健康，也不参与该条件指标；这不是六分类识别性能。六类Recall仍是各类故障窗检出率。两域表的条件正常池各自限定在同一域。

每轮Val loss、Train分类指标和trained-best推理时延没有保存在原产物中，均明确标缺失。每轮Val聚合指标来自原history；只有best保存完整概率，本次对其复算，没有伪造其他epoch预测。

## 原始日志、历史和二进制位置

[原始来源清单](collected/source_manifest.json)记录81项文件的服务器路径、大小和SHA；[完成状态](collected/formal/status.json)显示八组结束，无活动或排队任务。

每组原训练记录公开于`collected/runs/{C0,Ccap,CO,CO9,CV,CR,CN,CON_run}/{report,history,progress}.json`，完整训练日志于`collected/logs/{variant}.log`，其中CON日志发布为`CON_training.log`。Windows将CON视为保留名，故只为Git创建逐字节一致的安全路径副本；原始本地／服务器路径不变，见[路径映射](publication_path_mapping.json)。这些文件按原字节归档，不覆写远端或历史资产。

八组initial／best／last权重及完整best预测NPZ均已取回，保持在本地`artifacts/M3_results_v1_20261103/collected/runs/{variant}/`和原服务器`/root/autodl-tmp/M3_fixed_anchor_v1_20261103/formal/runs/{variant}/`；**训练权重、NPZ和大归档不上传Git**。本地`M3_results.tar.gz`为12,902,709字节，82个归档成员，SHA见[远端归档回执](remote_archive_receipt.json)，全部已取回核对。

配置沿用[预审协议](../../configs/experiments/M3_preflight_v1.json)和[固定anchor执行协议](../../configs/experiments/M3_fixed_anchor_execution_v1.json)。复核脚本为[scripts/research/collect_m3_formal_results.py](../../scripts/research/collect_m3_formal_results.py)，[针对性测试](../../tests/test_m3_result_collection.py)覆盖hover分段不混其他目录、固定选点／早停、条件Precision及仅AP浮点舍入容差。结果不授权追加实验。
