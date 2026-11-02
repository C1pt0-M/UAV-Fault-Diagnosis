# Balanced M0-matched 与 M1 的逐窗口分析

只读source-Val相同原ID的32,381窗与冻结四维sidecar；固定0.5，不重新训练、不访问最终test／Real。

- 结果与解释：`docs/research/2026-11-03/m1_balanced_window_difference_v1.md`。
- `results/newly_missed_fault_windows.csv`：192新增漏检窗口（含原ID、案例、场景、150帧N1～N4物理分布、保存概率与分支margin）。
- `results/corrected_normal_false_alarm_windows.csv`：166纠正误报窗口，字段同上；`results/changed_windows.csv`再包含68恢复故障和29新增误报，合计455。
- `results/summary.json`：8组转移、故障类别／SIL-HIL／飞行模式／phase、全组残差分布、source-Train正常相同域×工况参考、同案例描述性对照与直接残差adapter margin分解。
- `results/fault_domain_mode_transitions.csv`、`results/per_case_transitions.csv`、`results/typical_windows.csv`、`results/within_case_controls.csv`：分层、案例、典型ID和配对可追溯清单。
- `results/all_paired_predictions.csv`为全部32,381窗口原ID与双模型分数的本地工作台账；因5.6MB且部分重复，仅SHA列入manifest，不随GitHub轻量证据上传。公开455个预测发生变化窗口的完整明细，以及核对双方漏检／误报汇总及抽样规则所需证据。
- `independent_verification_v2.json`独立从两模型保存预测、129个case sidecar及冻结M1 best的权重张量复核全部455变化窗口与105／84个直接分支跨界计数；训练和主干没有重跑。

所有二进制原资产仍在本地`D:/UAV-Data/causal_m1_continuous_v1`和历史M1结果目录；manifest中的SHA不表示将本地权重／NPZ／全量张量上传GitHub。源Val已用于开发，不能用本次诊断作为最终泛化证据。
