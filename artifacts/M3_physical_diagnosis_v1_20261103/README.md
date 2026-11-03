# M3物理数据诊断 v1

[诊断报告](../../docs/research/2026-11-03/M3_physical_data_diagnosis_v1.md)直接回答六个问题。结论：**停止当前13维无条件融合，保留C0／纯MSTCA基线，后续转向工况条件化残差的有限验证；不认定条件方法已成功。**

## 范围

已有共同Train151833、Val32381，全部窗口一次性计算统计摘要。154个CN新增Hover故障漏检全部保留：145纯故障、9混合／窗内标签不全有效；29案例的原生支持均核验。6份原始source-Val ULog做限定回溯，不重建全库日志或回放。

无服务器、GPU、模型构造／前向／拟合／训练、阈值搜索、Test／Real或新候选公式。旧9计算调用原函数，新4使用原sidecar及原生dt定义。原模型、公式、配置、归一化、标签和输入数据未修改。

## 主要证据

- [signed_case_AUC.csv](signed_case_AUC.csv)：同split／域／目录，一案例一中位摘要，对独立正常的固定signed AUC。不是窗口分类性能或模型增量。
- [motion_conditioned_signed_AUC.csv](motion_conditioned_signed_AUC.csv)、[控制覆盖](motion_control_coverage.csv)：再控制此前冻结vel／gyro运动代理；不匹配的故障状态明确保留。
- [案例特征中位数](case_feature_medians.csv)、[案例×状态中位数](motion_conditioned_case_medians.csv)：13候选和原26的可追溯摘要，不把大量窗当独立样本。
- [工况分布](condition_distributions.csv)、[均值／时间方差／半窗变化](condition_time_moments.csv)、[同案例前后响应](same_case_response.csv)。方差分别命名为案例间或时间窗内，不混用标准误或置信区间。
- [13×Base26相关](Base26_candidate_correlations.csv)、[Base26摘要匹配](Base26_summary_matched_case_differences.csv)：报告条件化与共同支持。摘要近邻不等于完整序列相同，低相关／匹配后差异都不是独立增量证明。
- [Hover窗口](hover_window_medians.csv)、[错误分层分布](hover_error_distributions.csv)、[同案例正确故障对照](hover_same_case_correct_controls.csv)：保留所有154个目标和两种标签纯度。
- [154窗原生来源检查](all154_native_provenance.csv)：有效性、reset、年龄／异步、cache对应和原公式点核验，无重填数据。
- [6日志原生值／时钟核验](six_raw_log_checks.csv)、[原始sensor话题分阶段统计](raw_sensor_topic_phase_statistics.csv)、[逐窗旧9响应](six_case_old9_window_responses.csv)。GNSS整数坐标原样统计，不擅自当物理真值；稀疏原话题、小样本和共享估计链路的限制见报告。
- [统计阶段回执](completion.json)、[执行日志](execution.log)、[独立核验](independent_verification.json)、[针对性测试日志](pytest_physical_diagnosis.log)。独立直接成对比较5694项案例AUC，最大舍入误差1.11e−16；3项测试通过。

## 图

只画两个必要问题，不做均值柱或窗口级显著性：

1. [状态控制对signed AUC的影响](figures/hover_AUC_state_control.png)：每点一个split／域统计，无连线、无方向翻转、无伪置信区间；故障／正常案例数在CSV和图注。
2. [原始故障与正常Hover曲线](figures/native_hover_fault_and_normal.png)：原裁剪连续独立帧；N3/N4显示用signed symlog，未变更候选输入；无效位置留空，灰带为变化窗并集。

PNG 300dpi、SVG及灰度版一并归档。字体／边界／刻度布局程序检查通过；当前图像工具未向代理展示图片，因此**没有冒称独立人工视觉复核通过**。

## 本地大资产

`local_summaries/`包含Train/Val原ID对应的统计NPZ和元数据／运动代理；`local_traces/`包含6案例原始物理序列。只本地保留、不进Git，SHA和字节索引见[发布清单](publication_manifest.json)。原始ULog／旧缓存／权重／预测均保留原处，本次不上传它们。

`completion.json`保留首次摘要阶段的回执；后续运动条件统计、原始sensor描述、图、报告及最终发布范围由`publication_manifest.json`索引，不伪造首次就已生成全部后续证据。

复算（输出必须是新目录）：

```bash
python scripts/research/diagnose_m3_physical_data.py --output <新的独立目录>
python scripts/research/plot_m3_physical_diagnosis.py --results <统计目录>
python -m pytest -q tests/test_m3_physical_diagnosis.py
```

所有结果仅支持开发Train／Val上的描述与机制假设；不作为真实损伤认证、最终盲测、独立传感器信息证明或研究路线成功证明。
