# 有限工况条件化N残差验证 v1

[简洁结论报告](../../docs/research/2026-11-03/M3_conditioned_residual_validation_v1.md)只回答四个指定问题。

**结论：不支持本次固定统计方案达到“保持检出、降低误报”，不建议据此进入模型实验。**五种已覆盖运动状态内的排序有局部提高，但正常总FP550→818；机动／translation状态减少误报时又损失检出。没有为得到正面结果调整分组、阈值、参考人口或范围。

## 唯一方案与冻结记录

- [protocol.json](protocol.json)、[SHA锁](protocol.json.sha256)：本轮新Val结果读取前固定的统计方案，不修改历史实验配置。
- N1～N4原有signed完整150帧中位数；只用原连续残差缓存，未加入旧9维。
- 状态沿用此前冻结的`observed_states(raw, columns)`，只看过去150帧物理vel_xyz／gyro_xyz；标签、类别、目录、域和未来帧不传入该函数。
- Train完整已知正常74700窗，包含独立正常及故障案例已知正常部分；每个案例总权重相等，案例内等权。没有训练分类器、回归器或神经网络。
- 各维固定Q0.005／Q0.5／Q0.995，任一维超出边界算异常；精确边界接受。99%是每维参考范围，不是四维联合1%误报保证。归一化距离是统计异常分数，不是概率、修改后的物理特征或C0输出。
- [normal_references.json](normal_references.json)在Val加载前保存并核对未改变。无条件和条件化都使用相同分位／判定规则，无搜索、无方向翻转、无Val调整。

## 覆盖与结果

- [Train状态覆盖](Train_state_support.csv)、[正常参考分布](Train_normal_reference_distributions.csv)：五状态通过固定最低覆盖门槛；descent没有正常Train参考。
- [Val状态覆盖](Val_state_coverage.csv)：原32381个ID全部保留，下降216个故障窗标为条件化无法验证；没有fallback或补填。
- [全Val逐窗台账](Val_window_anomaly_ledger.csv)：原ID、标签、状态、N1～N4、两种异常分数／标记及可用性。条件化缺失分数保留空值，不擅自补0或判正常。
- [共同人口配对指标](paired_Val_metrics.csv)：两种方法均在相同32165窗上比较，正常15883、故障16282；包含总体、纯故障补充、六类、域、状态、状态×域／故障、重点Hover、正常来源、四种迁移及样本量标记。
- [无条件全32381窗补充](unconditional_all_Val_supplement.json)：不同覆盖分母独立命名，不混入主比较。
- [案例异常率](Val_case_rates.csv)、[逐维标记解释](per_feature_flag_counts.csv)：逐维数据仅解释联合规则，不作为候选筛选或另一个被挑选的检验方案。
- 比较表的`adequate_descriptive_cell`采用事先固定的最低窗／案例／association group数；False表示该格不足以验证正常—故障区分。正常单类来源行没有故障样本，不计算其ROC AUC。所有计数仍保留。

## 核验与图

[执行日志](execution.log)、[完成回执](completion.json)、[独立核验](independent_verification.json)、[3项测试](pytest_conditioned_residuals.log)。独立按参考上下界核对32381个窗口，32组配对计数和案例等权AUC复算通过；Val状态与此前因果规则的全部对应结果一致。

[单张三面板图](figures/conditional_reference_tradeoffs.png)分别展示正常误报、故障异常标记率及案例等权AUC。不连接不同分类状态，不画伪置信区间，保留零起点／完整AUC轴及缺失覆盖说明；PNG300dpi、SVG、灰度版归档。程序布局／字体自检通过；当前图像附件未向代理展示，未冒称人工视觉复核通过。

## 执行边界

没有服务器、GPU、网络模型前向、训练、新模型设计、Test／Real、阈值／分组搜索或额外研究方向。C0基线、历史数据、N公式和实验配置未修改。病例、窗口重叠和SIL/HIL配对仍有关联，案例／association-group等权只是描述性敏感性汇总，不是独立重复试验或显著性认证。

新增文件、源输入身份和发布范围见[publication_manifest.json](publication_manifest.json)。原始缓存、ULog、权重、预测及既有统计资产不在本次新增提交中。报告结论只针对这一固定方案，不证明全部条件化方法不可能有效。

复算需在新目录中先保存本协议和SHA锁，再执行：

```bash
python scripts/research/validate_m3_conditioned_residuals.py --output <新的协议目录>
python -m pytest -q tests/test_m3_conditioned_residuals.py
```

入口拒绝覆盖已有参考或结果，不自动重试或追加方案。
