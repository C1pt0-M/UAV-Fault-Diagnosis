# BN训练／推理模式窄对照：不足以解释主要误报

日期：2026-11-01 UTC。用户同意先做一个窄检查；若无明显模式差异，停止BN试探并恢复原训练预算。服务器仍可用，未开新付费实例。

## 结论

固定混合源train批次，关闭Dropout，只切换BN使用保存running统计或当前batch统计。两臂hover误报只下降0.82／1.40个百分点，仍很高，整体纯正常误报反而略升。**本次没有发现足以解释主要悬停失败的巨大BN模式差异，不把BN认定为唯一根因，也不宣称BN完全无影响。**按既定条件停止该诊断，进入原预算恢复的两臂训练，不继续扫BN系数或改部署统计。

|同批hover诊断|均匀模型|正常来源均衡模型|
|---|---:|---:|
|hover抽样次数（唯一窗口）|368（350）|714（642）|
|保存running统计误报|193／368（52.45%）|229／714（32.07%）|
|当前batch统计误报|190／368（51.63%）|219／714（30.67%）|
|batch减running|−0.82个百分点|−1.40个百分点|

原正常全部误报均匀403／2080→409／2080，均衡525／4243→536／4243；故障召回均匀99.38%→99.34%，均衡97.27%→97.04%。这些只属于训练抽样诊断，不是完整源val成绩，更不是实时或Real新结果。两个模型抽样分布不同，不能直接跨臂比较表内抽样率作受控性能结论。

## 固定方法

- 每臂严格取注册replacement sampler第一epoch的前32个完整512混合批，共16384次抽样。种子53＝原采样generator seed42＋11。索引事前保存，不按预测挑批。两臂唯一窗口15570／15464。
- 每对running／batch模式使用完全相同归一化张量与顺序，不用normal-only批估计BN，禁止Dropout／反向／optimizer。数据仅源train，未调用会额外打开val的prepare。
- 原模型eval；batch模型为deepcopy，仅BatchNorm1d training=true、track_running_stats=false，其余模块eval。现有buffers不写、BN counter不增。原模型与副本推理前后全部state_digest一致，checkpoint文件SHA不变，未保存替代权重。
- running模式与前次全train-normal预测按compact／原索引对齐，浮点最大差均匀3.6954879760742188e-06、均衡1.3113021850585938e-06，均小于事前1e-5数值容差，正常分类变化0；batch上下文改变导致浮点核小差异不等同数据身份改变。
- batch输出依赖混合故障／正常上下文，不能作为在线推理方案。重复抽样窗口不是独立重复试验，未扫描阈值或新模型。

## 完成和后续

脚本`scripts/research/diagnose_causal_bn_modes.py`；最小3项检查验证BN只用batch矩、Dropout关闭、状态／梯度不改，连同前次4项共7项先验测试通过。

产物`artifacts/causal_bn_mode_diagnosis_v1_20261101/`：plan.json、results/summary.json、两臂paired_predictions.npz、逐批／逐案例信息、independent_verification.json。9成员逐hash核验，归档1002498字节、SHA256 `d736e90cb9de6f35c685a2bad03d0a91d933e1fa7c79842ff1d161ad12081592`。本地重算混淆矩阵、hover分母／唯一窗口及误报变化一致。

用户已授权的条件满足后，在独立`artifacts/causal_source_longbudget_v1_20261101/`冻结新配置：原Linux执行协议为基准，仅恢复max100／warmup5／cosine100／min30／patience20；其余源数据、统计、版本化读取器、模型代码、loss、采样、seed42／初始化、总体Macro-F1选模与0.5阈值不变。两臂从头训练，不把旧短预算权重续到100，不追加第三臂或种子。训练结果另报，不预先承诺一定改善。
