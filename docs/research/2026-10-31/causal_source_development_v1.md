# 新版因果数据源开发配置、采样权重与选模规则已冻结

日期：2026-10-31。[返回项目进程](../../../项目进程.md)。按用户要求，数据加工之后不再扩大数据审计；本轮完成源开发配置、实际权重和微型训练链路检查。**正式有限重训尚未开始：本地及既有autodl_UAV均无可用GPU，未自行开卡／付费或CPU冒充正式结果。**

> **后续执行已完成：**用户取得GPU并授权继续后，已按Linux路径兼容执行修订完成均匀20轮／正常来源均衡22轮正式训练、checkpoint复载、源val预测重算及本地回收，见[正式结果](causal_source_finite_retrain_v1.md)。本报告“无GPU／正式未开始”仅描述准备阶段，当前不再阻塞。原冻结协议仍保留，数据／模型／训练参数未更改，test／Real没有评估。

## 1. 已冻结的实验

配置`configs/experiments/causal_source_development_v1.json`及同名.json.sha256锁；证据`artifacts/causal_source_development_v1_20261031/`，执行前设计在design.json。

- 新数据版本rflymad_causal_publication_replay_v1，源train162069窗口／2570个有合格窗口案例，源val34710／552案例；旧分母train168800／val35520及排除记录保留。
- 不改架构：现有OnlineKGMSTCA（如实称关系特征增强MSTCA，不宣称已实现知识图谱），26通道、hidden64、1层、dropout0.2、156491参数。两个对照从相同seed42随机初始化，不载旧模型。
- **只注册2次正式运行**：uniform_replacement与balanced_source_normal，都是seed42，每次最多30epoch。不是自动6任务、不是挑最好seed、不是临时试多个模型。更多种子需另行决策，不作为本阶段前置。
- 两次共同：AdamW lr3e-4、wd1e-4、batch512、FocalLoss gamma2且alpha=None、clip1、无增强／混合精度；warmup3、固定30轮cosine到1e-6。
- 新数据输入／统计／旧索引、采样helper／模型／loss／lr／读取器及训练脚本均锁hash，不改旧六任务协议或关闭其检查。

## 2. 实际采样权重

依据**新版合格train**重算，未使用val／最终test／Real拟合：原正常案例正常21145、故障案例未激活正常61906、故障末端79018。旧3680历史后缀已排除，不能把旧权重照搬回来。

|来源|新版窗口数|uniform权重|balanced权重|
|---|---:|---:|---:|
|原正常案例|21145|1|1.963844880586427|
|故障案例的未激活窗口|61906|1|0.6707831227990825|
|故障末端|79018|1|1|

balanced仅把正常的两个来源池分成同等概率质量，**不是改二分类比例**。两臂预期正常概率51.2442%、故障48.7558%；balanced两个正常池各25.6221%。uniform相应为原正常13.0469%、故障案例未激活正常38.1973%。

两臂均WeightedRandomSampler replacement=true，每epoch162069次抽取；有限实际抽样数量可波动，必须逐epoch记录来源计数。不补抽被排除窗口、不按标签强行均衡。不再叠加类权重：loss alpha=None，相当各类权重1；避免同时改采样和类损失，破坏有限对照解释。

实际162069维权重与原缓存索引保存在preflight/sampling_weights.npz，采样概率表和分组隔离确认在preflight.json。故障目录内未激活窗口包括注入前／解除后情况，不将其全误称必然是故障前。

## 3. 选模与报告事前固定

- 每轮使用完整34710源val窗口、原顺序、不重采样或加权；argmax／阈值0.5固定，不扫阈值。
- checkpoint按**整体源val Macro-F1严格最大值**；精确相等时保留最早epoch。任何严格改进都保存，不因小于0.002就忽略最佳模型。
- 早停独立用已有selection_step：min10、patience8、significant_delta0.002，只从min_epoch后计数；与最佳checkpoint严格改善分开，最多30epoch。
- 必报整体、正常来源、正常各工况、原正常各工况含hover、故障案例类别。分层误报只作解释，不事后换选模准则。
- 两臂结果都报告，单seed只是源开发证据，不认证统计显著性、不作为真实域成绩。内部test／Real数据不读、不用作选模或阈值来源。
- 不和旧插值模型直接比较后宣称单一因素提升；数据／覆盖／统计都变过。不给任何臂自动部署资格。

## 4. 已执行检查与未执行工作

新版源专用入口`scripts/research/train_causal_source_v1.py`复用已有采样、分层指标、lr和选模实现，读取仅train／val版本化合格样本，索引／标签／normal_origin／仿真配对隔离检查通过；当前数据hash和代码hash匹配。读取build_manifest包含完成元数据，不打开test／Real tensor或ledger。

两臂CPU微型链路检查都完成：每臂2epoch×128draws、batch32、8次优化器更新；每个合格val案例取2窗，共1104窗口，只验证流程，不是正式训练成绩。初始化digest完全一致、loss／梯度有限、参数确实更新、最佳checkpoint复载结果一致。**本轮确有微型优化器更新，不能再写“从未训练”；但正式完整源重训未运行，也不发布CPU-check指标作性能。**

本地torch2.8.0+cpu、远端/root/miniconda3/bin/python torch2.5.1+cu124均cuda_available=false、device_count=0。只做SSH读取计算环境，没有改远端数据／安装环境／开卡。新入口在没有CUDA时先明确报BLOCKED_NO_CUDA，不加载训练数据、不创建正式run目录；CUDA恢复后仍需满足固定torch2.5.1+cu124／numpy2.1.3／CUDA12.4环境，不能静默迁移数值环境。

一次运行旧2026-10-30六任务协议的回归时，旧trainer hash pin拒绝现有已加安全门禁的train_online_detector文件。该项保留为**旧协议不兼容／拒绝**，不篡改旧pin或宣传全历史测试通过；新协议已锁当前代码，针对新版检查通过。见legacy_protocol_check.json。

## 5. 启动与复现

CPU检查已输出cpu_check_uniform／cpu_check_balanced及paired_cpu_verification.json，不需要为了“再确认”重复运行。冻结配置的sha、preflight、两臂初始／best／last权重、history、抽样计数和报告均留档。没有测试集成绩、没有部署替换。

正式GPU可用且满足环境后，在完整冻结代码和源数据位置运行：

```bash
bash scripts/experiments/run_causal_source_v1.sh DATA_ROOT NEW_RUN_ROOT
```

脚本只顺序执行两臂，拒绝已有run根目录；不分配GPU，不拷入旧测试数据，不自动扩大种子。只需要源train／val缓存、源metadata、norm／contract／build完成元数据和对应SHA文件，不需要上传test／Real缓存进行开发。

当前完成的是**固定配置／实际采样权重／选模规则＋可执行微型检查**。下一步唯一正式执行阻塞是可用CUDA计算环境；无需重复原始数据审计、重新生成标签或换模型试分数。
