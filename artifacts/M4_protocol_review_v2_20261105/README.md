# M4协议审查 v2：已发布v1统计资产的补充分析

[主报告](../../docs/research/M4协议修订与无训练补充审查_v2_2026-11-05.md)；[待审批修订协议](revised_protocol_proposal.json)。本任务没有正式训练授权，未构造/加载模型、读取权重、生成h或U/C分数。

## 数据与边界

- 所有分析源数据均为相邻`M4_design_preflight_v1_20261105/`中已发布CSV/GZ/JSON。没有读取原始缓存、遥测、模型权重、Test/Real或新Val预测。
- v1所有文件、v1报告、技术总览和工作区既有修改的初始SHA见[initial_snapshot.json](initial_snapshot.json)。结束重新核对；无v1覆盖，无模型/历史代码/配置/权重修改，无commit/push。
- [analysis_plan.json](analysis_plan.json)在支持结果前登记D1/D2/D3、原三折、c尺度、精确核、组排除、代表距离、三组ESS候选、域质量下限及Q95/Q99。没有根据outer故障覆盖选择阈值。
- D1=83051已知正常；D2=21145独立正常；D3=61906故障案例未激活正常。D1为D2/D3不相交并集，连case/group身份也核对；保留原SIL/HIL配对。
- 9个fold×population任务各162069查询，共1458621条记录；每行保留原ID/case/group/role和指标，身份可连接`query_identity.csv.gz`。每个query只用对应inner-fit人口参考；inner-fit query排除整关联组，dev/outer全inner-fit参考。c尺度固定由inner-fit人口拟合，不对每个留组query再拟合尺度；因此不称完整独立留组训练验证。

## 产物索引

|产物|内容|
|---|---|
|`normal_population_support.csv`|三人口、三折三角色、六状态/六目录/域、窗口案例组、双成员/单成员和v1门槛|
|`normal_origin_phase_mass.csv` / `normal_group_case_contributions.csv`|来源阶段权重、案例及关联组贡献、每案例窗口数|
|`normal_population_distributions.csv` / `normal_state_domain_distributions.csv`|四c变量组/案例/窗等权分布；每分层内重新等权，不是全参考中的层质量|
|`condition_scalers.csv`|每fold/人口的inner-fit正常均值、标准差、floor和零方差记录|
|`case_representatives_f*_D*.csv` / `group_representatives_f*_D*.csv`|原生c单位正常代表点和案例/组身份；仅距离用，不替代核参考窗|
|`condition_support_queries_f*_D*.csv.gz`|全部原ID的Group/Case ESS、域质量/域内ESS、最近案例/组和5案例距离、各固定门禁|
|`distance_thresholds_fit_normal_LOO.csv`|仅inner-fit正常留组查询的加权Q95/Q99；不作泛化覆盖保证|
|`condition_support_distribution.csv` / `joint_gate_coverage.csv`|逐角色/标签/状态/目录/来源/域百分位和所有候选门禁覆盖；含原分母|
|`heldout_fault_domain_coverage.csv` / `heldout_normal_origin_coverage.csv`|主门禁六故障×域及独立/故障案例正常的外层覆盖|
|`condition_AUC_all_fixed_direction.csv` / `condition_case_identity_values.csv.gz`|四条件量原固定方向AUC及底层案例身份；保留无支持空值，不重算有利方向|
|`pre_post_identity_changes.csv` / `pre_post_delta_by_fault_domain*.csv`|保留原配对身份、时距、条件差值，按故障/域/路由改变描述|
|`revised_protocol_proposal.json`|D1主参考、校准B、CF-infer-only/no-refit待审方案及门禁，无执行器|
|`completion.json` / `independent_verification.json`|主计算完成和独立数据/数值核验范围|
|`independent_query_probes.csv`|直接权重/距离复算的原ID、组、角色和误差|
|`support_report_tables.md` / `support_key_findings.json`|报告数值表及机器可读摘要，来源为完整c支持结果|
|`delivery_verification.json` / `publication_manifest.json`|边界/Git/SHA/链接检查及全部新文件字节SHA（清单不哈希自身）|

ESS是条件权重的有效集中程度，不是独立飞行次数。域内ESS用域权重重新归一，另报域总质量。高ESS可同时远离正常代表点，因此联合距离门禁；代表点均值又有多模态压缩局限。门禁覆盖不是Recall/FPR，没有诊断性能或正常h支持结果。故障条件有支持不等于故障正常；无支持未来修正0、基础头回退、保留全部评价分母。

## 脚本、执行记录与复算

主完成实现为`analyze_v2_parallel.py`，既有后端Python3.12.4/NumPy1.26.4/pandas2.2.3环境，4个CPU查询线程、每线程BLAS1，CUDA不可见。无torch/model依赖，无新依赖安装。它只读取v1资产和本任务计划；每次写文件使用exclusive-create，续接时对既有CSV重新推导核对而不覆盖。

原`analyze_v2.py`字符串组比较较慢，中断于第一任务；`analyze_v2_fast.py`改一一对应整数编码，续接比较时发现原不可用AUC的空值序列化类型不同，修正比较方式后又因计算时长中断。`analyze_v2_parallel.py`只进一步并行独立查询批次，保持精确全部参考窗和数学定义。未输出完成的query表不作为结果；已有支持/代表CSV保留并核对。[performance_revision.json](performance_revision.json)、[execution_revision_parallel.json](execution_revision_parallel.json)记录过程，主脚本SHA见completion。

`supplement_population.py`只用v1 CSV补状态×域分布及完整配对汇总。`verify_and_summarize.py`读取v1与新v2查询台账，独立按直接差向量重算选定查询核/组及案例聚合/距离、核对全量ID/角色/六候选标志/正常LOO阈值、来源并集、原AUC/配对；不导入主分析函数，不运行模型。`build_support_tables.py`只将新v2结果汇总为表，不选最优门禁。

不要在本目录重跑覆盖已有证据；需要重新计算应建立独立新版本、复制已登记计划并保留原仓库相对路径。禁止运行v1原始张量统计脚本。本目录的核验也只读允许统计资产，不为复核而读取原缓存/权重。

## 审批边界

校准B只是数学选择，未来h/分数数值验证未进行。CF-infer-only是待批准简化，并非已经消除全部锚点风险或检验原保护损失；原CF若坚持训练保护，整编码器/头的Group Cross-Fitting另审。no-refit只支持保守研究探针结论。R*或纯探索定位及正式训练仍须项目负责人明确决定。本次不自动Git提交，也不启动任何正式实验。
