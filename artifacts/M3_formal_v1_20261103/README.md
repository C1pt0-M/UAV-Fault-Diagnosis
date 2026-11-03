# M3正式环境复现与停止回执 v1

**当前状态：Adapter精确初始化身份门禁失败，八组正式训练未启动，优化器更新0次。**已确认的总体／六类／SIL与HIL总体Recall允许损失均0个百分点，记录在本批authorization.json；不改写原预审协议的历史状态。

完整分析：[M3_server_reproduction_failure_v1.md](../../docs/research/2026-11-03/M3_server_reproduction_failure_v1.md)。正式服务器别名`autodl_UAV3`，独立工作目录`/root/autodl-tmp/M3_formal_v1_20261103`。原M0／M1／M2资产未改，未访问Test／Real。

|证据／日志|内容|
|---|---|
|[authorization.json](authorization.json)|用户确认的8组有限任务、0个百分点硬约束、域子类只报告和失败停止规则|
|[environment_reproduction.json](environment_reproduction.json)|首包源码字节SHA失败，0作业|
|[environment_reproduction.log](environment_reproduction.log)|首包错误堆栈|
|[packaging_diagnosis.json](packaging_diagnosis.json)|Git LF与冻结本地CRLF差异；原SHA未修改|
|[environment_bundle_manifest.json](environment_bundle_manifest.json)|首包文件身份，包本身仅本地／服务器|
|[environment_bundle_exact_bytes_manifest.json](environment_bundle_exact_bytes_manifest.json)|按原SHA完整打包的文件身份，首包不覆盖|
|[environment_reproduction_exact_bytes.json](environment_reproduction_exact_bytes.json)|完整原字节包源码通过后，Adapter身份失败回执|
|[environment_reproduction_exact_bytes.log](environment_reproduction_exact_bytes.log)|实际Adapter错误堆栈|
|[read_only_diagnosis/diagnostic.json](read_only_diagnosis/diagnostic.json)|失败后只读诊断：M0精确、七Adapter互相一致但不匹配本地、八组100轮采样原ID精确一致；明确不授予训练资格|
|[read_only_diagnosis.log](read_only_diagnosis.log)|只读诊断结束日志|
|[adapter_difference.json](adapter_difference.json)|六Adapter张量逐项比较，734个参数坐标不同，max1.49e−8|
|[stopped_before_training.json](stopped_before_training.json)|远端runs与formal启动锁均不存在|
|[pytest_environment_gate.log](pytest_environment_gate.log)|新增与既有M3门禁9项测试通过；无优化器更新|
|[delivery_manifest.json](delivery_manifest.json)|本批交付文件及本地二进制SHA，当前停止点|

二进制仅本地／服务器保留：environment_bundle.tar.gz、environment_bundle_exact_bytes.tar.gz、read_only_diagnosis/adapter_untrained_diagnostic.npz。后者是只读差异诊断用未训练张量，未用作正式anchor或变更初值。Git发布JSON／日志／报告／脚本／测试及清单，不发布上述二进制。

新增脚本为`reproduce_m3_environment.py`和`diagnose_m3_environment_difference.py`。前者严格失败退出并保留失败回执；后者只为定位已发生的失败，永远不产生训练PASS或优化器。没有创建M3训练入口，也没有将未完成的训练／验证／连续回放集成检查冒称通过。

继续推进需要显式决定如何固定跨环境公共Adapter初始化，然后保持原特征／结构／预算／0个百分点约束，重新通过全部门禁。本批不自动修改初始化、降低SHA精度、接受allclose或注册新seed。
