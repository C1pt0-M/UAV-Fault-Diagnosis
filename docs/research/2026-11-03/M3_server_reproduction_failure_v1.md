# M3正式环境复现失败记录 v1

**M3在服务器初始化门禁被阻塞，八组正式训练均未启动，优化器更新为0。**M0公共未训练初始化精确复现、八组100轮抽样ID均一致；13槽Adapter与本地冻结初始化存在约1e−8量级的逐位差异，未满足冻结协议的精确身份要求。按用户“失败先停、不得绕过检查”的指令，未继续训练入口验收或正式实验。

本记录对应用户后续授权的M3下一阶段，不改写此前[预审报告](M3_preflight_review_v1.md)或[M3_preflight_v1](../../../configs/experiments/M3_preflight_v1.json)。本次执行与失败证据见[独立资产目录](../../../artifacts/M3_formal_v1_20261103/README.md)。

## 已确认并登记的接受标准

- 本批C0为参照，总体及GPS、Accelerometer、Barometer、Gyroscope、Magnetometer、Motor Recall允许下降均为0个百分点。
- SIL和HIL各自总体Recall也是0个百分点硬约束。
- 六类故障×SIL/HIL全部报告，暂不增加这些交叉子类的硬门槛。
- 沿冻结数据、共同原ID、归一化、模型、seed42、Balanced采样和原训练配置，固定0.5，以总体Val Macro-F1严格选best；不按召回换epoch，不使用Test／Real选模。
- 正式授权仅C0、Ccap、CO、CO9、CV、CR、CN、CON八组，且须全部门禁通过；不包含追加、重跑、改seed或放宽规则。

这些新决定独立保存在[authorization.json](../../../artifacts/M3_formal_v1_20261103/authorization.json)。原预审协议的历史`null`字段和SHA保持不动；本轮没有因为获准训练就把未通过的门禁标为完成。

## 正式服务器与复现结果

使用既有服务器别名`autodl_UAV3`，独立目录`/root/autodl-tmp/M3_formal_v1_20261103`，Python`/root/miniconda3/bin/python`。环境为PyTorch2.5.1+cu124、NumPy2.1.3、CUDA12.4，与冻结正式环境一致，CUDA可用。复现和诊断实际都在CPU运行，没有启动GPU训练。

数据以只读方式引用既有源Train导出及metadata，本轮未改写M0／M1／M2目录。初始化、采样诊断没有读取Val进行模型评价，也没有读取Test／Real。

|检查|结果|证据与边界|
|---|---|---|
|正式软件版本|通过|2.5.1+cu124／2.1.3／CUDA12.4|
|冻结源码SHA|原始字节包通过|首个Git归档包有换行／漏文件问题，详见下节；不修改pin|
|原M0随机未训练anchor|通过|seed42重新构造完整历史模型，全部state张量逐位一致|
|纯M0公共映射|通过|41个encoder／head张量及BN buffers，digest `a68551099cba00575f1eec6edc31ccf47b39ca8e0fa204ae2fe065d3400be646`|
|八组模型参数／公共权重|通过|C0 105938，其余109538；每组公共state精确相同|
|七组Adapter相互一致|通过|正式环境内七组均生成相同digest|
|Adapter与本地冻结身份|**失败**|正式环境digest与预审记录不同，正式门禁立即异常退出|
|初始eval输出与C0|通过，仅只读诊断|末投影零初始化使八组logits精确一致；不抵消Conv初值差异|
|源Train ID／标签／采样权重|通过，仅只读诊断|核对冻结导出与ledger SHA、原ID／标签和采样权重|
|八组100轮抽样原ID|通过，仅只读诊断|800组epoch摘要逐项与本地冻结记录相同|
|M3训练／验证／连续回放入口验收|未进入|按先决环境失败停止，未标记通过|
|正式训练与结果|0作业、0更新|没有runs目录或formal_execution_started.lock；没有正式best／last／每轮history|

正式失败回执为[environment_reproduction_exact_bytes.json](../../../artifacts/M3_formal_v1_20261103/environment_reproduction_exact_bytes.json)。失败后的[read_only_diagnosis/diagnostic.json](../../../artifacts/M3_formal_v1_20261103/read_only_diagnosis/diagnostic.json)明确标记`DIAGNOSTIC_ONLY`和`no_training_clearance=true`，不会被当作PASS门禁。

## Adapter差异的精确范围

本地冻结环境是PyTorch2.8.0+cpu／NumPy2.5.1，正式环境是2.5.1+cu124／2.1.3。相同源码、seed42、模块构造顺序，在本地重复生成仍匹配原冻结digest；正式环境生成另一digest：

- 本地：`1edd07581437d47f930040d325d6a22ee34a9a28b095d9c8c40703ca134cebe2`
- 正式服务器：`5b3f69c1f04497b529e61f03ac670ae85bb5eab81dd98c9223cce8d650c6486b`

|Adapter张量|总坐标数|不同坐标数|最大绝对差|
|---|---:|---:|---:|
|首Conv weight (16×13×3)|624|312|1.4901161193847656e−8|
|首Conv bias (16)|16|9|1.4901161193847656e−8|
|第二Conv weight (16×16×3)|768|400|1.4901161193847656e−8|
|第二Conv bias (16)|16|13|7.450580596923828e−9|
|末Linear weight (128×16)|2048|0|0|
|末Linear bias (128)|128|0|0|

合计734/3600参数坐标不同。例：首Conv[0,0,0]本地0.11565929651260376、服务器0.11565931141376495。逐张量SHA与最大差位置保存在[adapter_difference.json](../../../artifacts/M3_formal_v1_20261103/adapter_difference.json)。远端原始诊断张量NPZ仅本地／服务器保存，不推入Git，清单记录SHA。

这种差异符合跨PyTorch版本／平台初始化数值实现差异的表现；本轮没有进一步证明具体底层算子的来源，不能仅凭小量级就称无影响。末投影为0使初始输出相同，但后续末投影的梯度依赖前层特征，因此不能宣称学习轨迹仍逐位等价。没有将精确比较改为allclose，没有替换Adapter权重或将服务器digest注册成新合格起点。

## 首个打包失败与原字节恢复

第一次使用`git archive`打包时，仓库中的LF版`model_contract.py`与协议冻结的本地CRLF字节不同；还有三个SHA依赖的数据脚本未包含在最初选择路径。第一门禁停在model_contract SHA，未进入初始化，更没有训练。

该打包错误已明确记录在[packaging_diagnosis.json](../../../artifacts/M3_formal_v1_20261103/packaging_diagnosis.json)。后续仅将符合原SHA的本地原字节和完整依赖打包，解压至新的`code_exact_bytes/`目录；首包、失败日志和原解压目录均保留。没有修改源码、换行或历史pin来凑通过。原字节包随后通过源码门禁，才暴露本次Adapter实际初始化差异。两次环境检查均是0次优化器更新，不能计为训练重跑。

## 当前停止点与后续决策

本轮已完成服务器复现、失败定位、无训练的补充采样诊断和证据归档，**没有完成M3训练／验证／回放入口、入口集成测试或八组正式结果**。没有将未进入的工作冒称完成，也没有训练结果可供召回／FPR／连续事件分析。

恢复推进需要先明确精确初始化合同：一种方向是将已冻结的本地未训练Adapter保存为固定公共anchor，正式环境严格加载同一张量，再验收八组；另一种方向是明确另版登记正式环境重新生成的公共初值。两者都需要记录合同变更及重新通过门禁，本轮没有自行选择或实施。原模型／特征／优化配置不因该问题被重新设计，原0个百分点召回约束也不放宽。

Git归档本次授权记录、门禁脚本、必要测试、失败回执、采样诊断、差异表、日志索引和报告。大源码包及诊断张量保持本地／服务器，仅列SHA；原权重、日志和缓存未改写。推送这一失败记录不表示批准修改初始化或继续训练。
