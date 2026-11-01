# 26通道全量因果发布回放数据v1

As of 2026-10-31。版本`rflymad_causal_publication_replay_v1`。**这是4153个冻结闭集案例的全量因果回放重建，不是此前16例小规模试验。**完成状态以build_manifest.json为准；该清单未生成时不要使用。

## 处理范围与可用窗口

|范围|原窗口|输入和标签均可用|不纳入使用|案例数|完全无合格窗口案例|
|---|---:|---:|---:|---:|---:|
|源train（含历史补充）|168800|162069|6731|2580＋历史后缀|10（不含后缀）|
|源val|35520|34710|810|555|3|
|内部SIL/HIL test|35392|34802|590|553|1|
|Real原465角色|29319|26815|2504|465|10|

4153案例、15973496原网格帧全部处理并保留时间轴。24案例在原cap64选窗内完全没有合格窗口，1414案例部分窗口不合格，2715案例原选窗全部合格。不物理删文件，不按模型分数换样本，筛后不补抽。

训练6731排除包括：2936输入不合格＋115输入可用但末端标签不确定＋3680历史RflySim正常后缀。后缀缺少已证明的逐话题采样／可用时间，不能拿旧插值或快照输入冒充因果版；旧后缀X/y仍完整留在旧缓存，本版每个旧索引明确列入排除台账，不虚构案例身份。

其他范围的排除（不重复相加）：
- val：782输入不合格＋28仅标签不确定＝810。
- test：564输入不合格＋26仅标签不确定＝590。
- Real：1779输入不合格＋725仅标签不确定＝2504。9条原bag不确定案例仍维持旧标签隔离，未强行修。

每个窗口可能同时多个话题超龄或同时标签不确定，逐原因计数有交集，不能相加当总数。

## 因果输入规则

所有连续输入直接从原始ULog所需话题提取，**不再复用旧加工CSV的插值数值**。保留原CSV时间范围／20ms网格、原训练验证测试成员、仿真配对和cap64索引；不扩大裁剪、不补抽窗口。

只选原发布时间不晚于网格时刻的最后记录；明确原ULog publication timestamp仅是回放可用时间代理，不是网络物理到达认证。样本时刻优先既定timestamp_sample／惯性relative字段，异常时钟不静默排序／回退。控制80ms、输出120ms、gyro40ms、accel40ms、气压100ms、姿态80ms、速度140ms、磁60ms，沿用已冻结的小规模工程预算，未根据全库覆盖／模型分数修改。

超龄／缺失／非有限／原时钟待核的每行保留位置、原因与NaN，不补零／未来点。窗口内所有必要输入帧合格才可用；监督还要求末端标签资格通过。连续时间轴不压缩，后续事件不能跨空档硬连。

3条已知SIL惯性时钟复核案例仍排除，不因标签可读而批准。原始GTD／ULog或ROS／ULog标签来源仍沿既定修复定义，未用故障或正常类别改采样规则。消息事件不是独立物理损伤真值。

## 文件

- input_contract.json：此数据版冻结输入规则身份；RUNNING／EXECUTION_REVISION记录构建来源，最终build_manifest.json记录完成与验证身份。
- cases/*.npz：4153份完整原时间网格，canonical_rows（float64，气压高度在此阶段为绝对值）、row_input_valid、标签参考及每个观测组源索引／采样时间／可用时间／年龄／原因。
- windows/*.npz：每案例监督合格原cap64窗口，float32、已每窗减首气压、未归一化，不含任何不合格窗口。
- case_records/*.json：逐例哈希／状态／原始文件／输出身份和原因。
- ledgers/{split}_all_windows.csv：原缓存全集索引、原标签、新标签、输入／标签／联合资格、phase等；train还包含3680历史后缀。
- ledgers/{split}_included_windows.csv、{split}_excluded_windows.csv：互斥并完整覆盖旧分母，排除不等于删除原始飞行。
- cache/train.pt、val.pt：仅合格原索引的x／y紧凑缓存，带data_version、input_contract_sha256、norm_sha256及original_cache_index。
- cache/test_evidence.pt、real_evidence.pt：保留原全部索引，合格X为新因果窗口，不合格X明确NaN且目标-1；故意无通用y。完整逐行输入即使末端标签不确定仍在cases/保留。
- norm_stats.json：**重新拟合的新因果源train统计**，仅162069合格源train窗口、24310350个含重叠帧贡献，无val／test／Real／历史补充后缀。旧数据统计未覆盖，不复用小规模诊断系数。

Real角色保持：适配10例640原窗口→598合格，校准5例320→301，主正常33例2112→1884，主故障392例25088→22947，补充正常25例1159→1085。角色不互换、不补抽；旧“640适配全部可用”只适用于旧输入版本，新版必须记录覆盖变化。不能将全角色混算测试成绩。

## 安全读取

旧train_online_detector入口仍拒绝本目录，不能复制旧test.pt／norm_stats绕过。新读取器`uav_fault.data.causal_cache.CausalCacheDataset`要求期望版本、split、purpose，Real还必须一个明确role，先过滤输入／标签mask才返回归一化样本。最终build_manifest未完成或版本／哈希／角色不符就拒绝。示例：

```python
from uav_fault.data.causal_cache import CausalCacheDataset, VERSION
train = CausalCacheDataset(
    'D:/UAV-Data/causal_input_v1', expected_version=VERSION,
    split='train', purpose='source_development',
)
real_fault_test = CausalCacheDataset(
    'D:/UAV-Data/causal_input_v1', expected_version=VERSION,
    split='real', purpose='evaluation', role='test_fault',
)
```

这只是合格样本读取，不是训练启动命令。读取器报告original_population、excluded_population及original_cache_indices；输入统计按此版本读取并归一化，不把-1送旧FocalLoss，不用Real故障训练。采样／类别权重与早停仍需新实验配置明确，不自动复用旧元数据或成绩。

## 边界

本版完成已冻结4153闭集案例的因果**发布回放**输入加工＋不可用排除；不是全6605原始案例全部开发，不涉及开放集、角色外28条Real、31通道离线／旧部署。

未训练／推理模型、未调阈值、未报新性能。旧分数不能与新筛后分数直接归因比较；新年龄预算与新统计／覆盖已改变输入版本。实际实时采集链路尚缺逐话题source／receive时钟，不因回放因果PASS就自动认证部署或物理告警延迟。

9条Real原bag仍未知；不可用窗口排除不是证明整份原始输入物理损坏。完整时间轴保留用于后续带空档的回放，不跨mask做事件真值。

构建cache时曾因NPZ数组在每窗口重复解压而暂停优化，只调整I/O并保留已完成3135原case输出，未改年龄／采样／标签规则；中断、原脚本、续跑身份和日志全部留在artifacts/causal_input_full_v1_20261031。
