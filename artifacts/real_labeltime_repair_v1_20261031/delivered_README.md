# Real标签时间修订v1：实际数据已生成，含显式待核隔离

As of 2026-10-31。版本：`rflymad_real_labeltime_v1`。

## 已交付

- 保持原465条角色、官方裁剪、50Hz时间轴、连续输入与cap64旧索引。原始资料、旧CSV、旧缓存、源修复版及模型均不覆盖。
- 456条原bag相关命令／同步记录完整验证，修复副本位于`cases/`；9条相关记录索引／头不一致的案例保留在`quarantine/`，所有旧字段字符串不变，标签参考明确未知。隔离是保守待核，不表示每个窗口输入都坏了。
- 在可验证的456条中，1032个逐时刻二分类状态与旧加工不同。按原ROS发送消息的时间保持恢复离散ID，不插值ID、不改成ULog接收时刻；未覆盖末行ID为-1并置无效，不能视作正常。
- `real_all_roles_evidence.pt`保留原29319个窗口，X与旧缓存逐元素相同；28544个窗口末端可判定、775个末端暂不参加监督评估。10个可判定末端实际改标；另1个候选变化因两源不一致而保留未知。
- 775个无效末端：9条待核案例576个窗口＋其余199个ROS／ULog状态不一致边界。完整保留旧分母、旧标签和输入，不补抽、不将剔除后变化说成模型提升。

|角色|案例|原窗口|可判定末端|待核末端|
|---|---:|---:|---:|---:|
|adapt_normal|10|640|640|0|
|calibrate_normal|5|320|320|0|
|test_normal|33|2112|1984|128|
|test_fault|392|25088|24441|647|
|supplemental_normal_test|25|1159|1159|0|

10适配／5校准的输入、标签、角色和旧索引全部不变。test_normal中2案例待核，不擅自改为其他角色；主测试不与适配、校准或补充测试混算。

## 文件与读取约定

- `repair_manifest.json`：完成状态、来源身份、输出哈希、角色ID映射与限制。
- `case_manifest.csv`、`case_repair_results.jsonl`：新旧案例路径、哈希及隔离状态。
- `cases/`、`quarantine/`：实际CSV副本。新增`labeltime_state_reference`、`labeltime_supervision_valid`、`labeltime_input_clock_review`。最后一个标记在本版本表示原bag相关记录需要复核，不等于已证明整例惯性时钟倒退。
- `references/`：每案例完整原时间轴、历史状态、ROS命令ID／状态、ULog状态、覆盖与逐行有效性；不删帧。
- `real_cap64_window_links.csv`：按原缓存行序的角色、窗口边界、新旧标签与资格；`level2_label`只表示修订合格标签或uncertain，旧值在`historical_level2_label`。
- `evaluation_ineligible_window_ledger.csv`：775个待核窗口的原索引／原因，非删除名单。
- `stride10_replay_window_links.csv`：保留196142个原连续回放窗口及末端／整窗资格；5677个末端暂不合格。整窗资格只作描述，不是新调出来的删除阈值。
- `norm_stats.json`：历史源train统计原字节复制，不在Real重拟合。

`real_all_roles_evidence.pt`是角色分离证据包，**不是直接送入旧训练器的train.pt／test.pt**，故意没有通用`y`键：

|键|含义|
|---|---|
|x|原29319×26×150输入，float32，未归一化，逐元素不变|
|historical_y|旧0／1标签，不应当作已修监督|
|ros_reference_y|可验证ROS时间保持参考；未知为-1|
|qualified_reference_y|两来源覆盖且状态一致的参考；其余为-1|
|endpoint_label_valid|末端监督／分类评估资格，bool|
|whole_window_label_valid|全部150帧标签资格，仅作另列描述|
|role_id|adapt_normal=0、calibrate_normal=1、test_normal=2、test_fault=3、supplemental_normal_test=4|
|source_cache_index|完整原缓存索引0—29318|

后续使用必须先明确选角色，再按资格读取目标；不能把-1送入现有FocalLoss，也不能先前向再只遮损失。Real故障角色始终不用于监督训练。分类评估应报告角色原分母与无效数，而不是只报剩余子集高分。连续回放始终保留全时间轴；事件／延迟评分还需要逐行mask感知的评估器，**本版本没有执行或认证新事件评分**。

旧布尔CSV标签入口遇到共享labeltime标记会拒绝读取，避免把未知行误当正常。`mode`、`controls`等其他历史命令字段原样保留，不是已修监督；采用本版本参考和清单。不要向本目录补旧test.pt或复制旧协议来绕过检查。

## 仍未解决／未执行

9条原bag相关消息记录不一致仍待恢复／确认；未宣称所有Real已完全修复。ROS发送、ULog接收是两种消息事件，不是独立测得的物理损伤真值。保留原首同步锚点，不认证精确物理起点、因果在线输入或实时延迟。原输入插值和其他输入质量限制仍在。

本轮未训练、未推理模型、未看概率、未调阈值、未更新历史性能或部署。旧Real分数是旧标签／覆盖口径；不能直接与新覆盖比较归因模型改善。未修内部SIL／HIL test、31通道部署／离线数据，也未扩入原493条中角色外的28条。

## 核验与复现

仓库入口：`scripts/research/verify_real_label_bags.py`、`scripts/data/repair_real_labeltime.py`、`scripts/research/verify_real_label_repair.py`。原bag读取依赖隔离工具目录`D:/tmp/real_labeltime_tools_20261031`中的rosbags 0.11.9，未修改项目依赖或全局包；代码将其追加至sys.path，避免覆盖既有NumPy／Torch。

产物记录：`D:/UAV-Fault-Diagnosis/artifacts/real_labeltime_repair_v1_20261031/`。标准bag Reader第二遍验证456条参考并独立复现9条失败；全部465份CSV逐字段、行时间／状态／mask与缓存索引独立核验；全部X、角色、原缓存、原bag与归一化身份复核通过。86项相关回归测试通过。

构建首次执行到399条时因240秒工具时限中断，续跑前核验已完成文件，保存1份未登记文件到`interrupted_files/`，仅补66条。文件和脚本快照留档，未改原始资产。`CSV_COMPLETE.json`是CSV阶段历史检查点；最终状态以`repair_manifest.json`为准。
