# MSTCA 基线（M0）结构与流水线审查 v1

|问题编号|结论|严重程度|证据等级|
|---|---|---|---|
|A1 形状、参数、MAC|无问题：已核对，105938 参数，6764928 Conv/Linear MAC/窗|仅需记录|代码事实＋实测|
|A2 末端帧与第二次池化|需关注：丢弃的是第二池化输入位置 74，不是原始最新帧；最新帧仍影响输出|仅需记录|代码事实＋实测|
|A3 对称 padding、边界|需关注：局部有效上下文变少，有边界位置线索；不能等同完全丢帧|仅需记录|代码事实＋实测＋推断|
|A4 SE/GAP 与位置|需关注：没有显式位置编码，仍非严格平移不变；短故障受全窗平均稀释|仅需记录|代码事实＋实测＋推断|
|A5 确定性冗余通道|无问题：已核对，无结构正确性错误；冗余不等于新增独立信息|仅需记录|代码事实＋推断|
|A6 BN、Dropout、B=1|无问题：已核对，验证/推理 eval，Dropout 仅 head，B=1 可用|仅需记录|代码事实＋实测|
|B7 初始化映射与去 KG|有问题：当前映射正确，但直接按 seed 构造不是旧锚点；新消融须统一初始化契约|影响公平消融|代码事实＋实测|
|B8 可复现性|需关注：本机 CPU 重复逐位一致；跨版本重新初始化不同，GPU 逐位复现未核实|影响公平消融|代码事实＋实测|
|C9 标准化与训练/在线一致性|有问题：源统计/列序、单窗张量已核对；通用在线路径不等于冻结因果输入路径|影响公平消融|代码事实＋实测|
|C10 高度逐窗相对化|有问题：训练与 predict 单窗一致；predict_records_batch 对整段减首值，后续窗口不一致|影响公平消融|代码事实＋实测|
|C11 mask/-1 与 BN|无问题：已核对，源 compact 数据在前向前排除未知/不合格窗，抽样 batch 合法|仅需记录|代码事实＋实测|
|C12 FocalLoss|无问题：已核对，gamma=2 正确；gamma=0 的值及梯度与 CE 一致|仅需记录|代码事实＋实测|
|C13 Balanced 采样|无问题：已核对，两正常来源各占正常质量一半；10 万次经验比例吻合|仅需记录|代码事实＋实测|
|C14 选点、早停、公平性|有问题：实现符合冻结规则，但 val 兼做选点和评估，不能称独立确认集|影响公平消融|代码事实＋实测＋推断|
|C15 预测规则|需关注：实际为 softmax 后 argmax；并列正常，数学等价 p_fault>0.5；极近 logits 可舍入成并列|仅需记录|代码事实＋实测|
|D16 测试覆盖|有问题：已有初始化/数据门禁测试较好，缺少重要结构和 26 维在线批单一致性回归|影响公平消融|代码事实；选定子集实测|

**结论：没有发现足以判定已有 M0/M2/M3/ENS 结果无效的证据。M0 的形状、规模、损失、采样和源数据准入链路通过本次检查。应先冻结初始化与选点评估规则，再做结构消融；不能把通用 FaultDetector 的批量路径直接当作已通过审查的 26 维因果回放路径。**

严重程度沿用任务书三档。“影响公平消融”也包括训练/回放契约不一致带来的比较风险，不表示已证明污染历史实验。历史结果影响只按本次检查到的代码路径判断；未重读历史预测、未重算历史性能。

## 1. 范围、证据和复现

### 1.1 执行边界

- 未做 GPU 训练、CPU 优化器训练或任何参数更新；只做 CPU 前向/反向，优化步数为 **0**。
- 未读取内部 test split 或任何 Real 数据；没有运行会打开这些数据的入口。
- 只访问源 train/val 缓存及其 ledgers、`input_contract.json`、`norm_stats.json`。用 mmap 各取 **64 个特征窗口，共 128 窗**；为了采样质量和准入核对，读取源 train/val 完整标签、原 ID 和源 metadata。没有扫描全部 X，也没有重新计算全量缓存 SHA 或全量归一化统计。
- 为避免生产 Dataset 构造器遍历全部 X，抽取后在本任务 artifacts 中创建 64 窗 train/val 子集，再运行真正的 `CausalCacheDataset.__init__`、`__getitem__` 和 DataLoader。该子集是审查夹具，不是新的正式数据版本。
- 只加载冻结的**未训练**初始化锚点。在线 parity 使用新建的未训练 MSTCA 夹具；`runtime_fixture/best.pt` 只是满足加载器文件名要求，**不是训练得到的 best checkpoint**。
- 没有修改已有源码、配置、协议、数据或权重，没有 commit/push。任务开始即存在的 `scripts/research/case_relative_features_dev.py` 修改及两份 `tmp_control_compensation_audit*.log` 未触碰。
- Python 命令使用 `-I -B`；禁写 pyc，禁用 CUDA 可见设备。主检查另有数据文件白名单和源数据写入拒绝钩子。

### 1.2 文件与执行命令

脚本：

1. `scripts/research/audit_mstca/audit_baseline.py`
2. `scripts/research/audit_mstca/check_edge_cases.py`

从仓库根目录执行：

```text
python -I -B scripts/research/audit_mstca/audit_baseline.py
python -I -B scripts/research/audit_mstca/check_edge_cases.py
```

脚本拒绝覆盖已经存在的目标输出；重跑应使用新的审查输出目录，不要删除或覆盖旧实验。主检查正式取数输出为：

- `artifacts/mstca_baseline_audit_v1_20261106/checks_v2/results.json`，下文简称 **R**。
- 同目录 `source_evidence.json`：源码 SHA、AST 符号起止行、全部 12 项冻结代码 pin 的吻合结果。
- `artifacts/mstca_baseline_audit_v1_20261106/edge_cases.json`，下文简称 **E**。
- `artifacts/mstca_baseline_audit_v1_20261106/pytest.txt`：**22 passed in 4.02s**。

根目录的第一轮 `results.json`/`console.txt` 是保留的探索输出：其绝对高度合成样本没有同步重算 `acc_norm`，故混入了通道不一致差异。本文只引用修正合成样本后的 **checks_v2**，不引用第一轮该比较值。

测试执行选取：`tests/test_causal_mstca_m0.py` 全部用例；`tests/test_causal_cache.py` 中 compact source、incomplete build、portable path、traversal 四组用例（含参数化）。使用 `pytest.main`，显式加入包/研究脚本目录，`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`、`-p no:cacheprovider`，临时文件限定在本任务 artifacts 的 `pytest_tmp/`。**没有运行涉及 test/Real fixture 的用例**；对这些用例仅阅读源码来判断覆盖范围。

**实测环境：**Python 3.12.4、PyTorch 2.8.0+cpu、NumPy 2.5.1、pandas 2.3.1、4 CPU threads。不是正式训练要求的 PyTorch 2.5.1+cu124 / NumPy 2.1.3 / CUDA 12.4 环境。

下文路径简写仅为排版：`lstm.py` 指 `packages/uav_fault/uav_fault/models/cnn/lstm.py`；`causal_cache.py` 指 `packages/uav_fault/uav_fault/data/causal_cache.py`；`fault_detector.py` 指 `packages/uav_fault/uav_fault/online/fault_detector.py`；研究脚本位于 `scripts/research/`。每条代码结论均给出文件和行号。

## 2. 问题清单：按严重程度排序

### P1. val 同时用于选点、停止和报告；现有组 CV 也不是完全 train 内选点（C14）

**严重程度：影响公平消融。**

- **代码事实：**`train_causal_mstca_m0.py:168-171,195-200` 每轮跑完整源 val，并以其 Macro-F1 调用 `selection_step`；`train_source_normal_sampling.py:136-148` 分离“任意严格提升保存 best”和“超过 0.002 的显著提升重置耐心”。`train_m0_groupcv.py:82-84,103-106` 的折模型同样用外部源 val 选 checkpoint，之后才预测 train held-out 折（`:111-118`）。
- **复现：**查看上述调用；R `selection_probe` 显示 0.800→0.801 会保存 best，但仍累计 bad_epochs，0.803 才重置显著提升参考。规则执行正确，问题在评估解释，而不是早停代码算错。
- **影响范围：**M0、M2、M3 的源 val 最优分数是**开发集选点后的分数**；不能把差异完全归因于结构。ENS 使用折内 held-out train 窗口不等于所有神经模型选择都没有使用外部 val。已有数值并不因此自动无效，但“val 从未参与开发”“纯 train 内独立 CV”等表述不成立。没有检查 test/Real 结果，也不据此判断其有效性。
- **建议设计：**新消融固定相同轮数与更新预算；或在外层 train 内按 association_group 再划 inner-selection，以 inner 分数选点，源 val 仅在结构/阈值冻结后确认一次。优缺点见第 5.6 节。旧结果保留原解释，不事后换选点规则重挑“更好”的旧 epoch。

### P2. 初始化依赖 KG 构造顺序；同 seed、跨版本都不足以复现锚点（B7/B8）

**严重程度：影响公平消融。**

- **代码事实：**`train_causal_mstca_m0.py:62-102` 实际以归档未训练状态为真值，而非简单构造 KG 再复制：只有正式环境匹配时才重建 KG 作锚点来源复核，然后加载归档公共状态。`lstm.py:321-346` 的构造顺序是 raw encoder → relation encoder → projection → gate → head，关系模块消耗 head 之前的 RNG。
- **实测/复现：**R `initialization`。归档映射 digest 为 `a68551099cba00575f1eec6edc31ccf47b39ca8e0fa204ae2fe065d3400be646`；本机直接 seed42 MSTCA 为 `727d8c8785d008c8c8283816dc12f946287a7502b2e1e08e728f0281ddb2b16b`。本机 fresh KG 映射为 `210983a58a5d1077b356dd0a8837693c2daa7e853e8bc5e44d39b98d97e91c53`。
- **重要区分：**本机直接 MSTCA 对本机 fresh KG 映射，**只差 head weight/bias**；本机直接 MSTCA 对旧归档，则还差多个 Conv weight/bias。不能把跨环境的全部差异都归咎于 KG 抽随机数。具体版本内部初始化差异原因**未核实**。
- **影响范围：**当前 M0 两采样臂映射正确，M3 公共锚点映射见 `train_causal_m3.py:55-67`，本次未发现初始化错载。M2 与 M0 只是公共 raw/head 一致，整模型不相同。ENS 其他 seed 的生成仍调用 KG（`train_m0_groupcv.py:28-42`）。若后续去 SE/改单尺度时各自直接构造，同名公共层也可能随 RNG 顺序改变，破坏受控比较。
- **建议设计：**把纯 M0 的 41 个未训练公共状态提取成独立、不可变、带 SHA/digest 的纯模型锚点；新结构先构造再显式加载公共张量，形状改变的模块使用独立命名子 seed。不要靠构造无用 KG 或硬编码“烧掉”随机数维持 RNG。完整接口见第 7 节。

### P3. 通用批量回放的高度基线错用整段起点（C9/C10）

**严重程度：影响公平消融；若某结果使用该 26 维路径，则对应回放比较需重核，不能直接认可。**

- **代码事实：**`fault_detector.py:255-270` 减传入 records 的首帧高度；`:342-352` 单窗 predict 先取最新 150 帧；但 `:306-325` 批量方法先对整段 records 建矩阵和相对化，再切窗。应是 `h[t]-h[window_start]`，实际后续窗为 `h[t]-h[file_start]`。
- **复现：**主脚本生成 170 帧、起始高度 1000、每帧升高 0.02，使用同一个未训练 26 维模型比较 batch 与逐窗。R `online_edge_cases.batch_vs_window`：end=150 输入差 0；end=160 输入差 **0.02862662 z**；end=170 为 **0.05725324 z**，后者最大 logit 差 **0.0003610**。批大小造成的约 1e-9 logit 误差不是此处输入差的原因。
- **影响范围：**源码调用者包括 `scripts/eval/eval_telemetry_live.py:240-250` 和 `scripts/eval/eval_rfly_domain_holdout.py:76-89`。本次**没有运行**这些入口或读取它们的数据。已查 M0/M2 训练验证走 Dataset；M3 回放使用逐窗 `canonical_window`（`replay_causal_m3.py:45-48`）；ENS 源 val 回放显式逐窗减起点（`replay_ens_events_val.py:34-44`）。故没有证据说此 bug 使这些已查路径的既有结果无效。其他历史脚本/运行快照是否调用该错误路径，**未核实**。旧 31 维绝对高度资产没有 `baro_alt_relative`，不会触发这一相对化错误。
- **建议设计：**可以整段缓存原始特征，但必须切窗后再做依赖窗口起点的变换，再标准化；最好复用严格 `canonical_window`。新增随时间变化高度的 26 维 batch-vs-single 张量和 logit 回归，覆盖不止第一个窗口。不要修改旧实验预测来“静默修复”。

### P4. 通用 FaultDetector 不是冻结因果输入合同的等价实现（C9/C11）

**严重程度：影响公平消融（离线与在线/回放对比）；不据此认定源训练存在污染。**

- **代码事实：**`fault_detector.py:156-176,253,272` 对缺失/非有限的非稀疏必需字段返回 0，最终 `nan_to_num`；没有接收源合同的全窗可用 mask/话题年龄。冻结路径 `packages/uav_fault/uav_fault/causal_input.py:84-120` 则要求有效 bool mask、必需通道和有限值，不合格窗口拒绝前向。通用路径还优先相信已提供的 `acc_norm`，严格路径会从 accel 重算（分别 `fault_detector.py:162-171` 与 `causal_input.py:101-106`）。
- **复现：**R `online_edge_cases.missing_required_channel_silently_filled`：空 row 的 `gyro_x` 得到 0，而非 unavailable；这是合成输入检查，没有读取不合格的实际证据窗口。
- **影响范围：**源 M0/M2/M3 的 strict Dataset/metadata 门禁与此不同，不受这一通用推理补零行为直接影响。ENS 接入通用在线加载器前也必须统一输入合同。已有部署或其他回放有没有独立外层门禁、是否全都在前向之前执行，**未全量核实**；不能仅凭一个 canonical val 窗口 parity 通过就宣称原始在线链路通过。
- **建议设计：**冻结研究模型的 runtime 输入必须携带可用性/新鲜度并在建 batch 前拒绝整窗；只对合同明确允许的稀疏字段做因果补齐。发布资产标识严格 causal contract，不让旧兼容补零路径自动承接新研究模型。

### P5. 关键回归缺口（D16）

**严重程度：影响公平消融。**

- **代码事实/复现：**第 6 节按现有用例逐项映射。尤其 `tests/test_fault_detector.py:17-66` 使用 31 维、恒定高度，仅断言 schedule/返回格式，没有断言第二个及以后窗口的 batch/single 数值一致，因此无法发现 P3。
- **影响范围：**不说明 M0/M2/M3/ENS 已错；说明删除 SE、改单尺度或迁移在线时缺少自动防退化保障。现有 22 项选定测试通过不能替代缺失的语义回归。
- **建议设计：**在新实验实现时加入结构规模、公共初始化不被删层 RNG 影响、严格源窗口准入、26 维批单高度相对化一致性等最小回归。本任务只新增审查脚本，不改已有测试。

### P6. 最新帧的局部贡献较弱、时间位置表达较弱（A2–A4）

**严重程度：仅需记录；属于结构取舍，不是已证实的代码错误。**

- **复现：**R `tail_probe`、`gradient`、`step_probe`，E `permanent_steps`，数值见第 3 节。
- **影响范围：**M0/M2 raw encoder、M3 的 M0 骨干、ENS 神经成员共享这一类结构限制，但**已训练权重上的实际检测延迟/召回损失未核实**。不能用未训练合成探针宣称既有误报或漏检由池化导致。
- **建议设计：**先保留冻结基线。若之后另设“最新帧保留/末端聚合”实验，必须登记为独立结构变更，不能夹带到去 SE/单尺度消融；同步报告 MAC、参数、位置探针和延迟，不自动使用 ceil_mode 替换旧结果。

### P7. 绝对高度减法的 float32 舍入（C10）

**严重程度：仅需记录。**

- **代码事实：**严格 canonical 先 float64 减窗首值、再 cast float32（`causal_input.py:101-120`）；通用 detector 先填 float32 矩阵再减（`fault_detector.py:259-268`）。
- **复现：**R `online_edge_cases`，1000 m 绝对高度合成单窗最大差 **4.2021e-6 z**。已相对化 val 缓存窗口比较差为 0。
- **影响范围：**没有证据影响 M0/M2/M3/ENS 历史性能；更大高度偏置可能放大取消误差是**推断**，未做分布评估。
- **建议设计：**与 P3 一并统一到严格窗口变换的 dtype 顺序，不单独为这个小量级误差改变冻结基线。

### P8. 实际预测先 softmax，极近 logits 并不逐位等价于 logits.argmax（C15）

**严重程度：仅需记录。**

- **代码事实：**`train_causal_source_v1.py:123-132` 和 `fault_detector.py:283-304` 都对 softmax 概率做 argmax。
- **复现：**E `near_tie`：float32 `[0,1e-8]` 的 logits argmax=1，softmax 为 `[0.5,0.5]` 后 argmax=0；精确 `[0,0]` 两者均判正常。
- **影响范围：**M0/M2/M3 的已查评估实现是同一种概率 argmax，未发现不公平；历史预测是否出现这类近并列，**未核实**。不据此推断有性能变化。
- **建议设计：**新协议精确写成“softmax 后 argmax、概率并列取 normal，等价严格 p_fault>0.5”，或全臂统一改为 logit margin>0 并另冻结规则；不要一臂改、另一臂不改，也不要宣称二者有限精度下完全等价。

## 3. A：结构逐项核对

### A1. 张量、参数量与 MAC

**代码事实：**`lstm.py:169-231,240-257`。`num_layers` 在 MSTCA 构造中不控制网络层数；当前配置传 1，没有行为错误，但未来不得把改这个字段当作深度消融。

**实测：**R `structure`，单窗 B=1，Conv/Linear hooks：

|层/模块|输出（不含 B）|参数（含 BN affine，不含 BN buffer）|Conv/Linear MAC|
|---|---|---:|---:|
|输入|26×150|—|—|
|k3 Conv+BN+ReLU|24×150|1944|280800|
|k7 Conv+BN+ReLU|24×150|4440|655200|
|k15 Conv+BN+ReLU|16×150|6288|936000|
|concat → SE（64→16→64）|64×150|2128|2048|
|MaxPool(2)|64×75|0|0|
|Conv k5 + BN + ReLU|128×75|41344|3072000|
|MaxPool(2)|128×37|0|0|
|Conv k3 + BN + ReLU|128×37|49536|1818624|
|AdaptiveAvgPool(1) → squeeze|128|0|0|
|Dropout(0.2) → Linear|2|258|256|
|合计|2|**105938**|**6764928**|

已核对，无问题。MAC 按一个乘加记 1、仅计 Conv1d/Linear、每窗一次前向；不计 BN、SE 广播乘法、激活、池化、归约及 softmax。不能把这个数字当全算子 FLOPs；只把 Conv/Linear 乘加按 2 FLOPs 计是 13529856 FLOPs。

### A2. 第二次池化与最新帧

**代码事实：**`lstm.py:210-215`，第二 MaxPool 的默认 floor 使 75→37，位置 74 不作为池化输出候选。它前面还有 k5 卷积，原始最新帧也能经首层卷积、SE、邻接位置继续传播。因此“最新两帧被完全丢弃”不成立。

**实测：**未训练归档 M0、eval、32 个固定种子 N(0,1) 合成输入。同一个 `0.1×N(0,1)` 扰动模板分别放开头/末尾 k 帧，比较平均输出 logit L2 变化：

|k|开头扰动|末尾扰动|末尾/开头|
|---:|---:|---:|---:|
|1|2.94839e-5|1.95854e-5|0.6643|
|3|5.56206e-5|5.00613e-5|0.9000|
|5|7.55155e-5|6.34025e-5|0.8396|
|10|1.14767e-4|1.23858e-4|1.0792|

R `pool2_probe` 独立验证被丢弃的池化直接输入末位置梯度为 0；完整网络对原始末帧则非零。margin `logit_fault-logit_normal` 的绝对梯度按通道求和、32 样本取均值：首帧 **0.00138489**，中帧 t=75 **0.00233200**，末帧 **0.00105630**，末/中约 0.453。对比既受池化相位也受 padding/卷积/SE 影响，不能把所有差异都归到第二池化。

**推断：**最新少量帧的影响没有被设计成优先保留。实测只证明结构通路和当前随机锚点的敏感性，未核实已训练模型或物理故障的延迟。

### A3. padding 与感受野

**代码事实：**三个首层 k3/k7/k15 使用对称 1/3/7 零填充，block2/3 分别 2/1（`lstm.py:195-221`）。标准化输入首层的 0 是训练通道均值；**深层 padding 的 0 是隐特征零，不再是物理通道均值**。

**推断（由层参数计算）：**忽略 SE 的全局依赖、BN 在 eval 为逐点仿射，只看局部卷积/池化路径，最终 37 格在 GAP 前的名义 RF 分别为 **22、26、34 原始帧**，步距为 4。最长 k15 路第 j 格名义覆盖 `[4j-15,4j+18]`：中部可用真实上下文完整；首格与末格部分落在窗外，实际真实上下文较少。例如末格 j=36 的名义范围 `[129,162]`，只有 `[129,149]` 在窗内。

SE 的全时均值使严格的函数依赖可能遍及全窗；GAP 也汇总全部格。因此不能把 34 写成模型完整输出的全局 RF 上限。训练态 BN 还有批内/时间归约依赖，本段局部计算特指 eval 主路径。

**实测：**A2 梯度边缘低于中部，A4 等长脉冲不同位置输出不同。它们支持边界/位置效应存在，但没有单独隔离 padding 的因果贡献。**是否构成有害的已学习边界伪影未核实。**所有输入都是截至窗口末端的历史，窗口内对称卷积不等于读取当前时刻之后的数据。

### A4. 位置敏感性与阶跃

**代码事实：**SE 先 `x.mean(dim=2)`（`lstm.py:180-183`），末端使用 AdaptiveAvgPool（`:217-231`），无显式时间位置参数。但卷积仍识别局部序列形状，padding 和 stride 相位打破严格平移不变性；不能描述成“完全不知道故障在哪段”。

**实测：**主探针用同幅同长 30 帧的 +1 z 矩形阶跃/脉冲，26 通道同步变化，置于全零背景。这样移动位置时不改变总信号质量，避免把时长差误当位置差：

|位置（0-based，右端不含）|fault probability|相对全零 logit L2|
|---|---:|---:|
|[0,30)|0.4943457|0.0081191|
|[60,90)|0.4946638|0.0095340|
|[120,150)|0.4942847|0.0072927|

另补真正“从发生处持续到窗尾”的单位阶跃（E）：onset=0/75/140，持续 150/75/10 帧，fault probability 为 **0.4982609 / 0.4957884 / 0.4936568**。后者同时改变持续长度，不能单独用于证明位置敏感性。

**推断：**全局均值压缩对故障时长/出现频度敏感，对绝对时间位置表达较弱；在不触及边界、移动量为总步距整数倍时更接近平移不变。没有直接“最近帧权重大”机制。这里的全通道阶跃不是符合四元数/acc_norm 约束的物理故障，不能当作故障检出率实验。

### A5. acc_norm 和四元数冗余

**代码事实：**Base26 同时含 accel_xyz、acc_norm、att_q_wxyz（`packages/uav_fault/uav_fault/model_contract.py:43-74`），strict transform 重建 `sqrt(ax²+ay²+az²)`（`causal_input.py:101-105`）。

**推断：**已核对，无结构正确性问题。CNN 不要求通道线性独立；非线性 norm 可提供有用表征，但不增加独立传感器信息，可能重复加权加速度。有效单位四元数存在 3 个自由度而不是 4，逐通道 z-score 不要求标准化后仍保持单位范数。q 与 −q 的等价符号、上游四元数单位性/符号连续性是否被全部保证，本次**未核实**；不应凭冗余直接删通道。后续通道消融应保持相同预处理且重新训练，不能把它混进骨干结构消融。

### A6. BN/Dropout/逐窗在线

**代码事实：**M0 每 epoch 调 `.train()`（`train_causal_mstca_m0.py:180`），验证 helper 调 `.eval()`+no_grad（`train_causal_source_v1.py:123-132`）；在线构造器 `.eval()`（`fault_detector.py:135-137`），推理 no_grad（`:283-291`）。M0 唯一 Dropout 是 `head.0`（`lstm.py:251-254`）。

**实测：**R `batchnorm`：eval B=1 有限；同一样本 B=1 对 B=32 输出最大差 **7.45e-9**，是数值路径差异，不是 BN 在 eval 依赖其他样本。训练态 B=1 也可前向/反向，因为 BatchNorm1d 在每通道上还汇总时间长度 150/75/37，均不为 1；head/GAP 后没有 BN。**已核对，无问题**。若调用者后续手工把 detector.model 切为 train，不能再保证推理合同，这不属于当前正常调用路径。

## 4. B：初始化与可复现性

### B7. 精确映射

**代码事实：**`train_causal_mstca_m0.py:62-102`：

1. 核对归档 `initial.pt` 的文件 SHA 与全部状态 digest。
2. 用 `torch.random.fork_rng(devices=[])` 包住构造；设置 seed。正式版本匹配时重建 KG 并逐 tensor 检查归档确为新鲜随机状态；本机不匹配，所以该步骤本次**未核实**，receipt 明确为 false。
3. 对目标模型全部 state key：`encoder.* ← raw_encoder.*`，`head.* ← head.*`；校验 shape/dtype，clone，strict load，核对映射后的冻结 digest。
4. 映射 **41 个状态张量**：首三路和后两块 Conv/BN 的权重、偏置、BN running_mean/running_var/num_batches_tracked，SE 两个 Linear 的权重/偏置，head Linear 权重/偏置。Dropout、Pool、ReLU 无状态。
5. 丢弃 relation_encoder 的全部 **39** 状态张量、relation_proj 的 **2** 个及 gate 的 **4** 个；完整 key 清单在 R。没有执行 relation forward，也没有加载训练后的最佳权重。

**实测：**文件 SHA `b666b8a28129319f0b58e1c62ab7fa0e8698d61ec6aa32243824cda243f3e956`、归档状态 digest `7772ccc1a20ce8b6d97c7ce83fc4927a8ff9a5dc7bef17038ca767806e561c55` 与配置一致。两次映射得到的状态/前向逐位一致。当前映射本身**已核对，无问题**；与后续直接构造不等价的问题见 P2。

`state_digest` 按 key 排序，将 key+dtype+shape 和 contiguous CPU tensor bytes 哈希（`train_source_normal_sampling.py:127-133`）。它包含 BN buffers，不只是 trainable parameters；文件 SHA 与 tensor digest 不是同一概念。

### B8. 确定性边界

**实测：**R：同 seed 本机直接构造两次，digest 相等、相同 CPU eval 前向逐位相等；同一归档映射两次前向也逐位相等。跨版本 fresh 初始化则与归档不同，见 P2。没有做 GPU 运行，不能声称 GPU 位级验证通过。

**代码事实：**`train_causal_mstca_m0.py:136-145,164-171` 和 `train_causal_source_v1.py:21`：

- deterministic algorithms=True；cuDNN benchmark=False、deterministic=True。
- cuDNN TF32=True，matmul TF32=False；设置 NumPy、CPU torch、CUDA seeds。
- 导入前 `CUBLAS_WORKSPACE_CONFIG=:4096:8` 使用 setdefault；若外部预先设置了别值，这里不会强制覆盖。正式 M0 入口未显式像旧 helper 那样复核该环境变量（对比 `train_source_normal_sampling.py:49-57`），需后续协议显式记录，不在本次运行 GPU 验证。
- WeightedRandomSampler 独立 generator=seed+11；DataLoader generator=seed+101；num_workers=0。当前无 worker 多进程/随机增强因素。`build_model` 的 fork_rng 恢复外部 CPU RNG，避免环境条件分支改变训练 Dropout 起始 RNG。

**推断：**确定性设置不承诺跨 GPU 架构、驱动、cuDNN/PyTorch 版本逐位相同；TF32 本身不等于随机非确定性，但会改变算术精度/轨迹。以后增加 workers 或随机 transform 时，需显式处理 NumPy/Python worker RNG、采样顺序和 resume 状态。现有 checkpoint 不保存 optimizer/RNG，不能称无缝断点恢复（`train_source_normal_sampling.py:1-4`）。

## 5. C：数据和训练流水线

### C9. 标准化、列序和同窗 parity

**代码事实：**构建统计使用每个合格 source train window 的全部 150 帧、float64 parallel central moments、population std，重叠帧重复贡献（`scripts/data/build_full_causal_input.py:330-349,359-360`）。`norm_stats.json` 声明 fit_split=`source_train_only`、fit_windows=162069、fit_frame_contributions=24310350。**本次没有全量重算统计**；来源结论由构建代码、版本统计及 hash 一致性支持。

**代码事实：**Dataset 核对 contract/列序、训练统计来源和 identity，取窗只做 float32 z-score（`causal_cache.py:49-57,100-116`）；训练入口锁定 Base26 顺序（`train_causal_mstca_m0.py:38-39`）。FaultDetector 从 stats.columns 建矩阵并校验 schema（`fault_detector.py:92-116`；`model_contract.py:187-237`），没有硬编码为 31 维。

**实测：**配置 `data.columns`、输入 contract、实际 norm stats、`RFLY_SEQ_COLUMNS_V2` **26 列完全一致**；统计与合同的实际 SHA 匹配冻结配置。取 val compact_index=0 的同一未归一化窗口，Dataset 路径对比实际 FaultDetector 构造/建矩阵/单窗推理：**最大输入差 0，最大 logit 差 0**。两边使用同一个归档未训练模型状态，非训练最佳权重。这验证的是“已经 canonical 化的同一缓存窗口”，不是原始 ULog 到在线接收链路的全链复现。

通用加载器已支持 26 维，不需要强行套用旧 31 维资产。另一方面，M0 训练入口只复制 protocol/norm/trainer，未生成 detector 所需的顶层 `config.json`（`train_causal_mstca_m0.py:158-161`）；训练 run 不能未经显式打包直接视为可部署目录。本次用新夹具配置验证加载器，没有修改或加载现有部署资产。

严格因果运行时与通用加载器差异见 P3/P4。canonical val 同窗 parity 已核对无问题，但不能把 C9 整项写成“在线全链路已通过”。

### C10. baro_alt_relative

**代码事实：**构建原始窗口调用 `canonical_window`（`build_full_causal_input.py:162-168`），先减窗首高度，再写未归一化 raw float32 缓存；Dataset 不应再拿绝对高度减一次（`causal_cache.py:114-116`）。

**实测：**抽取 train/val 各 64 窗，raw 高度首帧绝对值最大均为 **0**。通用单窗 predict 对已相对化窗口再减首值不改变输入，因此 C9 数值完全一致。绝对高度入口的数学变换一致，但 dtype 次序不同，见 P7。多窗口 batch 入口不一致，见 P3。**单窗已核对无问题，批量路径有问题。**

### C11. 准入与未知标签

**代码事实：**源 train/val 是已 compact 的合格集合；构建器按 `supervised_valid` 对原 ID 留存（`build_full_causal_input.py:365-385`）。Dataset 对源 split 不再应用 evidence masks，但要求所选标签全部为 int64 0/1，并检查归一化有限值（`causal_cache.py:80-108`）。`train_causal_source_v1.py:59-84` 进一步核对全量源 metadata 的 input_valid/label_valid/supervised_valid 全为真、标签/ID 顺序、域和角色；M0 在 `model(x)` 前再检查有限值和二分类标签（`train_causal_mstca_m0.py:182-186`）。

**实测：**全量源 metadata 全合格，标签和原 ID 对缓存匹配，train/val case_uid 和 association_group 无交叉。两批各 **32×26×150**，标签仅 0/1，X 有限，R `source_batch` 保留抽窗 ID 和 batch labels。**已核对，无问题。**

evidence split 的 mask/-1 代码分支见 `causal_cache.py:83-99`，只做静态审查，没有读取任何实际 evidence 文件，也没有运行相应 fixture。通用 FaultDetector 本身不实施同一严格准入，单列 P4，不能混同为源训练 BN 被 -1 或 NaN 窗污染。

### C12. FocalLoss

**代码事实：**`packages/uav_fault/uav_fault/models/cnn/focal_loss.py:31-56` 实现 `-(1-p_t)^gamma log(p_t)`，可选 alpha，默认 None，默认 mean；M0 调用只传 gamma，实际 alpha=None（`train_causal_mstca_m0.py:172`）。概率 clamp 不替换原始 log_softmax 的 log(p_t)，避免低概率时截断 log loss。

**实测：**19 条随机二分类 logits，R `loss`：gamma=0 对 CE 的值差 **0**、梯度最大差 **0**；gamma=2 对直接公式值差 **0**。**已核对，无问题。**float32 的 `1-1e-8` 可舍入为 1，完全易样本可得零 focal 权重，符合其低权重行为；未扩展测试不在本协议中使用的加权/非法 reduction 组合。

### C13. Balanced 采样

**代码事实：**`smoke_normal_source_sampling.py:32-57`，令 N_original=21145，N_prefault=61906，N_normal=83051，N_fault=79018：

- original normal 单窗权重 `83051/(2×21145)`；prefault normal 单窗权重 `83051/(2×61906)`；fault 单窗权重 1。
- 两池质量各 41525.5，故条件于 normal 各一半，**不是全体样本中各 50%**。总质量 162069 与 uniform 不变。
- replacement=True，每 epoch 抽 len(train)=162069，独立 sampler generator（`train_causal_mstca_m0.py:164-167`）。这不是强制每个 batch 精确均衡。

**实测：**按实际源 metadata 重建 origins；两臂各抽 100000 次，只采索引、不加载相应特征 X：

|arm|original normal|prefault normal|fault|normal 内 original 比例|
|---|---:|---:|---:|---:|
|uniform|12908|38108|48984|25.3019%|
|balanced|25643|25442|48915|**50.1967%**|

balanced 理论总体比例为 **25.6221% / 25.6221% / 48.7558%**；经验故障占比 48.915%，吻合抽样波动。**已核对，无问题。**

### C14. 选点和早停的替代设计

**代码事实：**完整 val=34710，不抽样、不重加权；严格更大 Macro-F1 存 best，精确并列保留早 epoch。patience=20、min_epochs=30、显著提升阈值=0.002；保存 best 与重置耐心是两回事（`train_source_normal_sampling.py:136-148`；`train_causal_mstca_m0.py:168-175,195-201`）。实现已核对，无 off-by-one 错误证据；val 重用风险见 P1。

**设计/推断：**

|替代方案|优点|代价/限制|
|---|---|---|
|预先固定轮数/更新数，全部臂用末轮|训练曝光可比，不以 val 为结构试错反馈；最容易解释去 SE/单尺度|不同结构收敛速度不同；共同轮数应由 train 内先导决定，不能从源 val 反挑|
|train 内按 association_group 留 inner-selection|可早停，源 val 留作一次确认；case/pair 不交叉|可训练数据减少，选点仍有噪声；留组人口和阈值预算必须所有臂一致|
|嵌套组 CV：outer 留组评估，inner 选点|分开调参/选点/评估，最适合结构选择|算力更高，需预注册聚合规则，不能看完 outer 再换判据|
|继续沿用旧完整 val 规则|历史衔接简单，所有臂选择算法一致|只能称 source-development 比较；相同上限≠相同实际训练预算，最优 val 有选择乐观偏差|

建议后续结构选择优先使用 train 内方案；保留一臂按原合同复现历史 M0，但不要把旧 M0 best 与改过选点规则的新结构直接当作受控结构差。

### C15. 预测

**代码事实/实测：**源 evaluator 和 FaultDetector 均 softmax→argmax；标签顺序 normal=0/fault=1，概率精确并列取第一个。R `prediction` 对 `[0,0]`、`[-1,1]`、`[1,-1]` 得 `[0,1,0]`，与严格 `p_fault>0.5` 相同，没有阈值搜索。**常规规则已核对无问题**；协议文字中的“logits argmax”在有限精度极近 logits 时有差别，见 P8。事件状态机的 onset/clear 阈值不是这里的窗口 argmax 规则，本任务不重新审计事件政策。

## 6. D16：现有测试逐项覆盖

下表“未覆盖”指任务指定的两个主要测试文件，必要时补看 `tests/test_fault_detector.py`，不是声称整个仓库绝无其他相关测试。

|审查项|已有覆盖及证据|缺口/本次补充|
|---|---|---|
|A1|`tests/test_causal_mstca_m0.py:83-98` 参数量、模块列表、输出形状/有限值|未逐层 shape/MAC；本次 hooks 补测|
|A2/A3|未覆盖|未测末帧/边界/池化相位；本次扰动、梯度、pool probe 补测|
|A4|未覆盖|未测位置移动；本次等长脉冲和永久阶跃补测|
|A5|未覆盖|无冗余通道/四元数上游保证检查；本次只作结构判断|
|A6|上述 M0 测试调用 eval，做 B=2 零输入|未明确测 B=1 train/eval、Dropout 位置；本次补测|
|B7|`test_causal_mstca_m0.py:83-128` 精确映射、外部 RNG 保持、两臂同锚点、改锚拒绝、模拟匹配环境重建|没有去 SE/单尺度改构造后公共初始化不变测试；本次对真实归档和直接构造做 digest 比较|
|B8|`:115-120` 重复映射状态相等|没有重复 CPU 前向逐位检查及跨版本说明；本次补测，GPU 未核实|
|C9|M0 `:46-56,71-80` schema/数据与代码 pin 拒绝绕过；cache `:34-38` 基础 source reader|原 fixture mean=0/std=1，无法充分检测标准化数值错误；没有真实同窗 26 维在线 parity；本次补测|
|C10|指定两个文件未覆盖；`test_fault_detector.py:17-66` 只用旧31列恒定高度|没有26维、变化高度、后续窗批单数值回归；P3 正是此缺口|
|C11|`test_causal_cache.py:41-62` 合成 evidence mask、NaN 行排除、角色/用途限制；`:90-93` tamper 拒绝|这些 evidence 用例本次只静态阅读，未执行；没在真实源 batch/BN 调用前核对；本次源 batch 补测|
|C12|两个指定文件未覆盖|本次 gamma=0 值/梯度、gamma=2 公式补测|
|C13|M0 只锁配置，不检验采样经验分布|本次实际源标签/metadata 10万抽样补测|
|C14|M0 `:46-56,66-68` 锁定预算/early_stop 配置|未检验严格/显著改善分离和 val 独立性；本次 selection 合成轨迹＋调用链静态审查|
|C15|未直接断言概率并列决策|本次常规及近并列实测|
|执行边界|M0 `:131-159` 未授权/no CUDA/错环境/CPU training 拒绝；cache `:65-87` incomplete/迁移路径/traversal|本次选定22用例通过，不能替代全部回归或 GPU 验收|

**结论：**初始化身份和数据安全门禁的现有测试有实际价值；结构语义、online parity、统计采样和选点解释仍依赖本次审查脚本，不应把“22 passed”解释为所有 16 项已有常驻回归。

## 7. 后续消融的统一模型构造与初始化接口设计（只设计，不改代码）

### 7.1 最小接口与范围

**设计：**仅覆盖这次拟做的四臂，避免重造多模型平台：

```python
model = build_baseline(variant, in_channels=26, num_classes=2, dropout=0.2)
receipt = initialize_baseline(model, seed, pure_anchor, policy="shared_anchor_v1")
```

- `variant` 明确限定 `mstca`、`mstca_no_se`、`mstca_single_k7`、`cnn1d_matched`。数据合同、损失、采样、轮数/选点政策独立冻结，不藏在模型工厂内。
- 完整 M0 必须复用现有 `OnlineMSTCA`；去 SE 只移除 SE；单尺度 k7 维持 24/24/16 三路通道与 concat/后续块，避免同时更改通道数和分支数。若真正改成一个 64 通道卷积，应另立版本，不能冒充同一实验。
- CNN1D 的参数匹配只定义架构搜索约束：由整数通道宽度静态计算参数量，事先固定最接近 105938 的候选、容差、实际参数和 MAC；**具体宽度未在本次搜索，未核实能否精确匹配**。不为凑数添加不参与前向的参数。优先保持 encoder 输出 128，使 head 可公共复用；做不到时必须披露 head 形状变化。

### 7.2 不依赖 KG、同时精确保留现有 M0 锚点

**设计：**

1. 一次性离线转换归档：仅按当前验证通过的 `raw_encoder→encoder` / head 映射提取 41 个未训练状态，以新的纯 M0 文件保存，不编辑原文件。生产初始化只依赖纯 M0 状态，不 import/实例化 KG，也不执行无用 relation forward。
2. 转换记录保存原 SHA、原 state digest、映射表、新文件 SHA、纯 state digest、来源环境。纯 state digest 必须仍为 `a68551099cba00575f1eec6edc31ccf47b39ca8e0fa204ae2fe065d3400be646`。旧 seed42 历史 M0 的复现靠这份精确 tensor，不靠本地重新抽 RNG。
3. 构造模型使用隔离 RNG，之后按语义模块白名单加载公共状态；**同语义、同 shape、同 dtype 才共享**。删除 SE 不得让后续 Conv/head 的起点漂移。BN affine 和运行 buffers 一起共享，包含 num_batches_tracked。
4. 形状不同/新增模块使用 `hash(policy_version, seed, semantic_module_name, shape)` 派生的稳定子 seed，不能用 Python 的进程随机 hash。每模块局部 generator/隔离 RNG、明确 initializer 和 dtype；Conv/Linear 采用明确 fan-in 一致的既定默认分布，BN 默认 weight=1/bias=0/running_mean=0/running_var=1/counter=0。不对普通串联 Conv 一律零初始化。
5. 不静默做中心裁剪、补零或重缩放来把 k3/k15 权重塞到 k7；这会额外改变初始化统计。单尺度臂仅可精确复用原 k7 路及相同后续块/SE/head，改变 kernel shape 的两路用同一新模块规则初始化。CNN 只共享确有同语义/形状的块，不能因为名字/shape 碰巧一致就自动认领。
6. 对每个结构报告 `copied_keys / initialized_keys / removed_keys / incompatible_keys`、shape、参数/MAC、完整 state digest 和公共子集 digest；默认遇到未登记不兼容就拒绝运行。完整 M0 应严格加载全部状态，去 SE 只允许少对应 SE keys。

不同形状的网络不可能“全模型每个数相同”。公平的“同起点”应定义为：**公共部分逐位相同，变更部分遵循预注册且互不影响 RNG 的同一初始化规则**，而不是声称全部模型 digest 相同。

### 7.3 其他 seeds 与配对随机性

**设计：**旧归档只证明 seed42。新 seed 没有纯历史锚点时，用独立命名子 seed 生成该 seed 的纯 M0，保存为该 seed 的 anchor bank，再供其所有消融臂共享；seed42 的历史精确锚点作为明确登记的兼容条目。不能拿 seed42 的锚点冒充 seed43，也不能声称新 policy 生成的其他 seed 与旧 KG 生成轨迹相同。

如要所有 seeds 连“生成规则”完全统一，可另开 `module_seed_v1` 实验，**同时重跑 M0 对照**；历史 seed42 精确复现保留在兼容模式，不能无声明混合两套政策。训练采样 generator、DataLoader generator、训练 Dropout RNG 与模型构造 RNG 分离，逐 seed/折固定采样 ID 流；删层不应改变采样或公共层初始化。相同规则不保证不同网络产生相同优化轨迹，应报告实际轮数/更新数。

### 7.4 执行前最小验收

**设计：**在任何后续训练前，CPU 上至少验证：

- 完整 M0 参数/MAC/输出 shape 仍匹配，seed42 精确重现纯锚点 digest。
- 去 SE 不改变其余公共 tensors；单尺度/CNN 的变更 key 清单与登记一致。
- 同 seed 同结构重复构造/前向一致，模型构造不改变外部 RNG；源采样 ID 流一致。
- 实际源 val 小窗口的 Dataset 与严格逐窗 runtime 输入一致；26 维变化高度的 batch/single 多窗口输入一致。
- 非法/未知/过期输入先拒绝，再前向；B=1 eval 正常。
- 在新协议中明确选点和评估人口、按组隔离、浮点预测并列规则；不读取 test/Real 来决定结构或上述规则。

以上均为后续实现设计；本次没有改造模型、初始化入口、加载器、训练政策或已有结果。
