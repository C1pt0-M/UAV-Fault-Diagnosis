# S 组结构消融：实现与本地 CPU 冒烟 v1

## 1. 结果与执行边界

**S0/S1/S2/S3 各完成一次 seed42、fold0、两轮 CPU 冒烟，初始 digest 全部匹配 golden。新增测试 36 项通过；联合工厂测试为 76 passed / 4 skipped / 1 xfailed。**

按用户要求先执行 `git pull`，返回 `Already up to date.`。本次依据 r1：

```text
configs/experiments/s_ablation_mstca_v1.json
SHA256 = 0764635a75de22963b573180d1616b6da97e038b76565c247d12a2b0a6f61005
```

仅新增以下四个源码/测试文件、本报告及指定 artifacts：

- `scripts/research/train_s_ablation.py`
- `scripts/research/evaluate_s_ablation.py`
- `scripts/research/preflight_s_ablation.py`
- `tests/test_s_ablation.py`

没有修改协议、factory、已有测试、数据或权重；没有连接训练服务器、使用 GPU、执行正式训练或 commit/push。source val 三文件**仅以 rb 方式计算 SHA256，没有解析、加载或反序列化**；没有读取 test/Real。执行包装器用 audit hook 将这三个文件的打开限制在 `train_s_ablation.sha` 调用栈中，其他数据根路径使用精确文件白名单并拒绝写入。合成测试进程则拒绝整个 `D:/UAV-Data`。

实际源 X 每臂只读取 **512 fit + 256 inner + 256 heldout = 1024 个窗口**，并读取 train ledger 全部元数据。用 mmap 读取缓存布局后只复制选定 X/y/原 ID，未调用会加载 val 的 `source.prepare`，也未使用会全量扫描 X 的生产 Dataset 构造器。留出 X 直到优化器全部结束后才创建。

全部 Python 执行使用 `-I -B`；实际 CPU 运行显式设置 `CUDA_VISIBLE_DEVICES=''`、`CUBLAS_WORKSPACE_CONFIG=:4096:8`。本机为 Python 3.12.4 / PyTorch 2.8.0+cpu / NumPy 2.5.1，不是正式 CUDA 环境。

## 2. 协议逐条对照

下文脚本路径均位于 `scripts/research/`。最终完整行号索引：`artifacts/s_ablation_mstca_v1_smoke_20261106/implementation_map_release.json`。

|协议/任务条目|实现位置|核对结果|
|---|---|---|
|协议 SHA 锁、固定 r1 身份|`train_s_ablation.py:74-93`|锁文件与完整固定 SHA 双重核对，不能重新上锁后绕过|
|冻结代码 SHA、共享 helper SHA|同上|核对 S 协议八个文件及旧 M0 共享代码 pin，包括实际 sampler/LR/loss helper|
|新脚本执行前 manifest|`train_s_ablation.py:96-98,321-352`|四个新文件 SHA 写入 manifest；正式运行要求预先保存的 manifest 路径与 SHA；运行结束拒绝执行中源码漂移|
|九数据身份、r1 val 字节级核对|`train_s_ablation.py:101-114`|实际匹配 8 项，仅按冒烟许可跳过 `cache/train.pt` 全量 SHA；正式模式不跳过任何一项|
|association_group 字符串、外层/inner 哈希|`train_s_ablation.py:117-169`|保留前导零；分别使用 `h1cv\|` mod3、`s_ablation_inner_v1\|` mod10；完整 train metadata 的三折大小实际吻合|
|只读冒烟特征窗口|`train_s_ablation.py:172-221`|按正常来源分层、确定性抽取 512/256/256；归一化和有限值/标签/原 ID 校验在前向前|
|模型仅通过 factory 构造与初始化|`train_s_ablation.py:321-382`|检查 factory 不改变全局 CPU RNG，使用固定 module_seed_v1，断言 pinned test 中的 golden；保存 receipt|
|balanced_source_normal replacement sampler|`train_s_ablation.py:374-395`|复用已有 `sampling_weights`，在当前 fit 上重算；draws=len(fit)，seed+11；DataLoader seed+101、workers=0|
|训练损失与优化器|`train_s_ablation.py:408-438`|按 JSON 的 **FocalLoss gamma=2/alpha=None**，AdamW 3e-4/1e-4，batch512，clip1/error_if_nonfinite；不是把类平衡 CE 用作训练损失|
|warmup/cosine|同上|复用原 LR helper，warmup5、horizon100、min_lr1e-6；冒烟只截为2轮，不压缩调度|
|inner 类平衡 CE|`train_s_ablation.py:224-228,246-263`|eval；稳定 log-softmax CE 与数学 softmax/log 定义等价；float64 累积全 inner 的两类总损失/计数，不能按 batch 平均代替|
|best 严格降低、精确并列取早轮|`train_s_ablation.py:231-243`|任意严格下降保存 best；显著改善参考值只在 loss < reference×0.995 时更新|
|min30/patience20|同上|按“after min_epochs”解释为 epoch>30 开始计数；恒定损失序列在第50轮停止；冒烟最大2轮，仅验证写盘与流程|
|训练中不评估留出折|`train_s_ablation.py:354-357,438-468`|前期只构造 fit/inner；训练结束后才构造 heldout；best、last 各前向一次，原 ID 和概率写 npz|
|完整报告与新目录拒绝覆盖|`train_s_ablation.py:321-480`|逐轮 focal/inner CE、选择状态、best/last、优化步数、时间、参数/MAC、环境及 SHA；启动即拒绝已有目录|
|正式 CPU/环境/CUBLAS 拒绝|`train_s_ablation.py:274-285,321-352`|检查在可能有 setdefault 副作用的旧 sampler import 之前；无 CUDA、版本/工作区不符直接拒绝，不默默补环境|
|正式固定槽位、次序与48任务上限|`train_s_ablation.py:301-318`|正式 CV 36 个固定槽位、val12个固定槽位，依协议 seed→fold→arm 顺序要求前序 COMPLETE/同 manifest；不能换目录替换已完成 tuple；失败恢复需人工留存失败记录，不自动重跑|
|phase val 门禁|`train_s_ablation.py:288-298,321-352,438-468`|必须先提供真实 COMPLETE phase1 结论文件与 SHA；合成/不完整结论拒绝；全部 train 去 inner 训练后才载入 val，fold统一用0表示无外层折。本轮仅测试拒绝路径，未执行 val|
|完整36 run/12 val run检查|`evaluate_s_ablation.py:217-272`|正式输入只接受 COMPLETE，同协议、golden、全部数据 SHA、同 implementation；重复/乱序/缺失 ID、坏概率与预测文件 hash 拒绝；缺任一 run 时只给 INCOMPLETE，不出结论|
|FP@R97、R95、并列|`evaluate_s_ablation.py:42-75`|故障分数降序，选择达到目标累计质量的最大实际故障分数；匹配点用 >=，固定工作点用 >0.5|
|AP/分层/CE/Brier/资源/last|`evaluate_s_ablation.py:78-114,299-348`|全部所列字段实现；R97 排除 late_hover_normal 时保持 t* 不变；故障约束排除 late_hover_fault；六类、GPS/barometer、SIL/HIL 分开|
|cluster bootstrap 配对性|`evaluate_s_ablation.py:123-161`|2000次、seed20261106；每 replicate 只抽一次关联组，重复组用窗口权重表示，对全部12个 arm/seed 分数向量复用同一权重对象；每次重新计算 t*；95% percentile|
|r1 判读优先级与 AP|`evaluate_s_ablation.py:164-214`|seed一致性→TRADE_OFF→HELPS/BETTER→REMOVABLE/NO_ADVANTAGE→INCONCLUSIVE；重叠 CI 优先 HELPS 且 effect_below_margin；AP反向降级|
|历史 S0 sanity，不作受控比较|`evaluate_s_ablation.py:275-296`|通过显式历史根目录参数实现原 ID 对齐与 p>0.5 FP/TP 比较；本轮未读取历史全量预测，字段为 NOT_SUPPLIED_NOT_CHECKED|
|phase2 只确认、不改 phase1|`evaluate_s_ablation.py:299-348`|相同 phase1 SHA 门禁，不重新 bootstrap/选择；输出 CONFIRMED/NOT_CONFIRMED，保留 phase1 原结论；其他结论标 NOT_APPLICABLE。未执行|
|预检命令|`preflight_s_ablation.py:23-101`|本地检查协议/代码/8数据SHA/折大小/整份工厂测试及12个golden；正式才检查 SSH、精确CUDA环境和两个全新2轮GPU probe 的 last digest；本轮所有远程/GPU项 NOT_RUN|
|合成测试|`tests/test_s_ablation.py`|36项通过，覆盖要求的阈值/CE/早停/配对/每类判读/前导零哈希，另覆盖 val门禁、环境拒绝、固定槽位、缺run不出结论和训练中不触碰heldout|

任务书文字中的“类平衡 CE”按其“以冻结 JSON 为准”执行为 **inner 选点指标**；训练损失保持 JSON 明确指定的 FocalLoss，没有修改协议。

## 3. 四臂实际 CPU 冒烟

共同配置：seed42、fold0、512 fit / 256 inner / 256 heldout；每轮一个 batch、共 **2 个优化器步/臂**。每臂 best 和 last 都为第2轮；训练后各预测 heldout 一次。以下只证明工程流程可运行，**不是模型性能或结构消融结果**。

|臂|状态|参数/MAC|inner CE 第1→2轮|循环与预测耗时 s|初始 digest（前12位）|
|---|---|---|---|---:|---|
|S0|通过|105938 / 6764928|0.692605→0.690438|4.039|c86a31c99c38|
|S1|通过|103810 / 6762880|0.692313→0.687667|4.524|40cdc82c7516|
|S2|通过|105106 / 6640128|0.688846→0.686100|1.249|2f4cf7a1eb6d|
|S3|通过|105929 / 7097374|0.696693→0.691929|1.257|61c67e76cdf3|

`wall_time_s` 统计训练循环、inner 及末端 heldout 预测，不含启动前身份核对；不把这四个小样本计时作为结构速度比较。

输出根目录：`artifacts/s_ablation_mstca_v1_smoke_20261106/`。

- `runs/S{0,1,2,3}_seed42_fold0/`：report、factory receipt、implementation manifest、原 ID 人口、best/last、两个预测 npz、read_guard。
- `smoke_summary.json`：完整 digest、轮数、窗口数、耗时汇总。
- `preflight_release/preflight.json`：`LOCAL_GATES_PASS_FORMAL_NOT_AUTHORIZED`；四臂 smoke=true、golden_tests_passed=12、formal_clearance=false。
- 三折全量 metadata 实测分别为 **95142/12750/54177、97156/12157/52756、95122/11811/55136**（fit/inner/heldout），与 r1 一致。
- 八项身份核对通过；唯一跳过项始终为 `cache/train.pt` 全量 SHA。正式模式必须核对全部九项。

开发冒烟后仅加固了正式槽位及预检/评估门禁，没有替换或重跑四个已完成的真实冒烟。四个 run 保留执行时 manifest；`smoke_source_snapshot/` 是按开发差分还原并逐文件 SHA 验证吻合这些 manifest 的源码快照，`release_source_snapshot/` 为最终验收源码。`source_snapshots.json` 记录核对；最终服务端应使用 `preflight_release/implementation_manifest.json` 的新 SHA。**任何正式 phase1 尚未开始**，正式开始后不得再更改该 manifest 中的文件。

## 4. 完整合成评估演示

`synthetic_demo/` 使用上述四个真实小样本预测，加上其余 **32 个明确标为合成的 run**，构造同一 **768 个 train 元数据窗口**的完整36-run结构。不再读取任何额外源 X。

最终评估：`synthetic_evaluation_final/evaluation.json`、`summary.md`。

```text
status = COMPLETE_SYNTHETIC_DEMO_NOT_RESULT
performance_claim = false
population = 768
association_groups = 587
bootstrap_replicates = 2000
bootstrap_rng_seed = 20261106
```

输出包含 best 指标、每 seed 和聚合比值/AP区间、类约束、分层指标、last敏感性、每run资源和判读字段。**其中的结论只是判读代码演示，不能用于决定 SE/多尺度是否保留或宣称 MSTCA 优于 CNN。**真实评估默认拒绝这些 CPU/synthetic 状态，phase2 门禁也拒绝该合成评估文件。

缺 run 的合成单测验证返回 `INCOMPLETE`、空 decisions、performance_claim=false，不输出可误用的阶段1结论。

## 5. pytest

最终联合运行记录：`pytest_release.txt` / `pytest_release/junit.xml`。

```text
76 passed, 4 skipped, 1 xfailed in 11.47s
```

- 新增 `tests/test_s_ablation.py`：**36 passed，0失败、0跳过**。
- 既有 factory 测试：40 passed、4 skipped、1 xfailed；**12个完整 golden 均实际通过，未 skip/xfail**。
- 4个 skip 是既有转换输出测试限定旧 factory artifacts 根目录，而本轮只允许新 smoke 根目录；未绕过目录约束、未修改已有测试。它们不是 S 组逻辑用例。
- 1个严格 xfail 是冻结的 FaultDetector P3 记录；没有修改加载器或 P3。
- 新测试用合成 optimizer 流程检查 heldout Dataset 必须在两次 inner selection 之后才创建；这不增加源 X 的读取量。

关键边界用例：精确97%、并列/全同值；按全 inner 两类质量计 CE；严格0.995与早停第50轮；bootstrap每 replicate的12次调用共享同一权重对象；所有判读类型、seed/AP反向；先导零；val缺SHA前拒绝；无CUDA/版本/CUBLAS拒绝；字节散列不要求val能被解析；固定正式任务次序/manifest；缺run不出结论。

复现命令示例（目录须为全新；以下不是正式训练授权）：

```text
python -I -B scripts/research/train_s_ablation.py --protocol configs/experiments/s_ablation_mstca_v1.json --data D:/UAV-Data/causal_input_v1 --arm S0 --seed 42 --fold 0 --phase cv --cpu-smoke --output artifacts/s_ablation_mstca_v1_smoke_20261106/new_S0_smoke
python -I -B scripts/research/preflight_s_ablation.py --local --data D:/UAV-Data/causal_input_v1 --output artifacts/s_ablation_mstca_v1_smoke_20261106/new_local_preflight
```

不得为了挑选更好的冒烟数值反复运行；本轮四个真实冒烟均只执行一次。正式 run 另需精确 CUDA 环境、预先锁定的 implementation manifest 及协议固定输出槽位；不接受 CPU fallback。

## 6. 偏离清单与未执行项

**没有硬性边界偏离，没有修改冻结协议或 factory。**

按授权有意未执行：SSH、正式环境与 GPU 两次确定性、完整 train.pt SHA、任何正式 CV/val 训练、历史全量 M0 sanity 对比。前四项在本地预检中明确记为未通过正式验收，而不是被当作 PASS。phase val 只实现路径并测试拒绝门禁，实际推理/确认未核实。

所有额外运行输出都在指定新 artifacts 下；旧的开发测试/预检/合成演示输出原样保留，最终入口名称为 `pytest_release`、`preflight_release`、`synthetic_evaluation_final`。14个任务开始已有受保护文件（含协议/锁/factory/旧测试/加载器及用户原有修改）的前后 SHA 见 `final_verification.json`。未 commit/push。

## 7. 歧义与数值约定（不改协议）

r1 已解决原先提出的判读重叠、正常/故障 late_hover、val字节核对三项，本次没有再猜改这些规则。

仍需正式结果解释前明确的边界口径：

1. **FP(S0)=0 或 bootstrap 缺类时**，协议没有给出 0/0、不可估计阈值/区间的处理。本实现不加伪计数、不重抽、不丢弃 replicate；返回 unavailable，相关区间不估计、不给肯定结论，并记录 undefined replicate 数。这是保守错误处理，不制造有限比值。
2. **delta AP 的 seed 聚合与 TRADE_OFF 的 AP“反向”方向**没有独立公式。本实现公开记录 delta AP 为三个配对 seed 差的算术平均；TRADE_OFF 的 AP 方向参照 FP 比值点估计所在侧。该解释尚需架构审核在正式结果前确认，本轮合成判读不作为科学结论，也没有据此修改协议。

本地工程冒烟不依赖以上边界来作任何结构选择；它们不影响四臂 golden、训练、inner选点或文件输出已通过的事实。
