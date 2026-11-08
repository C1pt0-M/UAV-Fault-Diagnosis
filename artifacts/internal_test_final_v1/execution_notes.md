# 内部test最终评价v1（补充1授权重启）：完整偏差、失败与实现口径

## 授权、历史失败与现场
1. 第1次启动4516868因-I使用户site-packages不可见，导入threadpoolctl失败。没有加载模型、读取test案例payload、生成逐帧分数/指标或新锁；原results全null。架构方已核对“结果未被任何人看到”。依用户《内部test最终评价_PI指令_补充1.md》，本次属于已批准的同一次最终评价重启，不是额外一轮评价，不算重跑指标。历史ens_internal_test_v1已用过内部test一次的事实仍披露，本次是注册的第二次、最后一次使用。
2. 按要求git mv保留第1次现场：artifacts/internal_test_final_v1_attempt1_env_failure/；报告改名为内部test最终评价_第1次启动失败记录_v1.md。9个移动文件内容和mtime原样保留；旧checks目录所有文件不动。新检查只放checks_r2。原记录中的旧路径保持历史原文，没有事后重写。
3. 改名和修复已独立提交fe00ff444c7d73029a96ffb440a5650b94af26cd并先push，再做预检和正式评价。新正式参数-X utf8 -E -P -B，仍用D:/python3.12.4/python.exe，不安装依赖、不切换环境、不改CPU dispatch。
4. 外部评分目录因第1次启动已存在但为空。按补充1只放行已有空目录；非空目录、非目录路径、已有canonical结果/锁仍拒绝。没有删/移动空目录来绕过检查，没有覆盖任何已有评分。原协议、模型、评分内核、策略、工作点、指标公式均未更改。

## 完整预检与本次执行
5. 本轮合成测试一次26 passed、0 failed、0 skipped；真实数据/模型访问尝试0。旧checks保持不变。修复后的目录保护测试同时覆盖空目录接受、非空/文件路径拒绝和旧锁保护。
6. 预检以正式参数在子进程导入17个模块，含numpy/pandas/torch/sklearn.ensemble/threadpoolctl、MSTCA模型、causal_input、检测策略/事件聚合/monitoring_session/fault_detector、完整评分与事件模板。模型加载前、加载后、val打分后两份SHA均相同且匹配冻结值。
7. 预检仅用SIL/acce/GPS/TestCase_107_1406000106（3326有效帧）和HIL/acce/GPS/TestCase_107_2406000106（3246有效帧）。确定性选现有val索引中各domain的首个排序UID；不读取tasks.json、test台账/预测/payload或Real。使用与正式运行相同score_arrays、load_m0、全量GBT pickle和归一化、CPU16线程、M0批2048/GBT批4096，与旧val分数的时间轴和可用帧mask完全一致，四项最大差均0。没有val性能统计或阈值选择。预检一次PASS，阻止访问尝试0，用时15.215秒。
8. 补充1授权的正式启动一次，553例各打分一次，退出码0，总wall_time_s=576.7881263。进程detached，launcher内核等待而非轮询并记录退出码；没有I/O/模型/指标失败、重试或恢复。
9. 启动三资产SHA匹配、553任务/48GPS/10排除案例/277真实分组验证通过。34802窗全量对齐，无缺失、标签一致；GBT最大差2.9795787059860857e-08<=1e-6，M0最大差2.384185791015625e-07<=0.002。门通过后先于任何性能计算写新锁，UTC2026-11-08T11:24:36.997128+00:00。
10. 逐帧553个npz和index.csv保留D:/UAV-Data/internal_test_final_v1_scores/，不入库。每个npz的SHA及index SHA在receipt/scoring_receipt中；完成后只校验文件字节SHA，不再推理、事件聚合或重新计算指标。报告只读取已保存结果进行格式化，没有事后额外test分析。

## 事先实现口径与边界（不隐去）
11. PI原禁止项“不读train/val以外的其他留出数据”字面与授权test步骤冲突，按冻结协议为准，只访问本轮授权的source test输入。完整链路预检另按补充1授权读取两个val案例；未读Real或其他留出payload。tasks.json及指定GPS表为授权元数据，按test UID使用。
12. 分组取tasks.json中的真实group，与全部test窗口台账association_group逐行核对，保留前导零；无自造/缺失组。GPS排除10个案例的名单在评分前固定，SIL/HIL各5，不因看到结果增删。72独立正常案例、475个已评分故障案例（其中排除10，剩465）；未评分故障沿原post_onset_scored定义不计入“已评分”分母，不是后验删例。
13. 台账window_end_row为exclusive，帧一致性对齐物理末行=start+149=end-1；F3排除仍严格使用协议字面的window_end_row*0.02>30，不偷换物理末行。GBT窗口>=0.3744；M0窗口取原两列概率argmax（平局归normal）。F3的1pp边界用整数TP/N精确比较，不加epsilon。
14. 事件定义、曝光、false_event、detected和spans_onset直接复用GBT事件工作点模板。detected不把跨onset事件当作新检出，spans可与后续detected在同案例共存，不能把两者相加冒充互斥案例数。曝光为有分数且label-valid normal帧数*0.02；独立正常误报与故障前误报分开。
15. F2双方detected剔除案例排除集，但1%分母按冻结原文取完整已评分475，容许4.75，不四舍五入为5，同时披露去排除分母465。390>=345-4.75=340.25，PASS；单类缺口严格>2才标记，陀螺缺口恰2不标记，磁力计1，其余不缺。未放宽判据。
16. F4是次要描述，按“每类/全部共同检出”用完整已评分故障总体，含排除案例；每臂的类时延只在该臂detected案例中统计，表中中位数差不代表相同案例配对。全部共同检出328例的配对差另列，中位+0.30s、P90+8.698s。GPS中位慢9.34s、陀螺慢3.44s均标记，不改主判据CONFIRMED。
17. 排除集首个合资格confirmation用原grid timestamp_us/1e6的绝对时钟，不减crop起点；<30、=30、>30单列，等于30为0。10例中GBT无合资格事件检出，M0有7且全在>30；后段215个mode4 GPS窗口召回另给，不能用窗口召回冒充事件检出。
18. Bootstrap严格277个真实association_group、2000次、rng20261106。全部注册案例进抽样计划，事件CI仅独立正常贡献误报/曝光，窗口CI全部normal监督窗口贡献FP。两CI相同排序/RNG/group计划，固定工作点，不重选阈值；无伪计数、丢弃、重抽，未定义重复均0。直接复用事件helper中M0-tuned别名仍为同一0.60，重复别名不产生额外工作点；窗口FP聚类按相同计划最小实现。
19. 事件误报优势不等于窗口FP优势：事件5 vs24，比值0.20833；窗口502 vs475，比值1.05684，CI跨1。窗口阈值与事件onset是不同工作点族、正常总体也不同，不能写成所有指标均优。F3六类PASS、F4两类慢>2s都照实报告，不改判或重选。
20. M0用冻结部署B best.pt，不是没有全量模型的T0。窗口阈值训练时的T0参考不能在test复现，沿协议既定部署对照。ENS342只作已知历史常数，没有新推理/重算ENS、GBT@0.5或任何其他工作点。
21. 确认时延相对reference_state首个有效fault帧，并非认证物理损伤时刻；回放策略时延不含模型实时推理、网络/队列成本。此内部test已有历史使用，不是全新独立外部验证。CONFIRMED只由F1/F2决定；源域线结束、test永久关闭，不自动开展后续评价。
22. 原用户脏脚本、两个audit日志、旧锁、协议、模板、archived失败记录和旧checks的SHA/mtime保持不变。没有生成golden、改冻结策略/模型/部署默认值、结项报告或实验台账。docs/artifacts仅对指定新文件git add -f；不提交模型、原始输入或逐帧npz。

本次新预检和正式运行没有失败/重试；历史第1次导入失败已完整披露。没有其他阈值、判据放宽或未报告的额外test性能分析。Git若有后续异常会另行回报，不重启评价。
