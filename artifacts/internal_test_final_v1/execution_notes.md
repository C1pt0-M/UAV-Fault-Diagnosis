# 内部test最终评价v1：完整失败与偏差清单

## 结果状态
本轮唯一一次正式启动在FRAME_SCORING阶段、任何模型反序列化/案例评分之前失败：ModuleNotFoundError: No module named 'threadpoolctl'。没有任何性能指标，没有CONFIRMED或NOT_CONFIRMED科学结论。不能把环境失败写成NOT_CONFIRMED，也不能以指标尚未计算为由擅自重启。

## 所有失败、重试、实现选择、缺失项与额外诊断
1. 严格按e475a42冻结协议，只实现GBT窗口>=0.3744、M0窗口argmax、GBT事件onset0.95、M0事件onset0.60；没有test网格、附加工作点、拟合或训练。正式命令仅启动一次，无重启、恢复、模型/指标重算。
2. 执行前只做资产SHA与人口/台账列结构检查：553例、48GPS、排除10例、真实任务group277组，均符合启动条件。各GPS均有mode；排除名单在任何评分前固定。任务group与34802条test窗口台账association_group一致，保留前导零，不自造group。
3. 模型SHA与已有test_predictions.npz SHA全部匹配；旧锁存在。正式流程再次核验身份属于只读启动检查，不是指标重复计算。已有预测只读取schema、范围、索引/标签合法性用于对齐准备，没有阈值分类或性能统计。
4. 合成单测一次26 passed、0 failed、0 skipped；包括14项本轮新检查和12项既有事件检查。测试进程拒绝真实D:/UAV-Data和旧test模型/预测目录访问，访问尝试0。单测用假模型测试score_arrays，未覆盖score_cases导入完整scikit-learn依赖链；前置依赖探测不足是本轮实现/检查遗漏，不能用26通过声称实际评分环境已完整验收。
5. 正式运行选择-X utf8 -I -B，是实现选择而不是协议强制要求。-I排除了用户site-packages。threadpoolctl实际存在于C:/Users/23119/AppData/Roaming/Python/Python312/site-packages/threadpoolctl.py，但在所选隔离解释器中find_spec返回None。导入score_ens_frames第11行的sklearn.ensemble，沿sklearn.utils.parallel第13行导入threadpoolctl时失败。不能笼统写成系统根本没安装该包。
6. 异常发生在score_cases的from score_ens_frames import load_m0；尚未执行load_m0、pickle.load或案例循环。因此模型推理0、test案例payload读取0、生成逐帧npz0。外部评分目录已创建但为空，保留不删、不移、不改名；canonical输出目录也保留，脚本会拒绝重入。
7. 未安装依赖、未修改sys.path绕过隔离、未切换解释器、未修改环境/模型/批大小/CPU dispatch后重启。只进行无模型、无数据的find_spec环境可见性诊断，属于额外失败定位，不是额外test性能分析。
8. 一致性门未运行：对齐窗口数和GBT/M0最大差均null，不是0。新TEST_EVALUATED.lock未写，不能伪造门通过。F1/F2/F3/F4、全部描述性统计及两个bootstrap均未执行；events_per_case.csv、scoring_receipt.json和逐帧分数SHA无法生成。已授权输入的旧锁一直只读。
9. parity_gate.json和results.json在失败后仅补写NOT_RUN/NOT_EVALUATED行政状态，所有指标均null，不重算任何指标。原始receipt.json、run.log、invocation_r1.log不改写；failure_traceback.log为完整原始stdout/stderr的字节副本。没有伪造空事件表来暗示零事件。
10. 进程使用Windows detached启动，不依赖观察器；只做一次内核等待式观察、没有轮询。观察时进程已退出，OpenProcess返回87，无法再取得OS退出码；记录该事实，不虚报已捕获退出码，不据此重新启动。失败由原始traceback和receipt确认。
11. PI禁止项“不读train/val以外的其他留出数据”字面与本轮test授权冲突；依“以冻结协议为准”，只访问本轮明确授权test元数据/已有预测和资产。未读Real、其他留出、train/val案例payload或cache。tasks.json和指定GPS mode表是授权元数据，按test UID筛选使用。
12. 台账window_end_row为exclusive（end-start=150）；拟定一致性对齐物理末行为start+149=end-1。F3排除按协议字面用台账window_end_row*0.02>30，不能混成physical_end_row时钟。该实现已用合成数据检查，但未在真实test计算指标。
13. F4实现按“每类/全部共同检出”采用完整已评分故障总体（含案例排除集）；硬约束F2和窗口F3按各自注册排除定义。此为事先实现口径，不是看到结果后的取舍，真实F4未执行。事件bootstrap直接复用既有helper，其中M0-tuned别名固定0.60，等同同一已注册基线，不产生新工作点；窗口bootstrap与事件使用同一排序/group/RNG计划。实际两个bootstrap均未运行。
14. 源数据只读白名单、旧模型/预测和旧锁的禁止写入/删除保护已安装。8个既有受保护文件（含用户脏脚本/两个日志、原模板、协议、旧锁）SHA和mtime保持不变；旧模型和旧预测SHA也仍匹配启动值。未改case_relative开发脚本、原audit日志、golden、结项报告、实验台账或冻结协议。
15. docs/和artifacts/按现有忽略规则仅对明确列出的本轮新文件git add -f，不修改.gitignore，不stage用户已有脏文件。仅提交/推送失败证据和实现，不提交任何模型、原始test输入或逐帧npz。

没有其他指标尝试、隐藏重跑、放宽判据或阈值分析。若后续Git提交/推送失败，将另行追加报告，不借此重启评价。
