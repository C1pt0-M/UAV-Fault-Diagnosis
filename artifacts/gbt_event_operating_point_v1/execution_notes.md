# GBT事件级工作点v1：完整偏差、失败、重试、实现选择与未完成项

## 范围、协议和输入
- 当前开始HEAD1b06e9758a9cde34729ae01c1e8bca5878dfa868，包含冻结提交3a9ac66dfe8898d590ab89ffb38acf85a27d3e43。运行前确认该协议从冻结提交至当前没有diff。本文/协议的事件策略、11档网格、6类约束、排除集和选择规则均未更改；未发现两文直接冲突。
- 只用现有3135个source逐帧npz及index（train2580/val555）、指定GPS mode表和真实train_all_windows.csv中的association_group。没有加载模型checkpoint/GBT pkl、原始特征cache、internal test或Real。旧模块replay_m1_continuous虽然定义推理函数，本轮只调用aggregate_scores，未调用replay_case/model.forward。
- 只进行本地CPU聚合，8 workers（参照原脚本16）；没有训练、新推理、服务器连接或golden生成。运行一次，COMPLETE，用时632.2956524s。没有正式数据运行/复现门失败或重试。

## 所有失败、重试和定位变通
1. pytest r1：9 passed、1 failed。新旧case_events逐字段比较已完全相同，但额外toy断言错误地预期1个false event。0.4s gap在固定merge_gap=0.5s下被合并，正确为0。保留pytest_r1.txt/xml。
2. 修正/扩展toy为0.4s和0.8s两种gap后，pytest r2：10 passed、1 failed。剩余错误是预期合并事件能产生新detected；读取event_aggregator.py确认其保留原candidate start，所以跨onset事件不算新检出。只修toy预期，不改原/新事件算法、阈值或策略。保留pytest_r2.txt/xml。
3. pytest r3：11 passed；增加父/子进程源数据边界后，最终r4：12 passed、0 failed、0 skipped。各轮使用新的basetemp和日志，未覆盖/删除失败输出。边界测试中的预期PermissionError不是实际数据运行失败。
4. 专用路径搜索对ignored artifact/ledger弱命中，随后按明确目录名和文件系统确认。只列举到real/test文件名，没有打开其内容；group只读取train_all_windows。该台账有不属于score-index的行/空group，按case_uid先筛2580个train输入后，全部唯一、非空，共1296组，未修补/自造任何group。
5. 第一次元数据stdout显示中文路径有终端编码乱码，文件路径/UTF-8读取本身正确、没有输入解码失败；正式运行和测试统一-X utf8 -I -B及显式UTF-8日志。没有改输入文件字节或全局环境。

## 预先采用的实现口径（不隐去歧义）
6. 生产事件聚合直接调用冻结aggregate_scores+OnlineDetectionPolicy，summary直接复用ens_event_threshold.summarize。case级计数按其case_events相同公式扩展：只有finalized事件全区间label-valid且normal才算false；曝光为有M0分数且valid-normal帧数×0.02；detected只认candidate_start>=reference onset的confirmation。spans_onset不冒充detected；确认不因最终min_event过滤而重新定义。synthetic parity测试覆盖合并和不合并两支，复现门四数全部吻合。
7. 所有m0/gbt数组 shape、概率范围、时间顺序、label-valid状态检查；实际3135个case的有限分数mask一致，因此沿用M0的scored曝光定义没有改变GBT可用帧总体。只有ENS@0.75用于复现门，没有额外扫描ENS。
8. 父进程及Windows-spawn worker都安装只读输入白名单：D:/UAV-Data下只允许注册score npz、index、train分组台账。val score必须等待selection_train_only.json已写且GBT/M0均有可行train值。实际先gate、再train全网格、封存onset、再val全网格，未按val选择或重选。
9. mode映射覆盖所有227个source GPS案例，无缺失；排除集train36/val8，均已评分。硬约束a的GPS及b配对集合去掉该案例排除集；其他类保留。事件案例级排除比窗口级只去掉>30s窗口更粗，明确披露。
10. b取同一个case上(候选确认时延−M0@0.60确认时延)的中位数，不是两个边际中位数之差；无共同检出则b不可用/不满足。精确0.5允许，不加epsilon。目标误报率使用未舍入曝光；并列的detected总数按协议未限定的完整scored总体解释，之后onset低。本次GBT和M0各只有一个可行档，未触发该并列口径。
11. EV3：detected双方剔除排除集，比较389 vs337；1%分母按原文完整val已评分故障N472，容许4.72、下界332.28，同时附去排除后N464。原文未明确1%分母是否也去排除；为透明，事后仅补一条算术口径核对：若用464，下界332.36，389仍满足。此额外说明不改变onset/统计总体输出/判定，没有把容许数四舍五入为5。
12. 排除集按首个合资格confirmation的原grid时间timestamp_us/1e6（不减crop起点）分为<30、=30、>30，保证案例不重复；=30单列是为消除边界歧义，本轮计数全0。所有排除检出均在>30，train M0/GBT为7/16，val为7/2。
13. by-flight报警占用沿用参照summary的false-event occupancy/valid-scored-normal曝光，不换成所有未经label-valid验证的报警区间。描述性每类同时保留完整总体和去排除总体，报告主表用硬约束总体，并额外披露完整GPS计数。
14. Bootstrap抽全部train案例的真实1296个association_group，所有case配对；只有独立正常贡献误报事件和曝光，故障case贡献0。2000次、rng20261106，同一计划用于相对默认和tuned两比较。固定已选onset、不在bootstrap重选；0未定义重复，无伪计数、丢弃或重抽。CI条件于选定工作点，不包含训练/选点随机性。M0-tuned=默认0.60，两个比较点值和CI完全相同。

## 无法完成的项与结论边界
15. cap64一致性检查：无法核对。本地feature_groupcv_v1三折heldout分数存在，但真实T0 seed42三折heldout_predictions.npz在标准canonical/server_r1路径缺失；找到的smoke/测试npz不是正式T概率，明确不用。未另行推理或从服务器补算。本轮M0帧基线不能据此声称与T0窗口基线相同。
16. PASS只表示冻结EV1/2/3通过。val陀螺缺口3例按协议仅标记，不自行判失败/调整onset；GPS缺口1。GBT若有类别确认时延增加，照实报告，不能用总体median的合格覆盖尾部代价。0.95是固定网格最高且唯一可行GBT档，不扩网格寻找更高阈值，不宣称连续域全局最优。
17. 时延相对reference_state首个有效fault帧，不认证物理损伤时刻；只回放已有分数，不包含推理/网络/队列时延，不是实时部署性能认证。val早已用于开发，本轮不是全新独立测试。
18. 没有改窗口级0.3744、部署OnlineDetectionPolicy默认值、结项报告、实验台账、case_relative_features_dev.py或tmp_control_compensation_audit*.log。原始参照脚本/现有分数均只读。新增报告仅在指定新文件。

19. Git现有docs/*与artifacts忽略规则使新报告/产物不显示在普通status中；仅对列明的新文件使用git add -f，不改.gitignore。git diff曾提示已有脏case_relative文件CRLF将来可能归一化；本轮不stage、不normalize该文件，提交前后核对原始SHA不变。未更新结项报告或台账。

20. 提交前路径白名单检查失败一次：git diff --cached --name-only -z的末尾NUL被split解析成多余空字符串，导致集合断言失败。实际20个已stage文件均正确，git commit尚未调用；原断言内容保存在checks/commit_guard_failure_r1.txt。操作检查器改为仅过滤空分隔记录后重新检查并提交，不改实验源码/策略/结果、不移除任何用户文件，也没有训练/聚合重跑。

没有其他执行失败、重试、条件放宽或未报告的额外分析。正式case聚合一次成功；测试失败是夹具预期修正并全量保留。若后续提交/传输发生新异常，需另行追加回报。
