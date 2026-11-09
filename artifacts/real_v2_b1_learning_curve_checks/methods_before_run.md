# B1执行前口径（任何B1训练/评价之前保存）

1. 按补充1澄清（不算偏差）：无可行onset按最差家族率最大、motor+sensor总detected最多、onset最低排序，仍标no_feasible_onset；回执给主目标并列列表及总检出后并列列表。使用整数/Fraction比较边界与ties，不加epsilon。
2. 分割/配方/权重/网格/规则/判据冻结。只读315个dev非quarantine案例及source train.pt/train ledger；封存行读取后立即丢弃（复用B0）。输入白名单记录Real窗口/案例文件，拒绝sealed/quarantine/source val/内部test。Torch仅用于source train张量和指定模板，不实例化/推理M0或MSTCA，不用GPU。
3. Real特征和顺序逐项复用B0窗口读取/索引校验。先拟合5个R full外层模型并与B0 p_R全量对齐；>1e-6或标签/索引不符停。通过后这5个模型复用于1.0窗口/事件，不重拟合或重算B0指标。source特征之后才读取。
4. 每外层training集合按四个stratum_family key分组（不另加flight层），sha256('real_v2_b1|'+str(rep)+'|'+group)，每层ceil(f*n)，同rep比例嵌套，两臂确定性调用同一选择函数。1.0只rep0，全部四fold组。R权重每训练集按B0；S+R保持source原权重，Real整域乘单个标量至与source总量相等，无搜索。
5. 每臂16个fraction/repeat组合*5fold=80个外层fit（含gate中的R5）；加每臂20个有序(k,j)内层fit，共200个fit。相同三fold集合的(k,j)/(j,k)仍按协议各自拟合，不合并或少跑。source S不fit。
6. CPU native线程上限16、不改dispatch；固定GBT参数，其他sklearn默认（包括early_stopping auto，仅训练fold内部）。持久化每fit n_iter/time/窗数、Real组数和fold排除、域权重及model SHA。所有model pickle、window OOF、连续features/probabilities在D:/UAV-Data/real_v2_b1_learning_curve_runtime，不入Git；逐文件SHA写journal/receipt。
7. H1采用平均重复AP差，不先平均概率。group bootstrap288组/2000/rng20261108，每次对每臂每重复重算weighted AP后平均delta；预排序同分块实现，与sklearn sample_weight AP合成校验。0.25唯一主判据，其他比例描述。CI任何未定义重复不drop/redraw/pseudocount，记不可估并披露。
8. frame features每315案例一次，直接用case_windows+b0.stats，无特征/归一化变化；score为float64（GBT原predict_proba路线），NaN仅表示无输入有效窗口。新模型/分数都只写新路径，不覆盖。N1完整正常按role，supplemental不入N1/N2目标，仅描述。
9. B1协议normal_time明确包括scored；整事件false检查使用lv & ref0 & finite，优先于旧ens_event_threshold代码仅lv/ref0的写法。N2只统计fault case中end_t<onset的false事件，曝光只normal_time且t<onset；故障后false另列。onset沿模板为首个label-valid ref1；detected需candidate_start>=onset，分母需onset后有scored，spans另列不冒充新检出。
10. 每(k,arm)仅从j!=k inner OOF grid11档选onset；严格按N1+N2合并曝光而非两率平均排序。outer仅用选定一个onset，不扫outer；S只0.95。无可行标记本身不额外否决外层H2（原协议只看四个pooled点值）。全部grid和ties保留。
11. group bootstrap事件仅重采样已经固定的outer预测/选定onset，不重新训练/选择。四个H2量CI只描述；PI回复要求时延含CI，因此同一计划额外给motor/sensor median/P90 CI，不用于判定。样本缺分母/缺检出时记null/次数，不冒充0、不丢弃/重抽。
12. 每例配对AUC仅1.0 OOF，正类qualified_y1对该例pre_fault，不含post正常作负类；两类不齐记不可估并披露，分层分布只统计defined。按flight的N1/N2、supplement、post false都描述，不进入onset外层选择。
13. 概率阈值FPR@0.8沿B0 order statistic与>= ties，仅窗口描述，不是event onset。学习曲线原始metric CSV可供作图，本轮不增添未请求绘图。后果只按H1/H2返回用户；不启动B2、GPU、T1或修改台账/结项文档。
14. 入口拒绝已有canonical或runtime目录；异常保留已完成fits/分数/日志，不自动重跑。训练可能较长，detached+单次内核等待，不连续轮询。
