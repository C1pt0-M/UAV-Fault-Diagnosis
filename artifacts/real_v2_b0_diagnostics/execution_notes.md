# 跨域B0诊断：完整执行记录、实现选择、失败与边界

## 按补充1澄清（不算偏差）
- 在任何Real npz打开或B0指标计算前收到补充1：采用B。session只训练主分析5折，远端/近端子集复用相同OOF分数，不重新训练。远端负类end<=onset-250，正类不变。远端AUC和CI、被筛近端1314窗/264案例及其单独AUC和CI均报告。近端案例指至少一个近端窗口的案例，可与远端265案例重叠。

## 所有失败与重试
1. 合成测试r1为14 passed、1 failed：曝光测试将未附加family/dev_fold的原始toy台账直接传入audits，而实际流程先通过read_ledger附加metadata。失败是KeyError family。只补齐toy元数据，未改生产审计公式或真实数据。r1日志/XML/临时目录保留；r2为15 passed，增加正式访问断言后r3仍15 passed。所有轮真实数据/模型访问尝试0。
2. 正式B0仅执行一次，退出0，166.545秒；复现门、10个诊断fit、指标和曝光均无失败/重试/恢复。没有为改善结果重训、改标签、改排除或调整超参数。没有依赖安装、解释器切换、服务器操作或GPU调用。

## 输入边界与复现门
3. split SHA严格匹配9232ee35e7a59f4fac4534fc8e98fc454380b51e6e5d8439aaa0f769cd1090ee；冻结S SHA严格匹配df468524e2047205dd8422ab3e486ce7692f0ac312276e20394d6f88a1bb6774，运行结束再验相同。开发池320，先去5个quarantine再加载315案例，288真实组。quarantine与sealed都未开npz。
4. 台账按4096行chunk读入后，第一操作就是仅保留eligible UID，再校验/处理；不对封存台账行做标签、特征、统计。白名单只允许315例的windows/cases路径和real_included_windows，拒绝其他UAV-Data读入、所有输入写入和内部test目录（明确授权的S pickle除外）。实际窗口315文件/315打开、曝光297文件/297打开，所有路径在eligible集合内；封存/内部test打开0，阻止尝试0。实际文件名全部列入receipt与results.access。
5. gate只先读取eligible adapt_normal9例+calibrate_normal4例的窗口，用冻结S打分，与D0每例窗口数和>=0.5数量逐项一致后才加载剩余开发案例和拟合。不是先看B0结果再补门。D0只读取需要的4列，立即筛本轮gate UID和D0臂，不使用M0/ENS旧数值。
6. 直接读windows npz，原始索引集合、y==qualified_y、window_start_row逐一对齐；18480窗均保留。没有CausalCacheDataset、cache/real_evidence.pt、source train/val/test或封存payload读取。只复制/AST核对run_test.py的stats函数与GBT关键字参数，不import会触发顶层训练/test访问的原脚本。

## 事先确定的实现口径（methods_before_run.md已在打开Real npz前保存）
7. 130维特征为原始float32的mean/std/min/max/mean|diff|，无归一化、特征改造或域对齐。固定HistGradientBoostingClassifier(max_iter400,learning_rate.05,max_leaf_nodes31,l2_regularization1,random_state0)，其他sklearn默认（含early_stopping auto）。CPU native线程上限16只是资源控制，不是超参数搜索。
8. Session每训练折用sklearn compute_sample_weight('balanced')，两类总权重相等、总权重=训练窗数。R每训练折用正常总数，在两normal_origin各分一半，fault权重1；不从held fold计算权重。5个固定dev_fold、关联组互斥，session5fit/R5fit后销毁模型，不保存任何部署模型。默认内部early-stopping仅发生在外层训练fold内，不接触外层held组。OOF CSV中的p_session对非session任务窗口留空，不是漏预测。
9. normal_full按family normal且旧角色test_normal/adapt_normal/calibrate_normal；supplement按normal且非这些角色。实际32完整正常、18补充正常（1例与motor同stratum，不能只按stratum_family normal_segment17算）；正常总50例，motor154、sensor111。隔离quarantine先于这些数值分析，未因结果删例。
10. B0-2 motor/sensor只限制故障正类，正常侧保持pooled定义不变。(b)负类是所有开发故障案例pre_fault，(c)是所有normal_flight origin（含完整正常与补充段）。(b)是池化的故障-vs-pre_fault AUC，不是逐飞行案例配对AUC，不能声称每个案例内部都已可分。
11. 五个FPR正常子集按角色/phase识别；若有未覆盖正常，事先规定不删、不改标签，仍入all-normal分析并单列数量。实际未覆盖正常0。transition正常只qualified_y0；R拟合和(a)也包括qualified_y1的transition，不以whole_window_label_valid再筛一遍。
12. 所有AUC为窗口pooled点估计。预计算排序和同分段，bootstrap整组重复权重计算，与sklearn sample_weight AUC相同（toy含ties已验证）；不是case均值AUC。全288 eligible组同一配对计划，2000次rng20261108，不在bootstrap重训或重选参数；CI条件于固定OOF/源模型，不覆盖训练随机性。
13. 不伪计数、不丢弃或重抽缺类重复：某项有任一重复缺正/负类，该CI记null并报告数量。所有B0-2 AUC及session主/远/近AUC均2000次有效。给目录AUC额外附CI时，acce/circling/hover/waypoint分别有6/7/1/4次缺类，故额外CI无法估计；点AUC仍报告。velocity目录CI正常。没有改抽样总体或删重复来补出好看的区间；原要求的目录描述性AUC不缺。
14. AP只给(a)点值；(c)-(b)点差按要求报告。每臂FPR阈值取pooled故障降序ceil(recall*N)分数，即满足目标的最高fault order statistic，>=保留ties；同时报告实际召回。0.5目标实际0.50008217，0.8实际0.8，不能假称精确0.5。空正常子集FPR会null而非0，本次五个子集都非空。
15. FPR阈值是协议要求的描述，不是部署工作点，不在sealed评价、不写回配置。S不是Real OOF：它是冻结源模型直接概率；R才是Real 5折OOF。S在两类正常上接近1不意味着标签被改成fault，标签始终qualified_y。

## 标签曝光及额外描述
16. 曝光仅打开32个完整正常+265个故障case的row_input_valid/row_label_valid两数组，补充正常18例不需cases文件。normal_full valid小时0.73277222；fault pre取row_index<onset、valid交集，2.81141667小时。固定每行0.02秒，不读取原始传感值、reference_state或事件回放。
17. onset按台账非负唯一值，-1为缺失不当负时长；不一致停止，不补0。本次265故障全部有有效onset且至少1个pre_fault窗口，无缺失。故障前长度是onset_row*0.02，不是仅有有效行的时长；按family、flight及交叉给出分布。label_audit.csv保留每案例、各phase和qualified_y计数。
18. post_fault_normal为2097窗/154案例；transition计数按qualified_y0/1分开，涉及263案例，具体UID见results.B0_3.special_phase_cases和label_audit.csv。增加family×flight分布、近端描述CI、目录额外CI以及零未覆盖正常的检查均是标签/稳健性描述，不增加训练/模型/工作点。除此无额外test或封存分析。

## 解释边界、交付和不做的事
19. Session AUC0.97158为强会话差异；远端复用主OOF仍0.97388，说明主结果不是仅由近起点负窗支撑。这是描述性稳健性判断，不认证早期故障不存在或差异的物理原因。
20. R主(b)0.975056>=0.75按冻结规则建议GO；motor/sensor均>=0.65，无不支持层。S的(c)-(b)负值如实报告，不能称为(c)正向虚增；R/S比较不是已完成S+R价值验证。GO只是给用户的B1建议，B1没有启动，不自动冻结/修改B1协议或训练T1。
21. 全465 Real案例历史已有汇总结果，旧adapt/calibrate正常也用于过D2/D3。sealed仅是本路线未用于开发，不是盲测。B0诊断模型不供部署，不代表封存性能、实时检测或事件可用性。
22. 原用户脏case_relative脚本、两个audit日志、冻结split/protocol的原字节和mtime保持不变；不生成golden，不改源数据/源train/val、结项报告或台账。只把新增代码、合成测试、诊断产物和报告提交；模型/原始npz不入库，docs/artifacts用git add -f，提交后push。

没有其他正式运行失败、重试、特征/权重搜索或未披露的额外分析。补充1是执行前澄清，不计为偏差。后续Git若失败将另行回报，不重跑B0来解决传输问题。
