# 模型特征、预警语义和阶段捷径：逐项核对

日期：2026-11-01。范围：当前源码、已保存源验证结果、源train/val窗口元数据和无模型合成概率序列。没有连接GPU、重训、加载模型权重推理、读取原始日志／最终测试张量或修改运行逻辑。

## 结论

用户指出的特征物理语义问题和分类分数预警的能力边界确实存在；正常悬停后段高误报也已复算。但不能把9项全部判为数学无效，也不能把阶段捷径说成已证唯一根因。尤其lead_time_s不是稳定定义的k_on确认延迟，其实际调用还涉及完整事件输出／flush，甚至可为空。

## 1. 模型层：物理解释有问题，不等于所有统计运算无效

依据lstm.py:263、348起的OnlineKGMSTCA和_relation_features，以及data/causal_cache.py:114–116：当前26通道研究数据先逐通道z-score，随后才在网络内构造9项关系特征。模型是双时序编码器加投影／门控，没有知识图谱实体边或GNN消息传递。

- effort在当前26通道分支确实为mean(concat(z_ctrl,z_out))/clamp_min(z_acc_norm,0.1)。物理加速度模接近训练均值时z值接近零；所有z值不高于0.1均走同一分母。因此它不是推力效率，0.1也不是物理单位的低加速度阈值。但分母截断后的表达式数学上仍定义良好，不能直接推出它无信息或是全部误报的根因。31通道有独立actuator_effort列时走另一分支，不能混称所有部署版本公式完全相同。
- causal_pipeline.py:18–21把control[0:4]和output[0:4]分别映射成四个控制／输出槽。PX4 v1.13定义前三槽为roll/pitch/yaw，第四槽throttle，不是四个电机独立目标。源码逐槽abs(ctrl_i-out_i)没有混控映射，不能解释为执行器跟踪误差。输出PWM／驱动命令也不是测得的电机转速或推力；用户比喻中的“1号电机转速”应改成“1号输出槽的驱动命令”。官方字段约定不替代对全部历史自定义固件的实证，当前未做该扩大核查。
- vertical_consistency确为abs(diff(z_baro)-z_vel_z)，没有除0.02秒，没有逆变换到统一尺度；PX4 vz为NED向下正。对向上正的高度，在已对齐的物理单位和时钟中，理想运动学关系是dh/dt约等于-vz，而非直接相减。只是改为加号或补乘50还不足以校准传感器／估计器延迟与噪声。本次不替换公式。
- 输出标准差、极差、标准化角速度范数等可以是合法统计特征；逐通道z-score后都是无量纲数，问题是把它们解释成物理残差／能量，而非算式无法运算。9项作为整体的独立预测增益尚缺当前公平消融，不能先宣判全有害。

外部原始依据（本轮通过PX4官方页面核对）：
`https://docs.px4.io/v1.13/en/msg_docs/actuator_controls`
`https://docs.px4.io/v1.13/en/msg_docs/actuator_outputs`
`https://docs.px4.io/v1.14/en/msg_docs/VehicleLocalPosition`

## 2. 预警层：分数后处理，不是已验证先兆预测

risk_evaluator.py:31–88中risk_score为fault_prob的EMA，默认alpha0.35；normal/watch/warning/critical默认阈值0.35/0.55/0.75。另有短3点／长8点均值差的rising/falling/stable描述，默认trend_delta0.05；has_preview直接推到warning，has_confirmed直接推到critical。因此“单个分类分数的规则后处理”准确，“只有三个硬编码阈值且无其他状态”不够完整。这些值是构造参数，主MonitoringSession使用默认；独立RealtimeFeatureAlertEngine还覆盖为0.12/0.24/0.45并使用特征风险分数，不能外推整个仓库所有预警都只来自同一个模型。

这条主链没有将未来故障时间／退化趋势作为预测目标，也没有在上述方法中读取真实故障起点。EMA和事件滞回可用于告警平滑／去抖，但本身不是提前预知故障的证据；softmax的EMA也不是经校准的未来风险概率。

lead_time_s实际语义：
1. decision_engine.py:57–63计算调用方传入ts减preview_started_at；函数名confirm_event不说明ts必为k_on确认时间。
2. event_aggregator.py:168–250在第k_on点设置confirmed_t，但通常不返回事件；事件在低于clear阈值满足关闭条件、等待合并窗口后，或flush时才作为完整事件返回。
3. monitoring_session.py:180–235将new_events当确认事件并传入当前流时刻；stop_detection:246–267则传入event.end_t。preview可能早在低于onset时清空，last_preview_lead_time_s也可能保留旧值。

合成核验脚本scripts/research/check_warning_semantics.py直接调用现有聚合器和Session，未加载分类模型：

|输入与默认规则|confirmed_t|emitted_t|lead_time_s|
|---|---:|---:|---:|
|每0.1秒0.8分，0–1秒持续高分，然后flush|0.2|1.0|1.0|
|0–0.8秒0.8分，之后0.1分，持续到1.5秒|0.2|1.4|null|

第3个点确认所需时间是0.2秒，但第一例lead_time_s为1秒，第二例为空。因此“无独立故障真值，不能称早期先兆提前量”正确；“它一定就是3次连续阈值确认延迟”不正确。后续若修接口，应区分候选起点、confirmed_t、emitted_t、观测到的故障起点与UI送达时间，不能仅把lead_time_s重命名后当问题已解决。这里的0.1秒仅为合成推理间隔，不是声称所有线上配置均如此。

另一个已复现的文案错配：RiskEvaluator第一次输入0.8且has_confirmed=False，因score>=0.75仍得到critical，却输出“已形成稳定故障事件”。说明critical分数分级被文案当成事件确认，主decision_engine也有同样的critical理由。当前未改代码，只记录。

## 3. 决策边界：分布和错误模式已证，捷径因果未证

当前元数据复算：
- 源train正常83051，其中故障案例未激活正常61906，占74.54%；其phase为pre_fault59808、normal_flight2098。严格pre_fault占全部正常约72.01%，不是74%全部都是故障前早段。
- 源val正常17740，其中故障案例未激活正常13175，占74.27%；pre_fault12727、normal_flight448。
- 来源均衡臂在抽样时将独立正常与故障案例未激活正常两池的正常概率质量各分一半。因此74.54%是原始合格训练正常窗口组成，不能当均衡训练每轮的期望正常来源比例。
- 恢复原预算后的源val正常hover原裁剪最后四分之一198窗，均匀误报177（89.39%）、均衡135（68.18%）。是已保留裁剪区间的相对位置，不是已认证的飞行阶段；198重叠窗口也不是198次独立飞行。

依据artifacts/causal_source_longbudget_v1_20261101/local_results_verification.json及configs/experiments/causal_source_development_v1.json，两个split的included_windows.csv只读用于phase计数；不新增模型推理。

这些支持正常来源／阶段混杂与阶段捷径的研究假设，但并未直接测出模型使用了哪个通道作为捷径。时间位置不是当前显式输入；“故障都在中后段”“模型完全不理解正常动力学”不应当成这次已经证明的机制陈述。也尚未分离原始分支、关系分支、门控、训练目标及残留分布差异的贡献。

必须区分层次：89.39%／68.18%是源研究模型固定argmax／0.5的窗口误报，尚未经过UI的EMA、0.6事件阈值或多级预警。因此预警层语义问题不能直接解释这些窗口误报，研究权重也不等于已替换到31通道旧部署的在线权重。

## 汇报建议

可确认：当前关系分支缺乏所宣称的物理一致性；主预警链是分类分数的规则化分级及事件处理，尚不具备被验证的故障先兆预测能力；特定正常悬停片段有严重窗口级误报。

需保留为假设：阶段捷径是全部误报的原因、9维分支导致所有退步、删分支或换图网络即可解决。代码事实、数据分布、性能事实和机制因果应分开报告。

本轮新增最小可运行检查和本文，不修部署、不扩实验。检查PASS表示复现现状，绝不表示现有预警语义正确或系统验收通过。
