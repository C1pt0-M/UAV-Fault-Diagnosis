# M0纯26通道MSTCA：两臂正式训练记录

> 完成更新：北京时间2026-11-02 01:13:21，两臂均正常结束并完成服务器复算打包。随后取回34个成员，核对归档及33个清单成员SHA，并本地独立复算两臂完整34710预测与139轮选择／调度；全部通过。下文“尚在运行”属于启动阶段历史，当前状态以本更新和末尾完成结果为准。

状态：用户明确确认后已启动，尚在运行；本记录不把早期epoch分数当完成成绩。北京时间2026-11-02 00:55:41启动（UTC2026-11-01T16:55:41），SSH别名autodl_UAV3。

## 授权与范围

用户在准备／远端预检通过后确认“确认，开始吧”。仅授权按顺序完成两次M0正式源开发训练：uniform_replacement→balanced_source_normal，均seed42。重复收到的同一句确认不是四次训练授权；执行使用独占execution_started.lock及新目录检查，禁止重复启动、覆盖或失败自动重跑。

M0为现有OnlineMSTCA，26×150基础时序，正常／故障二分类，105938参数。不计算旧9维关系量、不使用新4维残差；旧类只用于初始化来源核验，没有旧模型训练或关系特征前向。

源train162069窗、val34710窗，数据版本rflymad_causal_publication_replay_v1，原标签／mask／原成员／逐窗气压／源train归一化不变，原案例与仿真配对组隔离不变。无最终内部test或Real参与训练／选模／阈值。

## 固定比较口径

- max100、warmup5、batch512、AdamW lr3e-4、wd1e-4、min_lr1e-6。
- Focal gamma2，alphaNone，clip1，无增强／AMP；每轮162069 replacement draws，317梯度步。
- 原两种采样定义保持：总正常／故障概率相同，Balanced仅均衡两种正常来源质量，不额外改类权重。
- 整体完整源val Macro-F1严格最大选best，同分最早；min30／patience20／显著delta0.002早停与严格best保存分离。固定0.5，不用hover FPR挑best。
- 两臂公共主干／头部从同一未训练seed42锚点映射，41个state张量精确一致；M0初始digest a68551099cba00575f1eec6edc31ccf47b39ca8e0fa204ae2fe065d3400be646。不声称不同架构优化轨迹或dropout掩码逐步相同。
- 准备协议SHA保持6ef47aead15fea51e4c7d4b53289383ecd038d67b82e3cae8c07ddcd8b450fe6，不为授权重写此前“准备阶段”历史。authorization.json单独记录本轮启动许可。

## 已观察的实际进度

已看到Uniform完成第1轮，history记录317次梯度更新，initial／best／last权重和源val历史按原循环保存，GPU运行。Balanced由同一有限执行器在Uniform成功结束后启动，不并发挤占GPU。

此处不列早期Accuracy或局部误报结论。最终要分别报告两臂完整源val、故障召回、独立正常／hover误报与故障子类，不能只挑误报较低者隐藏召回变化。单seed源val不是统计显著性、最终测试或真实域泛化证据。

## 执行、核验与取回

远端独立工作区/root/autodl-tmp/causal_pure_mstca_m0_v1_20261102/：
- execute_m0.py：有限两臂顺序执行；状态remote_status.json。
- authorization.json／launch_receipt.json：授权、协议、启动时间与PID记录。
- uniform_replacement.log／balanced_source_normal.log：训练日志。
- runs/{arm}/：initial.pt、best.pt、last.pt、history.json、report.json、source_validation_predictions.npz及快照。
- verify_and_package_m0.py：两臂结束后核对冻结代码／数据、317步和采样计数、LR／早停／strict best、34710原索引／标签／概率与所有分层指标、两臂初始状态和有限权重，然后打包结果，不触发训练。
- verified_m0_source_summary.json／verified_m0_results_manifest.json／verified_m0_results_receipt.json：成功后才生成。

本地辅助位于artifacts/causal_pure_mstca_m0_v1_20261101/（沿用准备资产目录）：execute_m0.py、verify_and_package_m0.py、poll_m0.py、authorization.json及launch_receipt.json。

只读查看：python artifacts/causal_pure_mstca_m0_v1_20261101/poll_m0.py。它只查询状态／末轮／GPU，不启动、重启或改变任何训练。

训练由已启动远端后台进程实际执行，不依赖聊天保持连接；没有承诺聊天会主动通知。完成后还需按manifest校验结果取回，并本地独立复算；只有完成并核验才报告正式成绩。不自动追加seed／模型、评估最终test／Real、部署或关闭付费服务器，不自动Git提交／推送。

## 完成结果（本地独立复核）

|源val指标|Uniform|Balanced|
|---|---:|---:|
|训练轮数／best轮|65／65|74／73|
|Accuracy|96.98%|97.24%|
|Macro-F1|0.969773|0.972353|
|故障召回|97.88%|96.81%|
|全正常FPR|689/17740=3.88%|417/17740=2.35%|
|独立正常FPR|685/4565=15.01%|408/4565=8.94%|
|正常hover FPR|341/754=45.23%|270/754=35.81%|

两臂分别65轮和74轮，按原min30／patience20／显著delta0.002规则正常早停；max100是上限。strict best与显著提升计数分离，因此Uniform末轮成为best但同时触发早停不构成异常，不能据此宣称完全收敛／最优。

与旧9维完整双分支的相同采样seed42固定0.5对照：M0两臂hover FPR均更高，但故障召回也更高；Uniform整体Macro-F1较低，Balanced较高。说明不能简单把误报全部归给旧9维，但不是逐特征增量或同召回工作点结论；架构与参数量不同、实际早停轮数也不同，只有一个seed。

归档verified_m0_results.tar.gz：3056639 bytes，SHA256 656d54b851859eb4a74cea88e4eabd83c85aa15abe4974859e9fce1214b4c660。本地results/包含两个best／initial／last、history、完整源val预测和训练／协议快照。local_recovery_verification.json与local_results_verification.json分别记录取回和指标复核。

检查时远端GPU0MiB／0%，训练进程已退出，无异常日志；未关闭服务器。无新训练／模型推理、最终test／Real读取、阈值扫描、额外seed、部署或Git推送。
