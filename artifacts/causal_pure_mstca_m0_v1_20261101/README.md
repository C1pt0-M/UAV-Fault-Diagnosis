# M0：纯26通道MSTCA的源验证结果归档

两组正式训练已于北京时间2026-11-02 01:13:21正常完成（UTC2026-11-01 17:13:21）。本目录沿用准备阶段20261101命名，不代表另一次运行。

## 实验与主要结果

现有OnlineMSTCA，输入26×150，105938参数，seed42。仅当前冻结因果source train162069／val34710，不加入旧9维或新4维。100轮上限、warmup5、min30／patience20，完整源val Macro-F1严格最大选best，固定0.5。

|源val指标|Uniform replacement|Balanced source normal|
|---|---:|---:|
|实际轮数／best轮|65／65|74／73|
|Accuracy|96.98%|97.24%|
|Macro-F1|0.969773|0.972353|
|故障召回|97.88%|96.81%|
|全正常FPR|689/17740＝3.88%|417/17740＝2.35%|
|独立正常FPR|685/4565＝15.01%|408/4565＝8.94%|
|独立正常hover FPR|341/754＝45.23%|270/754＝35.81%|

故障召回与误报必须同时报告。M0仍有正常hover局部短板；与旧9维完整双分支比较不能锁定逐特征贡献，也不是同召回工作点比较。单seed源val不是最终test／Real泛化、统计显著性或部署认证。

## 公开内容

- 独立训练入口、配置SHA锁和测试位于仓库scripts/research、configs/experiments及tests目录。
- execute_m0.py／verify_and_package_m0.py／poll_m0.py保留有限执行、结果核验和只读查询逻辑，不自动启动新训练。
- results/runs/*/history.json和report.json保留每轮调度、loss、分层验证、strict best／早停及正式结果，两臂均完整归档。
- preparation_completion.json和remote_uav3_results保留本地59项相关测试／远端预检证据。远端无pytest，未声称远端pytest套件通过；匹配CUDA环境的独立断言通过。
- local_recovery_verification.json和local_results_verification.json记录归档取回、两臂34710原索引／标签／预测、139轮调度／选模及41个共同初始化张量的本地复核。

## 不公开的本地／服务器资产

不上传initial.pt／best.pt／last.pt、source_validation_predictions.npz、sampling_weights.npz、数据缓存、原始日志、传输归档或凭据。原始verified_m0_results_manifest.json是完整本地归档清单，其权重和NPZ条目仅用于溯源，不表示这些文件也在GitHub；publication_manifest.json才是本次公开文件的精确清单。

GitHub中可查看完整结果与逐轮历史，但不承诺无D:/UAV-Data或原GPU环境即可运行全部原始预测复算。未训练初始化锚点本体同样留本地／服务器，hash和精确映射依据保留。

此前准备协议中的autodl_UAV2／BLOCKED状态和报告中的准备／运行阶段文字是冻结历史；当前执行别名autodl_UAV3，完成状态以completion.json及results/verified_m0_source_summary.json为准，不为更新状态改历史pin。

本次上传前重新运行59项M0及因果输入相关测试均通过；没有重新跑全项目或前端，不把历史冻结协议hash冲突声称为已修。无新训练、最终test／Real读取、部署变更或服务器关机。
