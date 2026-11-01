# 近期改动归档与后续交接

日期：2026-11-01。用户当前无法取得GPU，要求优先将近期改动及必要文档提交GitHub。本轮仅本地整理、测试和Git提交／推送，不连接训练服务器、不启动新训练、不改数据、冻结协议、检查点或部署。

## 已找到历史MSTCA对比，不能再说从未做过

现有资产包含`assets/models/online_detector_mstca_rfly/`和`assets/models/online_detector_kg_mstca_rfly/`。本轮只读取已有config和聚合metrics，没有运行推理、加载最终测试张量或重新评分。

两者配置声明相同旧协议`rflymad_online_real3_binary_v1`、`mode=fewshot`、31通道、150帧、二分类；训练轮数分别55和90，参数数108338和158891。保存的配置并不足以证明同seed／训练预算／逐样本成员完全匹配，因此称历史模型比较，不称严格控制所有因素的分支消融。

|历史聚合指标|纯MSTCA|旧名KG-MSTCA|
|---|---:|---:|
|val窗口数|5121|5121|
|val准确率|95.20%|96.27%|
|val Macro-F1|0.95179|0.96255|
|历史test准确率（5920窗）|96.01%|97.06%|
|历史external_test准确率（2726窗）|85.95%|85.07%|
|历史external_test Macro-F1|0.82380|0.81871|

指标直接来自各目录`metrics_val.json`、`metrics_test.json`和`metrics_external_test.json`。test／external_test标签是历史文件名，不外推为当前冻结源test／Real角色和口径。样本总数相同不证明身份逐窗相同。旧关系分支的收益不是所有分集一致；不能据此断言新版9项有益或有害。

当前已完成的是新版因果26通道数据上同一双分支模型的采样／预算对照，非纯MSTCA对照。后续先复用并核查既有对比，不把已有实验当不存在；是否补新版实验另行决定，当前不排GPU任务。

## 最近成果与阅读顺序

1. `项目进程.md`：统一状态和决策入口。
2. `docs/research/2026-10-31/dataset_labeltime_closeout.md`、`full_causal_input_v1.md`：标签修订与全量因果输入交付，二者改变范围有别。
3. `docs/research/2026-11-01/causal_source_longbudget_v1.md`：恢复预算后两臂正式成绩、误报／召回取舍和核验。
4. `docs/research/2026-11-01/relation_feature_scope_review.md`：9项启发式特征公式及物理语义局限，不宣称知识图谱。
5. `docs/research/2026-11-01/model_research_direction_v1.md`：待验证研究假设，非已成立创新、非已授权训练。文献条目保留此前记录，本次归档未重新联网查新，不将归档视为文献内容的独立认证。

## 本次GitHub归档边界

纳入近期源码、前后端测试、配置与冻结协议、原创报告／文献阅读笔记，以及轻量结果／完成／核验记录和必要执行脚本。保留原暂存区已有删除和归档整理，不恢复已删除的旧实现。冻结协议及证据源码的字节身份通过Git属性保护；不为了新提交改写历史hash。

不纳入原始日志、缓存、检查点／模型权重新增副本、NPZ逐窗张量、大型传输归档、第三方论文PDF／OCR全文、密钥、凭据、运行时数据库、node_modules及本机配置。这些留在本地；已有受版本控制的历史模型资产不删除。

数据根`D:/UAV-Data`不在Git仓库。GitHub提交不是完整数据／大模型备份；不承诺仅git clone即可在无外部数据或原GPU环境下复现所有实验。研究JSON中的历史绝对路径用于溯源，不是新机器自动可用的路径。

本地验证结果及归档清单见同目录`github_submission_verification.md`。如历史冻结hash测试因当前安全门禁代码与旧协议不同而失败，应如实记录，不能改旧pin或跳过后声称全部通过。
