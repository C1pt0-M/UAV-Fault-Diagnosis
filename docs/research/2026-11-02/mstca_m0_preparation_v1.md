# M0纯26通道MSTCA：适配与新服务器预检

> 后续执行更新：用户已明确“确认，开始吧”，北京时间2026-11-02 00:55:41启动两臂正式M0序列（UTC2026-11-01 16:55:41）。Uniform已完成首轮317次梯度更新，Balanced排队顺序执行。下文是此前准备阶段记录，不能将其中“未训练”误当当前状态；训练结果尚未完成，详见[正式训练记录](mstca_m0_training_v1.md)。

北京时间2026-11-02（远端核验UTC2026-11-01T16:47:58）。状态：准备与预检通过，**没有开始正式训练、没有优化器更新**。用户此前要求先检查训练入口与最新因果数据，本轮按该边界连接新服务器并核验，不把连接成功等同自动训练授权。

## 1. 连接问题已定位并解决

- 先前autodl_UAV2连接返回Permission denied，不能凭这个错误认定新密钥本身有问题。
- 用户后来给出的ssh UAV3中，UAV3在当前SSH配置里没有Host项，因此被当作主机名解析，尚未进入密钥认证。
- 当前配置真正的别名是**autodl_UAV3**，使用配置所指的新服务器与密钥；该完整别名已连接成功。未读取私钥、展示凭据或修改SSH配置。
- 后续执行统一用ssh autodl_UAV3。冻结M0准备协议里的autodl_UAV2／认证阻塞只是当时历史状态，不为更新主机名而改已锁协议；新执行位置和就绪状态由独立receipt记录。

## 2. 环境与克隆数据核对

|项目|实际核验|
|---|---|
|GPU|NVIDIA GeForce RTX4090 D，24564MiB；检查时0MiB／0%|
|PyTorch|2.5.1+cu124|
|NumPy|2.1.3|
|torch CUDA|12.4，CUDA available=True|
|Python|3.12.3，Linux|
|数据版本|rflymad_causal_publication_replay_v1|
|Train|162069窗，2570案例|
|Val|34710窗，552案例|
|输入|26通道×150帧，基础26通道统计保持不变|

数据位置沿用克隆的/root/autodl-tmp/causal_source_v1_20261031/unpacked/data。逐文件核对build_manifest、input_contract、norm_stats、两份cache_complete、cache/train.pt、cache/val.pt和两份included metadata CSV，全部SHA与原恢复长预算协议相同。cache目录仅有train.pt和val.pt，未读取最终test／Real张量或台账。

CausalCacheDataset预检已实际执行，标签／原索引／mask、train与val案例及SIL-HIL配对组隔离均通过。两种采样权重重新由同一来源构成生成：总正常概率约51.2442%、故障约48.7558%；均衡采样的两种正常来源各约25.6221%。未改样本成员或类权重。

## 3. M0适配内容

新增入口scripts/research/train_causal_mstca_m0.py，复用现有OnlineMSTCA(in_channels=26,hidden_size=64,num_layers=1,num_classes=2,dropout=0.2)，参数量**105938**。仅encoder和head，无relation分支、gate或新增运动学残差。模型实现lstm.py、原训练脚本、冻结旧配置和旧结果均不改。

复用原prepare／元数据、采样、FocalLoss、优化器、warmup-cosine、selection_step、分层指标与checkpoint／预测保存口径；新增独立协议configs/experiments/causal_pure_mstca_m0_v1.json及SHA锁，不覆盖历史协议。

拟议两次正式实验均seed42：uniform_replacement及balanced_source_normal。max100、warmup5、batch512、AdamW lr3e-4、wd1e-4、min_lr1e-6、Focal gamma2／alphaNone、clip1、无AMP／增强，min30／patience20；严格最大完整源val Macro-F1选best、同分最早，固定0.5口径。不用hover局部、最终test或Real选模型。准备协议不授权启动，run还要求显式--confirm-formal-run并通过CUDA／版本检查。

## 4. 初始化不是只说seed相同

以历史恢复预算保存的**未训练initial.pt**作精确对齐锚点，不使用best.pt／last.pt。锚点SHA和state digest已核，远端在相同正式环境重新seed42构造完整旧模型，所有初始张量与锚点逐元素一致，确认不是训练后的权重。旧类仅在初始化核验时构造，不调用关系特征或分类前向，也不是M0训练模型。

映射raw_encoder.*→encoder.*、head.*→head.*，共41个state张量（含BN buffer）精确一致，两种M0采样运行将从同一状态开始。整模型结构／参数量不同，不声称整模型初始张量完全相同；相同公共初始化不等于优化过程或dropout随机掩码逐步相同。

- 历史初始state digest：7772ccc1a20ce8b6d97c7ce83fc4927a8ff9a5dc7bef17038ca767806e561c55。
- M0映射后digest：a68551099cba00575f1eec6edc31ccf47b39ca8e0fa204ae2fe065d3400be646。
- M0准备协议SHA：6ef47aead15fea51e4c7d4b53289383ecd038d67b82e3cae8c07ddcd8b450fe6。

## 5. 核验与缺少的测试依赖

本地相关测试59项通过，包括纯模型结构与26输入、协议／代码hash门禁、预算不被重锁绕过、未授权／无CUDA／环境错误前置拒绝、初始化精确映射、两臂一致和受更新锚点拒绝。

远端不含pytest，首次尝试报告No module named pytest；**没有安装依赖，不能声称远端pytest套件通过**。改运行独立verify_remote_preparation.py断言：

- 完整源预检和全部冻结代码pin、原历史协议SHA符合；
- 匹配正式环境的fresh-seed锚点复现与41张量映射通过；
- 两个M0构造实例state精确一致，只有encoder／head；
- 禁用旧_relation_features仍能执行零输入2×26×150的CPU合成前向，结果为有限2×2，参数和BN状态前后digest不变；
- 准备模式尝试run被授权guard拒绝，无run目录、无优化器更新；
- 不读取真实样本进行模型推理，不读最终test／Real、不改原实验。

远端独立receipt状态PASS_REMOTE_M0_PREPARATION_ONLY_AWAITING_TRAINING_AUTHORIZATION。原冻结M0入口的preflight状态字符串写成PASS_LOCAL...，在远端也沿用该字符串；本轮不改已pin源码，由独立receipt核实实际运行主机与环境，避免靠改文字凑通过。

## 6. 保存位置与下一步

- 独立远端工作区：/root/autodl-tmp/causal_pure_mstca_m0_v1_20261102/。复制冻结源码并只追加M0入口、测试、配置和必要基础模块，不覆盖原克隆实验。
- 本地记录仍归artifacts/causal_pure_mstca_m0_v1_20261101/（沿用准备阶段目录，不因北京时间跨日重命名历史资产）。
- remote_uav3_clone_identity_verification.json：克隆来源核对。
- remote_uav3_staging.json：新工作区／当前SSH别名执行记录。
- remote_uav3_results/：远端预检、初始化、采样及独立核验原文，按SHA取回；m0_tests.log保留缺pytest记录。
- verify_remote_preparation.py：最小远端独立检查源码，无训练调用。

M0已准备好；后续须确认正式执行，再顺序运行两种采样。不扩大seed／架构，不新增训练数据或阈值扫描，不自动安装包、发布部署、提交或推送。此次没有训练成绩可报告，GPU状态只代表检查时刻，不视为未来占用保证。
