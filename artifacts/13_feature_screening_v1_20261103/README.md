# 13维候选特征综合筛选分析 v1

> Git归档更新：用户在分析完成后明确要求推送本批报告、源码和轻量证据；原文中的“未提交／不推送”记录分析阶段边界。此次归档不启动M3、不连接训练服务器、不改历史模型或数据。

报告：`docs/research/2026-11-03/13_feature_screening_analysis_v1.md`。

结果：恢复了本地历史候选证据（可见Git历史／分支未找到已提交记录），并只在已有36例响应与12例正常锚点的并集46个源Train/Val案例上补充CPU统计。13维共同full150有效2644窗／原26已准入2890窗，全部2944原候选ID与不可用原因保留。不连接服务器、不用GPU／Test／Real、不训练／加载分类器、不改历史数据／公式／统计／权重，不Git提交或推送。

## 直接交付

- `feature_evidence.csv`：13维公式、来源、单位、因果性、覆盖／尾部、正常／故障证据、冗余、初筛类别和实验角色。
- `feature_correlation.csv`：主Spearman13×13矩阵，Train28个案例各一条median向量；Pearson、配对组折叠、domain／label／观测状态及case内故障相关在results/和summaries/。
- `feature_fault_response.csv`：312条逐故障case×特征结果，固定signed正向AUC、pre／active分布、效应、同观测状态独立正常对照及未匹配窗。
- `feature_normal_stability.csv`：1131条正常case×观测状态×特征分布；状态仅来自过去150帧物理速度／角速度，不拿目录或裁剪进度当线上条件。
- `feature_screening_summary.json`：11项B、O4/O5两项C；A暂空，未证明单维受控增量，B/C非最终删留。
- `M3_controlled_design.json`：只提案不执行。统一adapter与相同额外参数基础重表达控制，首轮8配置；条件追加≤4；Uniform确认另批≤3。启动前必须明确召回容忍界、数据／初始化／缩放及授权。
- `figures_final/`：Spearman热力图和逐case正常climb→low-motion位移，300DPI PNG、SVG和灰度；机器和读图自检已完成。

## 来源与复现

`plan.json`在新分析前固定46案例、full150资格、信号状态阈值与统计口径；`evidence_recovery.json`和`recovered_historical_evidence/recovery_manifest.json`记录历史恢复与原SHA，旧140证据未与新150数据混拼。

主脚本（只CPU，输出必须新目录）：

```bash
python scripts/research/analyze_13_feature_screening.py \
  --plan artifacts/13_feature_screening_v1_20261103/plan.json \
  --data D:/UAV-Data/causal_input_v1 \
  --features D:/UAV-Data/causal_m1_continuous_v1 \
  --historical artifacts/candidate_feature_evidence_v1_20261101 \
  --output <new-output-directory>
```

`summarize_13_feature_screening.py`产生逐case内label相关、每目录等case响应和粗case分布重叠；`build_13_feature_evidence.py`合并数值及明示人工判断；`plot_13_feature_screening.py`只画保存统计。`verify_13_feature_screening.py`独立重建138个代表窗、rank相关、312个AUC、状态匹配与585个状态配对位移，见`independent_verification.json`。57项相关单元／回归通过，未运行历史SHA-pin不兼容的全库测试来改pin凑绿。

`delivery_manifest.json`列出报告／脚本／CSV／JSON／图和恢复副本SHA；原输入身份在results/analysis_completion.json。所有成果目前只在本地；本任务明确禁止自动Git stage／commit／push，普通git status也会因docs/artifacts被ignore而不显示新报告。

## 解释限制

故障仅24个hover案例、每split/type只有SIL/HIL2例，不足以推断同类型强度曲线或所有工况。状态是被故障影响的观测代理，条件匹配可能吸收故障效应；Train气压案例只有3/19、4/26 active可匹配，未匹配不隐藏。Val正常只6个hover，SIL/HIL配对和重叠窗口不独立，不报p值。相关高不等于删维，相关低／单量AUC强不等于MSTCA增量；N2/N3/N4分别保留，未知的机理／泛化／强度结论不补写为事实。
