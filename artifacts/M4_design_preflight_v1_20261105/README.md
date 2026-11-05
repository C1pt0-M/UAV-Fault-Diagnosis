# M4设计可行性静态资产 v1

本目录仅为source Train静态统计和未来研究协议草案；结论B：缩小可主张范围，暂缓正式M4训练。

[完整报告](../../docs/research/M4专项研究设计与数据可行性报告_v1_2026-11-05.md)包含数据口径、分组缺口、M0监督暴露修正、U/C两个方向及CF保护消融。没有M4实现、预测或性能。`future_protocol_draft.json`明确不授权启动。

## 数据与执行范围

- 只读六项冻结source Train文件，列表和SHA见[analysis_plan.json](analysis_plan.json)、[input_identity.json](input_identity.json)。`cache/train.pt`是数据张量，不是模型checkpoint。
- 162069合格窗、2570案例、1296原关联组；全部原ID保留，不受N残差有效性限制；历史原source2580案例中10个无合格窗。
- 83051整窗正常、79018故障端点；其中73098整窗纯故障用于描述。正常参考拟合资格与故障端点监督分母分开。
- CPU既有环境：Python3.12.4 / NumPy1.26.4 / pandas2.2.3 / torch2.5.1+cpu；`CUDA_VISIBLE_DEVICES=''`，`python -B`。
- 无模型构造、权重加载、前向、训练、GPU、服务器、Test/Real数据读取、新Val数据分析或依赖安装。
- 不修改源数据/缓存/统计、模型、历史配置/SHA锁和其他历史资产；不commit/push。

## 结果入口

|文件|内容|
|---|---|
|`completion.json`|全部人口、排除案例、状态支持及5/3折摘要|
|`window_state_ledger.csv.gz`|162069个原始ID、案例/组/域/目录、标签/正常/纯故障资格、状态与5项物理摘要|
|`case_identity_and_availability.csv` / `association_groups.csv`|2580原案例的可用性、1296冻结关联组，不重新定义配对|
|`coverage_*.csv`|域、故障目录、运行目录、状态及交叉人口；各行案例可重复，不能直接相加|
|`normal_state_support.csv`|固定状态正常支持：五状态通过工程门槛、descent为0|
|`case_state_feature_summaries.csv.gz`|案例×状态×完整已知标签的31物理摘要；不是神经latent|
|`normal_case_state_distributions.csv`|正常物理案例分布、按域和状态分层|
|`conditional_case_AUC_descriptive.csv`|同状态/域，完整故障对独立正常的固定方向案例AUC；不翻号、不调阈值、不推断因果或增量|
|`known_normal_case_dynamics.csv`|正常案例中的状态变化；不是连续路由抖动统计|
|`independent_normal_climb_low_case_pairs.csv`|329案例/168组，同案例low−climb物理摘要差值|
|`case_pre_post_state_pairs.csv` / `pre_post_*.csv`|2198案例/1127组的固定最近正常/最早纯故障配对；间隔保留，不是连续事件延迟或因果归因|
|`fold_assignments_{5,3}.csv`|冻结组的确定性外层划分，所有原ID恰好一次留出|
|`fold_populations_*` / `fold_*support_*` / `fold_state_summary_*`|外层人口、六故障×域和状态细分门槛；5折167/360、3折87/216细分格不足|
|`independent_normal_state_domain_flight_support.csv` / `fold_hover_priority_support.csv`|独立正常支持、Hover低运动重点支持缺口|
|`nested_design_plan.json` / `nested_group_assignments_3.csv`|三折内部开发身份草案；没有训练|
|`nested_populations_3.csv` / `nested_*support_3.csv` / `nested_design_verification.json`|内层人口和计数；案例/组隔离，五状态拟合正常支持，下降无支持|
|`asset_exposure_inventory.json`|历史M0/物理Train摘要资产检查范围和监督暴露限制|
|`future_protocol_draft.json`|未批准、未执行的四组公平协议及待定项|
|`initial_workspace_snapshot.json` / `delivery_verification.json`|原工作区修改保护及最终交付核查|
|`publication_manifest.json`|本次所有交付文件（不含清单自身）的字节/SHA；包含报告SHA，不修改历史SHA锁|

约13MB压缩逐窗/案例表用于ID溯源，其余为轻量CSV/JSON/脚本，没有复制2.53GB原缓存、权重或原始遥测。是否将这些本地ignored资产纳入Git发布由用户另行决定。

## 方法和复核

1. `analysis_plan.json`在读取Train张量统计前写入；已有observed_motion边界不变。`analyze_train.py`只AST抽取原函数，生成全人口物理摘要和身份/支持表；951次标量/向量对照。`state_rule_identity.json`记录函数SHA。
2. `supplement_support.py`只读新台账，补独立正常和Hover支持，不读源张量或模型。
3. `verify_preflight.py`独立读取允许Train与本次表，核对全部ID/标签/总数、7708个案例首中尾物理探针、fold无跨组、全部2198前后配对，并直接两两比较100个案例AUC。结果[PASS](independent_verification.json)，最大AUC舍入差1.11e-16；输入六文件SHA前后相同。核验描述计算，不验证诊断有效性。
4. 嵌套补充`extend_design_metadata.py`第一次执行写出四个新CSV后，因末尾配对读取错用`qualified_y`（实际为`known_label`）失败。保留原脚本及`metadata_extension_failure_v1.json`，未删除或覆盖表。`design_metadata_reproduce_v2.py`修正字段，重新构造并核对四表及329配对，输出`nested_design_verification.json`；状态PASS。**不要再运行失败v1脚本。**
5. `verify_delivery.py`仅检查允许Train文件SHA、既有保护文件、Git状态、新报告链接/结构、JSON/新脚本语法及关键表数值；不运行研究模型。

原始统计脚本使用exclusive-create，已完成目录不直接重跑。若复算主分析，应先创建独立新版本目录，复制原计划与脚本，保持`ROOT=OUT.parents[1]`的仓库相对位置；比较输出内容而非gzip压缩头时间戳。**本任务没有授权重新加工历史数据。**

当前已有独立核验可直接阅读；下列命令只是复现说明。本目录核验JSON也使用exclusive-create，若重新执行应先在独立新目录中输出，避免覆盖现有记录：

```powershell
$env:CUDA_VISIBLE_DEVICES=''
# 使用既有环境，不安装依赖；输出应另选全新目录。
& .\apps\dashboard\backend\.venv\Scripts\python.exe -B <新目录>\verify_preflight.py
```

主分析读取`train_all_windows.csv`曾出现pandas混合列DtypeWarning；所用身份字段明确按字符串读取，所有数值状态/标签来自合格台账与原Train张量，混合的未用列未进入计算。后续全ID/标签核验通过；没有因该警告改写源CSV或补值。

## 结论边界

“状态分布不同”“可可靠观测工况”“诊断有增益”是三件事。当前只支持第一类描述和支持/风险审计。历史完整Train监督M0上的留组参考仍属训练内分析；真正验证需未来各fold独立编码器和专属统计。本目录不包含这种训练或其结果。

5/3折整体六故障×域支持可行，但全状态细分、下降正常参考及部分Hover故障仍不足；3折不是全覆盖证明。当前协议的连续条件局部支持、故障吸收、绝对R*和正式训练批准仍待决定，不能自动启动。
