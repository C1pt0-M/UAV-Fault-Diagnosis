# M1全量连续特征与训练准入报告v1

2026-11-02（北京时间）。状态：全量源Train／Val连续四维特征、独立核验、Dataset和CPU门禁已完成；四次正式作业已按本轮授权启动，训练／最终响应结果仍在运行，不提前宣布性能改善。技术时序解释见[m1_streaming_technical_review_v1.md](m1_streaming_technical_review_v1.md)，模型与公共预训练比较见[m1_model_and_experiment_protocol_v1.md](m1_model_and_experiment_protocol_v1.md)。

## 1. 连续流改动和保留边界

不是重修数据库：按已冻结源3135案例（train2580＋val555）读取receipt登记ULog及原case shard，除原已无合格监督窗口的案例不救回外，所有原生来源SHA核对。原时间网格20ms，物理公式固定200ms，保留max_age_us、observation／availability、所有提供的reset／device counter、latest-invalid拒绝与内部stale检查。

连续残差按每个原crop时间点计算一次，读取crop前200ms真实已到达历史，不将每个滑窗前10帧重新屏蔽。保留原独立窗口原型源码和旧证据，新增continuous_kinematic_residuals.py；它只去除逐窗prefix重启，不改物理量、坐标、时域或因果有效性。

新目录D:/UAV-Data/causal_m1_continuous_v1：逐案例原N1～N4 float64物理值、逐帧4维valid及2组原因bits、所有原window ID／start/end／基础资格／label、所需原生消息的timestamp_sample／availability、值、valid、source index和counter；invalid_details按topic保存每个无效时间点更细的端点stale／missing／flag／nonfinite、中间gap、reset或非法四元数原因，单独manifest与SHA。

所有原窗口名单保留，新ledgers包含all／included／excluded，历史RflySim后缀仍原本不准入，不因新残差救回。原26数据、norm、M0／M2模型源码／协议／权重未修改。

## 2. 全150帧严格资格：没有退到140

|源集合|原26维合格窗|新共同全150有效窗|新增排除|占原合格|
|---|---:|---:|---:|---:|
|Train|162069|151833|10236|6.3158%|
|Val|34710|32381|2329|6.7099%|
|合计|196779|184214|12565|6.3853%|

原训练分母168800／验证35520仍在all台账（含原来排除的suffix与不可用样本），不补抽、不填0或均值。未把无效点送到forward或BatchNorm。

覆盖不是无偏的：Train正常从83051保留74700、排除8351（约10.06%），故障79018保留77133、排除1885（约2.39%）；Val正常17740保留15883、排除1857（约10.47%），故障16970保留16498、排除472（约2.78%）。按故障目录，barometer约12.28%／12.95%损失高于总体，其余目录、日志工况、phase、域、案例及原因完整保存。

新增4案例原合格窗全部失去资格（不是删除飞行）：train HIL hover magnetometer TestCase_15、22、23，分别1／3／4原合格窗；val HIL waypoint magnetometer TestCase_28原4窗。全量读取没有来源身份／数据值不一致被掩盖的整例失败。

此覆盖损失和类别偏差不能当模型提升。处理方案是保持严格规则、不放宽200ms或年龄门禁，正式比较仅用同一原ID／150时域人口；原M0全覆盖结果另外保留。M0-matched与M1同时使用151833／32381，重算正常来源采样权重，不套旧人口权重。结论只能覆盖共同有效域，不能声称所有原始正常／故障记录均可实时输出。

## 3. 标准化

基础26维norm SHA保持cd61a134842d0322d9e4bc76962ac0a8c0ff90566786c423d3096721cd856a30。

仅source-train、原监督合格、四维全150有效、全窗标签已知且正常的74700窗口用于四维拟合；150帧逐窗贡献共11205000，重叠贡献显式保留，2566案例有贡献（包含故障案例的正常未激活整窗）。Val从未参与拟合。无标准差floor、裁剪或补值，若退化应报错。

|列|mean|population std|
|---|---:|---:|
|N1 rh_signed_m（m）|0.0006571460356322342|0.0478383909123426|
|N2 rR_x_rad（rad）|−0.000008453983789439151|0.002774148992871336|
|N3 rR_y_rad（rad）|−0.00014794400527821613|0.004341781967649567|
|N4 rR_z_rad（rad）|0.00006074784120070256|0.0006533522759711983|

residual_norm_stats.json独立保存上述统计、样本数、标签资格、来源base norm和plan hash。压缩cache只存全150合格float32残差与显式全窗valid、base_compact_index／original_index／y，逐帧无效原值及更细原因仍在case／invalid_details，不只留下过滤后的成功样本。

## 4. 完整性和独立核验

- 3135案例全部sidecar、receipt、导出cache及ledgers的SHA和byte清单已完成。完整manifest SHA：bda2169adaf48946e46596900a683388597079ee13e599b876d0f0fcde6f204e。
- 原窗口ID、时间戳、label、mask逐例重建；独立长精度sum/sumsq复算训练统计，mean最大差3.47e-18、std最大差1.25e-16。
- 所有可复用旧固定案例，共114099个共同有效grid点与原公式数值差0；新的连续prefix改变不被混同为公式变更。
- 真实原生消息到达缓冲回放固定18个源val案例，batch残差／mask对齐，未使用未来值或按每窗重启；独立4例额外直接stream重算通过。
- Dataset读取原CausalCacheDataset，按split＋原ID精确对齐、验证标签／SHA／4维列顺序／全窗mask、有限值及正尺度；M0和M1人口一致，first／middle／last抽查原26归一化张量完全不变。
- 100项新旧相关单元／回归测试通过，涵盖前缀不变、非交换旋转、late arrival、内部stale、reset、重复无效消息、输入形状、invalid在BN前拒绝、参数映射、CPU更新和原告警k_on／merge／flush语义。
- 对真实全部共同人口的读入路径，两次CPU小训练检查（M0-matched及M1各2轮×4批＝8次优化）通过；仅pipeline检查，不报为正式性能，不占额外正式作业。

生成期间发生Windows进度文件replace被短时读取锁拒绝，工作进程仍完成逐例输出；保留故障日志后以同plan/sidecar SHA验证续做，不覆盖已完成case。只对status写入加重试，未改物理计算／mask或旧代码。恢复与生成源码hash在执行证据中记录。

## 5. 启动与计算基本检查，尚非最终故障延迟

固定18源val hover完整原生日志启动可用窗口：M1相对M0输入就绪延迟median0.200s、P90约3.038s、max3.360s。原始消息尚未齐、年龄／reset中断＋全150连续有效窗口可使实际启动晚于理想2.98s／3.18s；不能把200ms当全部启动或告警耗时。初次仅看crop起点后短段的有界观测保留，最终full-log结果无截尾，未用未标注前后日志训练或计故障分数。

原生fixed18回放残差计算P90约0.636–0.777ms，检查平台上无单次超过20ms。另CPU合成前向50次的median约M0 0.513ms、M1 0.658ms，仅模型形状／开销可行性，不是trained replay或实机端到端证明。

正式训练后连续推理会记录每帧残差wall time、预处理／设备传输／模型推理、实际更新间隔、串行20ms预算积压，0.5首次越界、真实部署0.6／0.4、连续3点、min_event0.5s／merge0.5s的候选／确认／最终输出。缺口断开不可硬连。GTD和ULog command各自作为注入记录参考、时间差披露，不按窗口标签猜物理损伤真值；漏检／来源未知／截尾分别保留。

## 6. 正式执行状态

全部门禁receipt PASS_ALL_M1_PRETRAINING_GATES，协议SHA6f48ef8ad0f36d83fe2e69898efb66730f05a9434d44fdd5180120ef0c7a7fd6。autodl_UAV3匹配torch2.5.1+cu124／NumPy2.1.3／CUDA12.4；独立code/features工作区，压缩派生输入归档SHA、人口、公共M0最佳预训练state、初始两模型eval函数一致均再次核验。

北京时间03:17:55（UTC19:17:55）启动有限序列：M0_matched Uniform→M1 Uniform→M0_matched Balanced→M1 Balanced，seed42、100轮上限和原训练超参数／strict best规则不变。公共参数两个变体都从对应arm M0 best开始，新优化器不续旧状态；不是从头或未见val实验。最多4次，失败不重跑，不新增seed／跨度。完成后独立结果核验、固定18源val连续回放已排队，未额外训练。

当前没有正式M1结果或“已降低误报／保留召回／告警未变慢”结论。训练／回放进度可运行python artifacts/causal_m1_continuous_v1_20261102/poll.py查看。没有自动部署／关机，不触碰最终test／Real。GitHub只归档源码、协议、原创报告和轻量manifest／结果，权重、NPZ、输入归档及原始日志不上传；后续提交号另记。
