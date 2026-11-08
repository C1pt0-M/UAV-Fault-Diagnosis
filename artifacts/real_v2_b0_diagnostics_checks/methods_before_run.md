# B0执行前实现口径（补充1已收到，尚未打开Real npz）

- 只用认证split中的dev且非quarantine案例；320去5后315。台账按chunk读入后的第一操作就是按该UID集合筛行，再做任何校验或计算。输入审计白名单拒绝sealed/quarantine/source/internal-test payload和输入写入；逐一记录实际npz文件。
- 只复制源stats函数，不import带顶层训练/test访问的run_test.py；合成测试用AST单独提取该函数作逐元素等值验证，同时核对GBT参数。运行不import Torch/MSTCA或CausalCacheDataset。
- gate只加载eligible adapt/calibrate normal窗口，S只predict，不fit。每例窗口数及>=0.5计数完全一致后才加载其余案例和训练诊断模型。每个窗口shard只打开一次，按original_cache_index与qualified_y/window_start_row严格对齐。
- session每训练折用sklearn compute_sample_weight('balanced')，两类权重总量相等、总量为训练窗口数。R每训练折故障权重1，正常总量为训练正常数，两种normal_origin各一半。均只用训练折计算权重；固定400/.05/31/1.0/random_state0，其他sklearn默认（含默认early_stopping）。仅控制native CPU线程上限16，不搜索任何配方。
- normal_full按family normal且旧角色test_normal/adapt_normal/calibrate_normal识别，supplemental按normal且非上述角色识别；不用stratum_family代替案例性质，因补充正常可能与motor案例同组。FPR五个命名正常总体只作注册子集；若有不在这些子集的qualified_y0，保留在全部正常AUC/拟合里并披露数量，不改phase/标签/排除。
- 按补充1澄清：session只做主任务5折；远/近起点分析均使用同一主任务OOF，不重训。远端负类end<=onset-250；近端是pre_fault负类补集。正类不变；近端案例数指至少含一个被筛窗口的案例，可与远端案例重叠。
- B0-2 motor/sensor只限制正类，正常侧保持pooled定义不变。S是冻结源模型分数，不称为Real OOF；R为5折OOF。
- 所有AUC按窗口合并点估计。预计算排序/同分段，bootstrap按整关联组重复计数，与sklearn带sample_weight的AUC等价（合成测试覆盖ties）。所有AUC共用315个eligible案例的真实group抽样计划，2000次rng20261108；飞行目录及近端描述也给CI。任一重复缺正/负类记null，不丢弃/重抽/加伪计数；该项CI记不可估并报告未定义数。
- AP只按要求给(a)点值，不额外bootstrap；(c)-(b)给点差。FPR在pooled故障分数上取ceil(recall*N)的降序值（满足召回的最高阈值），>=包含ties，同时给实际召回；空正常总体FPR=null而不是0。这些阈值仅描述，不选/写入部署工作点。
- 曝光仅打开完整正常和故障案例的cases npz，仅取row_input_valid/row_label_valid；正常完整valid曝光，故障前取row index<onset。-1 onset是缺失而非负时长；缺失单列、不造0。输入异常不补值。分布给family、flight及其交叉，所有case/phase/label明细保留CSV。
- 模型仅在内存生成每折预测后销毁，10个诊断fit（session5、R5）；不保存部署模型，不启动B1。GO/STOP仅建议，实际后续需用户决定。
