# M1：连续200ms四维运动学残差实验

状态：全量特征和GPU前门禁完成，四个授权正式作业运行中；正式成绩和最终故障响应回放尚未完成。

## 已完成的交付

- 原生消息连续计算N1～N4，不再每150帧滑窗重置前10帧。固定200ms、20ms网格、原始物理单位和所有原年龄／reset／因果门禁。
- 冻结3135源train/val案例，原cap64及原窗口ID不变；严格全150有效、不填0／均值／补抽。Train151833/162069，Val32381/34710共同合格，覆盖和类别偏差完整报告。
- source train正常74700窗、11205000重叠帧贡献独立拟合四维统计；基础26通道统计不变。
- 全量manifest及每文件SHA，逐位置无效原因及原生时钟／counter保留在本地新数据根D:/UAV-Data/causal_m1_continuous_v1。GitHub只保留清单、统计、协议和轻量证据，不含其NPZ／PT本体。
- 独立核对全部窗口ID／mask／label、train统计、Dataset及旧公式114099共同有效点（数值差0）。100项相关测试、实数据CPU两模型各8次更新和18例原生回放基本检查通过。
- 完整18日志启动额外输入准备延迟median0.2s、P90约3.038s、max3.36s；这不是故障告警延迟，更不能写成实机端到端认证。

## 模型与有限实验

M1为独立OnlineMSTCAKinematicM1：未改原26维MSTCA骨干；4维Conv4→16(k3)→ReLU→Conv16→16(k3)→ReLU→pool→Linear16→128后，在基础池化特征处相加一次，再用原分类头。新增3168参数，总109106；无旧9维、无KG门控、非简单30维原始拼接。新增projection初始化0，公共参数从对应arm的M0最佳预训练权重加载，两个变体完全相同起点，不偏给M1预训练优势。

固定4作业：M0-matched Uniform→M1 Uniform→M0-matched Balanced→M1 Balanced，seed42、M0原100轮上限／warmup5／min30／patience20／完整共同val Macro-F1严格best／0.5分类阈值不变。不新增种子、跨度或结构；失败不自动重跑。

这是同公共预训练起点的继续训练比较，不是从头或未见val实验；共同人口变化和预训练范围、容量差异均披露。M0及M2旧资产保持原样。

## 查看和后续

运行只读查询：python artifacts/causal_m1_continuous_v1_20261102/poll.py。远端autodl_UAV3:/root/autodl-tmp/causal_m1_continuous_v1_20261102。

execute_formal.py限定4次；postrun_verify_replay.py等待结束后仅执行结果核验及固定18source-val回放，不额外训练。原事件0.6／0.4、k_on/off3、min0.5s／merge0.5s保留，0.5模型越界、候选、确认、最终事件发出／flush分开。GTD及ULog command是不同注入记录参考，不从窗口label猜物理损伤起点。

结果完成后需再按SHA取回复算和更新报告／Git提交。当前不宣称悬停误报改善、故障召回保持或实际告警不变慢。没有最终test／Real使用、服务器关机或部署。

## 公开与本地资产区别

build_manifest.json和invalid_details_manifest.json列出完整本地数据SHA，不等于其中二进制被上传。source_m1_execution.tar.gz、fixed_source_val_replay.tar.gz、cache/*.pt、cases/*.npz、CPU／正式模型权重和预测NPZ按既有策略留本地／服务器。Git只有代码、测试、原创报告、协议、清单及轻量JSON。publication_manifest.json（若已存在）才对应具体公开文件集合。
