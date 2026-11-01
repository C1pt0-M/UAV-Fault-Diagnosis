# 正式源域正常来源采样对照 v1

本任务只比较采样，不更换网络，不做Real适配或最终测试。配置已冻结：

- `configs/experiments/normal_source_sampling_train_v1.json`
- SHA-256：`088bbb0c1ce2d831321304322dab35491b2fdab1dab9fb05aef2dae85d1fcf23`
- 启动：`scripts/experiments/run_source_normal_sampling.sh`
- 单任务／预检／汇总：`scripts/research/train_source_normal_sampling.py`

## 固定设置

| 项目 | 设置 |
|---|---|
| 实验臂 | 等权有放回采样／两类SIL-HIL正常来源等权采样 |
| 种子 | 42、43、44，各两臂，共6任务，顺序执行 |
| 初始化 | 每种子重新随机初始化；同种子的两臂初始状态摘要必须相同，不加载旧权重 |
| 模型 | KG-MSTCA旧结构名，26通道、150帧、156491参数；不是知识图谱 |
| 每轮训练 | 有放回抽168800窗口，batch512，num_workers=0 |
| 优化 | AdamW，lr=3e-4，weight_decay=1e-4，FocalLoss gamma=2，梯度裁剪1 |
| 训练长度 | 最多100轮，5轮预热＋原有余弦退火函数，最低lr参数1e-6 |
| 增强／精度 | 不做数据增强；float32，不使用AMP；确定性算法，cuDNN TF32开、矩阵乘TF32关 |
| 每轮验证 | 完整源验证35520窗口，不再使用CPU检查的2220子集 |
| 最佳权重 | 整体源验证Macro-F1严格最大，完全相同时保留最早轮；任意小改善均保存 |
| 早停 | 独立显著改善参考，delta=0.002；从第30轮起累计不改善轮次，20次停止 |

选模在两臂中完全相同，以隔离采样作用；不根据某臂结果再换指标。必须同时报告原无故障、故障前、正常各工况、原无故障悬停、故障类别指标，不能用总体F1掩盖短板。

两臂的预期正常／故障与RflySim正常概率保持不变，仅把两类SIL/HIL正常池的合计83350权重等分为各41675。有限抽样数量会波动；这不是历史无放回训练的精确重放。

## 冻结包与输入

最小包根目录包含`SHA256SUMS`，其中有代码、配置、使用说明及`inputs/source_sampling/`下的3份源索引文件。不包含任何数据张量、原CSV、Real预测、部署资产或认证信息。

服务器已存在的缓存路径：`/root/autodl-tmp/online_cache_v2/`。只读取`train.pt`、`val.pt`、`norm_stats.json`，逐一核对冻结哈希，不读取同目录中的其他测试缓存。

推荐解压到独立目录，不覆盖`normal_adapt_v1`、旧`runs`或旧Python包：

```bash
mkdir -p /root/autodl-tmp/experiments
tar -xzf source_normal_sampling_train_v1_20261030.tar.gz -C /root/autodl-tmp/experiments
cd /root/autodl-tmp/experiments/source_normal_sampling_v1
sha256sum --check --strict SHA256SUMS
```

如果目标目录已有旧版本，先选择新位置，不能用解压覆盖它。包内Python文件按自身位置定位`packages/uav_fault`，无需安装整个项目。

## 服务器依赖：先检查，不自动安装

此前服务器解释器已核实为Python 3.12.3；正式库版本固定为Torch `2.5.1+cu124`／CUDA `12.4`、NumPy `2.1.3`、pandas `2.3.1`，每次记录实际Python版本。GPU型号记录在产物中，不限死RTX型号。

上一轮远端缺少pandas；服务器重新开启后须先核实。不要执行项目整体`pip install -e .`或盲目更新Torch。本最小实验包使用独立实验环境约定，不更改项目应用依赖范围。确需补齐时可执行：

```bash
/root/miniconda3/bin/python -m pip install 'numpy==2.1.3' 'pandas==2.3.1'
```

这不是启动脚本的自动行为；用户确认后手动安装。既有CUDA Torch应保留；若Torch版本不符，先解决环境，不允许静默换用CPU。启动脚本在创建训练根目录前验证依赖版本、CUDA及包完整性。

## 启动GPU任务

前台命令（断开会话可能中止）：

```bash
SOURCE_RUN_ROOT=/root/autodl-tmp/experiments/source_normal_sampling_v1/runs/registered_v1 \
  bash scripts/experiments/run_source_normal_sampling.sh
```

生产运行宜从不会随SSH退出终止的会话或受控后台进程启动，并记录PID。使用nohup的示例（日志名和运行根目录都必须未被占用）：

```bash
mkdir -p /root/autodl-tmp/experiments/source_normal_sampling_launch_logs
(
  set -o noclobber
  SOURCE_RUN_ROOT=/root/autodl-tmp/experiments/source_normal_sampling_v1/runs/registered_v1 \
    nohup bash scripts/experiments/run_source_normal_sampling.sh \
    > /root/autodl-tmp/experiments/source_normal_sampling_launch_logs/registered_v1.log 2>&1 < /dev/null &
  echo "launcher PID: $!"
)
```

可修改的环境变量仅为`PYTHON_BIN`、`SOURCE_CACHE_DIR`、`SOURCE_AUDIT_DIR`、`SOURCE_RUN_ROOT`等运行位置；不接受超参数、种子或实验臂覆盖。已存在运行根目录直接拒绝。默认解释器为已验证的`/root/miniconda3/bin/python`，不依赖非交互SSH的`python3`命令。

六任务全部完成后自动生成`summary.json`，包含三个配对种子的逐种子数值、均值、样本标准差及“均衡−等权”配对差。不能只挑最好种子。CPU检查、缺失任务、非完整源验证、不同代码包或配对初始状态不一致均拒绝汇总。

## 产物与中断边界

每任务位于`seed_<seed>/<arm>/`，保存`initial.pt`、原始`best.pt/last.pt`、`config.json`、逐轮`history.json`、`report.json`、源归一化及代码快照、最佳模型完整源验证概率。每轮原子更新权重和历史，不修改原数据或部署目录。

**不支持优化器／RNG断点恢复。**失败保留全部产物，脚本fail-fast，不覆盖重跑。可用同配置在新目录显式重跑一个注册任务：

```bash
/root/miniconda3/bin/python -u scripts/research/train_source_normal_sampling.py run \
  --protocol configs/experiments/normal_source_sampling_train_v1.json \
  --cache /root/autodl-tmp/online_cache_v2 --audit inputs/source_sampling \
  --seed 42 --arm balanced_source_normal --output /root/autodl-tmp/new_run/seed_42/balanced_source_normal
```

重跑后要汇总，六个完成任务必须保持相同逻辑目录布局和冻结身份；可在新的根目录建立指向已完成任务的符号链接，不覆盖原失败任务。所有任务仍须来自同代码包和配置。

配置的`protocol_name`是实验研究协议；模型`config.json`的`protocol_name`继续匹配源归一化输入合同，另用`experiment_protocol_name`记录本实验，避免错误声明导致模型合同冲突。这不代表通过发布验收。

## 已验证与尚未验证

已在本地对正式入口做数据／源码预检、两臂seed42的CPU受限检查（各2小轮次、128抽样/轮、2220验证窗口）、最佳保存重载、初始化配对、选模／早停逻辑、三种子汇总拒绝边界，以及Bash语法检查。

本地CPU检查使用Torch 2.8.0+cpu，不属于正式环境；其状态为`COMPLETED_CPU_CHECK_NOT_FORMAL`，不能混入正式汇总。真正的CUDA执行、远端pandas补齐及吞吐／显存尚未验收；本次没有连接服务器或启动GPU训练。所有源验证结果都是开发证据，不是Real泛化、事件实时能力或自动部署依据。
