# AutoDL 模型训练交接文档

本文档覆盖当前主线训练：离线三分类诊断 `normal/motor/sensor`，以及在线 Real3 二分类检测 `normal/fault(motor|sensor)`。目标是把本地仓库和 `D:\UAV-Data` 上传到 AutoDL，完成 split/cache 构建、训练、结果检查和回传。

## 当前状态

代码层面的数据切分已经做好：

- 离线：`rflymad_transfer3_v1`，入口是 `build_transfer3_splits.py`。
- 在线：`rflymad_online_real3_binary_v1`，入口是 `build_online_real3_splits.py`。
- 统一入口：`scripts/train_pipeline.py`。

但新的 split/cache 还没有真实生成。之前只做了 dry-run，验证命令拼接正确，不会写入 `D:\UAV-Data\training_runs`。真正生成 split/cache 和训练，需要在 AutoDL 上执行 `--execute`。

## 本地上传前检查

本地建议保留两个目录：

```text
D:\UAV-Fault-Diagnosis\
D:\UAV-Data\
```

`D:\UAV-Data` 至少需要包含：

```text
UAV-Data/
├─ manifest/
│  └─ manifest_v2.csv
├─ processed_SIL_50hz/
├─ processed_HIL_50hz/
└─ processed_Real_50hz/
```

旧的 `cache_*`、`results_*`、`splits_*` 可以不上传；它们是历史产物。为了减少上传体积，优先上传 manifest 和三个 processed CSV 目录。

如果使用压缩包上传，建议在 Windows PowerShell 中执行：

```powershell
cd D:\
Compress-Archive -Path .\UAV-Fault-Diagnosis -DestinationPath .\uav_fault_repo.zip -Force
Compress-Archive -Path .\UAV-Data\manifest,.\UAV-Data\processed_SIL_50hz,.\UAV-Data\processed_HIL_50hz,.\UAV-Data\processed_Real_50hz -DestinationPath .\uav_data_real3_source.zip -Force
```

也可以用 AutoDL 网盘、Jupyter 上传、`scp`、`rsync` 等方式上传。上传后推荐在 AutoDL 上统一放到：

```text
/root/UAV-Fault-Diagnosis
/root/UAV-Data
```

## AutoDL 环境准备

进入 AutoDL 终端：

```bash
cd /root
unzip -q uav_fault_repo.zip -d /root
unzip -q uav_data_real3_source.zip -d /root/UAV-Data
```

如果解压后多了一层目录，整理成：

```bash
ls /root/UAV-Fault-Diagnosis
ls /root/UAV-Data/manifest/manifest_v2.csv
ls /root/UAV-Data/processed_SIL_50hz | head
ls /root/UAV-Data/processed_HIL_50hz | head
ls /root/UAV-Data/processed_Real_50hz | head
```

安装依赖。AutoDL 镜像通常自带 CUDA 版 PyTorch，如果没有再按镜像 CUDA 版本安装 PyTorch。项目侧依赖可以先装：

```bash
cd /root/UAV-Fault-Diagnosis
python -m pip install -U pip
python -m pip install -r requirements.txt tqdm
python - <<'PY'
import torch, numpy, pandas
print("torch", torch.__version__)
print("cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
PY
```

如果 `torch.cuda.is_available()` 是 `False`，先不要训练，检查 AutoDL 镜像和 PyTorch CUDA 版本。

## 路径处理

`manifest_v2.csv` 里可能记录的是 Windows 路径。不要手工改 manifest，训练入口会自动生成：

```text
/root/UAV-Data/training_runs/<run>/manifest/manifest_v2.local.csv
```

这个 local manifest 会把 `source_csv_path` 转成当前 AutoDL 能访问的 `/root/UAV-Data/...` 路径。

## 先 dry-run

先只打印命令，确认路径和协议：

```bash
cd /root/UAV-Fault-Diagnosis
python -B scripts/train_pipeline.py \
  --task both \
  --data-root /root/UAV-Data \
  --run-name real3_v1
```

检查输出中在线部分应该是：

```text
online/splits real3 binary
online/cache binary fewshot windows
online/train backbone suite
```

并且 cache 命令应包含：

```text
--protocol-name rflymad_online_real3_binary_v1 --label-space real3 --source-fault-classes motor sensor
```

## 一次性执行全流程

确认 dry-run 无误后执行：

```bash
cd /root/UAV-Fault-Diagnosis
python -B scripts/train_pipeline.py \
  --task both \
  --data-root /root/UAV-Data \
  --run-name real3_v1 \
  --execute
```

默认会依次执行：

1. 离线三分类 split。
2. 离线三分类 cache。
3. 离线 `cnn_lstm_v2_normalproto` 训练。
4. 离线结果汇总。
5. 在线 Real3 二分类 split。
6. 在线短窗口 cache。
7. 在线 `kg_mstca/mstca/cnn1d` backbone 训练和汇总。

默认输出目录：

```text
/root/UAV-Data/training_runs/real3_v1/
├─ manifest/manifest_v2.local.csv
├─ offline_transfer3/
│  ├─ splits_transfer3_v1/
│  ├─ cache_transfer3_fewshot_v1/
│  ├─ results/transfer3_cnnlstm_normalproto/
│  └─ summary/
└─ online_real3_binary/
   ├─ splits_online_real3_binary_v1/
   ├─ cache_online_real3_binary_v1/
   ├─ models/
   └─ summary/
```

## 分阶段执行

如果担心一次性运行时间太长，可以分阶段跑。

只生成 split/cache：

```bash
python -B scripts/train_pipeline.py \
  --task both \
  --data-root /root/UAV-Data \
  --run-name real3_v1 \
  --stages splits cache \
  --execute
```

已有 split/cache 后只训练和汇总：

```bash
python -B scripts/train_pipeline.py \
  --task both \
  --data-root /root/UAV-Data \
  --work-root /root/UAV-Data/training_runs/real3_v1 \
  --stages train summary \
  --execute \
  --force
```

只跑在线：

```bash
python -B scripts/train_pipeline.py \
  --task online \
  --data-root /root/UAV-Data \
  --run-name real3_online_v1 \
  --execute
```

只跑离线：

```bash
python -B scripts/train_pipeline.py \
  --task offline \
  --data-root /root/UAV-Data \
  --run-name transfer3_offline_v1 \
  --execute
```

## 训练参数建议

默认参数是保守基线：

```text
offline: epochs=120, batch_size=32, lr=3e-4
online : epochs=100, batch_size=512, lr=3e-4
online models: kg_mstca, mstca, cnn1d
```

AutoDL 显存充足时，可以先保持默认。后续做论文实验再扩展：

```bash
python -B scripts/train_pipeline.py \
  --task online \
  --data-root /root/UAV-Data \
  --run-name real3_online_backbone_full \
  --online-models kg_mstca mstca cnn1d bilstm transformer \
  --online-epochs 120 \
  --execute
```

## 训练时监控

建议开一个终端跑训练，另一个终端监控：

```bash
nvidia-smi -l 5
```

如果想保留日志：

```bash
cd /root/UAV-Fault-Diagnosis
mkdir -p /root/UAV-Data/training_runs/logs
python -B scripts/train_pipeline.py \
  --task both \
  --data-root /root/UAV-Data \
  --run-name real3_v1 \
  --execute 2>&1 | tee /root/UAV-Data/training_runs/logs/real3_v1.log
```

## 结果检查

训练结束后重点检查这些文件。

离线：

```text
/root/UAV-Data/training_runs/real3_v1/offline_transfer3/results/transfer3_cnnlstm_normalproto/
├─ best.pt
├─ config.json
├─ metrics_val.json
├─ metrics_test.json
└─ metrics_external_test.json

/root/UAV-Data/training_runs/real3_v1/offline_transfer3/summary/
├─ transfer3_backbone_summary.csv
└─ transfer3_backbone_ranking.md
```

在线：

```text
/root/UAV-Data/training_runs/real3_v1/online_real3_binary/models/
├─ online_detector_kg_mstca_rfly/
├─ online_detector_mstca_rfly/
└─ online_detector_cnn1d_rfly/

/root/UAV-Data/training_runs/real3_v1/online_real3_binary/summary/
├─ online_backbone_summary.csv
└─ online_backbone_ranking.md
```

快速查看核心指标：

```bash
python - <<'PY'
import json
from pathlib import Path

run = Path("/root/UAV-Data/training_runs/real3_v1")
offline = run / "offline_transfer3/results/transfer3_cnnlstm_normalproto"
print("offline external:", json.loads((offline / "metrics_external_test.json").read_text()))

for model_dir in sorted((run / "online_real3_binary/models").iterdir()):
    if not model_dir.is_dir():
        continue
    m = json.loads((model_dir / "metrics_test.json").read_text())
    print(model_dir.name, {
        "accuracy": m.get("accuracy"),
        "macro_f1": m.get("macro_f1"),
        "f1_fault": m.get("f1_fault"),
        "recall": m.get("recall"),
    })
PY
```

## 回传到本地

建议只下载 run 结果，不下载 cache。cache 可以重建，体积通常较大。

在 AutoDL 上打包：

```bash
cd /root/UAV-Data/training_runs
tar -czf real3_v1_results.tar.gz \
  real3_v1/manifest \
  real3_v1/offline_transfer3/results \
  real3_v1/offline_transfer3/summary \
  real3_v1/online_real3_binary/models \
  real3_v1/online_real3_binary/summary
```

下载到本地后解压到：

```text
D:\UAV-Data\training_runs\real3_v1\
```

也可以用 `scp`：

```powershell
scp root@<AutoDL主机>:/root/UAV-Data/training_runs/real3_v1_results.tar.gz D:\UAV-Data\training_runs\
```

## 更新本地运行资产

先不要直接覆盖 `assets/models`。建议先备份：

```powershell
cd D:\UAV-Fault-Diagnosis
Copy-Item .\assets\models .\assets\models_backup_real3_v1 -Recurse -Force
```

确认指标合格后，再把结果目录中的模型复制到运行资产：

```powershell
Copy-Item D:\UAV-Data\training_runs\real3_v1\offline_transfer3\results\transfer3_cnnlstm_normalproto .\assets\models\transfer3_cnnlstm_normalproto -Recurse -Force
Copy-Item D:\UAV-Data\training_runs\real3_v1\online_real3_binary\models\online_detector_kg_mstca_rfly .\assets\models\online_detector_kg_mstca_rfly -Recurse -Force
Copy-Item D:\UAV-Data\training_runs\real3_v1\online_real3_binary\models\online_detector_mstca_rfly .\assets\models\online_detector_mstca_rfly -Recurse -Force
Copy-Item D:\UAV-Data\training_runs\real3_v1\online_real3_binary\models\online_detector_cnn1d_rfly .\assets\models\online_detector_cnn1d_rfly -Recurse -Force
```

更新后启动 Dashboard 做一次冒烟验证：

```powershell
cd D:\UAV-Fault-Diagnosis
.\apps\dashboard\backend\.venv\Scripts\python.exe -m uav_dashboard_api
```

重点确认：

- 后端启动无模型加载错误。
- 离线诊断能读取 `transfer3_cnnlstm_normalproto`。
- 实时检测能读取三个在线 detector 的 `best.pt/config.json/norm_stats.json`。
- 在线模型的 `config.json` 中包含 `protocol_name=rflymad_online_real3_binary_v1`。

## 常见问题

`manifest missing source_csv_path`：当前流水线需要 manifest 中存在 `source_csv_path`。确认上传的是 `manifest_v2.csv`，不是旧 manifest。

`FileNotFoundError: Case_xxx.csv`：通常是 AutoDL 数据目录结构和 manifest 路径不一致。优先使用 `scripts/train_pipeline.py` 自动生成 local manifest，不要加 `--skip-local-manifest`。

`torch.cuda.is_available() == False`：不要继续训练，先换 AutoDL 镜像或重装 CUDA 版 PyTorch。

`work root is not empty`：说明同名 run 已存在。换 `--run-name`，或者确认要复用后加 `--force`。

在线 summary 缺少 `telemetry_live_eval.csv` 不影响当前训练指标；这个文件是额外实时回放评估产物，不是 backbone 训练必需项。

## 下一步建议

第一轮 AutoDL 先跑默认主线，目标是拿到统一协议下的 baseline：

```bash
python -B scripts/train_pipeline.py --task both --data-root /root/UAV-Data --run-name real3_v1 --execute
```

拿到结果后，再决定是否进入下一轮模型设计：例如在线 KG-MSTCA 的结构消融、离线 CNN-LSTM v2 的对比模型、跨域适配或论文表格需要的重复种子实验。
