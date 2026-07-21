# AutoDL 最小训练包交接文档

本文档只覆盖模型训练，不覆盖 Dashboard、前端、后端服务和运行资产。因此不需要上传整个项目，也不需要安装项目总依赖。AutoDL 上只需要一个最小训练包、RflyMAD 处理后的 CSV 数据、以及训练必需 Python 包。

## 当前状态

代码层面的数据切分已经完成：

- 离线三分类：`normal/motor/sensor`，协议 `rflymad_transfer3_v1`。
- 在线二分类：`normal/fault`，但数据范围压缩到 `normal/motor/sensor`，协议 `rflymad_online_real3_binary_v1`。
- 统一入口：`scripts/train_pipeline.py`。

新的 split/cache 还没有真实生成。之前只做了 dry-run，所以 AutoDL 上第一次执行 `--execute` 时会生成新的 split、cache 和训练结果。

## 需要上传什么

只上传下面两类内容。

训练代码最小包：

```text
uav_fault_train_min/
├─ requirements-autodl-train.txt
├─ scripts/
│  ├─ train_pipeline.py
│  └─ eval/
│     ├─ run_online_backbone_suite.py
│     ├─ compare_online_backbones.py
│     └─ compare_transfer3_backbones.py
└─ packages/
   └─ uav_fault/
      └─ uav_fault/
         ├─ __init__.py
         ├─ data/
         │  ├─ build_transfer3_splits.py
         │  ├─ build_sequence_cache.py
         │  ├─ build_sequence_cache_3class.py
         │  ├─ build_online_real3_splits.py
         │  └─ build_online_cache.py
         └─ models/
            ├─ train_transfer3_normalproto.py
            ├─ train_online_detector.py
            └─ cnn/
               ├─ __init__.py
               ├─ cnn_lstm_v2.py
               ├─ cnn_lstm_v2_normalproto.py
               ├─ focal_loss.py
               └─ lstm.py
```

训练数据：

```text
UAV-Data/
├─ manifest/
│  └─ manifest_v2.csv
├─ processed_SIL_50hz/
├─ processed_HIL_50hz/
├─ processed_Real_50hz/
└─ extra_normals/
   └─ nf_buaa_case_189_1419000188.csv
```

`extra_normals/` 只用于在线二分类的 normal domain alignment。离线三分类不使用它。本轮要解决 RflySim/Buaa No Fault 实时误报，因此建议带上这个目录。

不需要上传：

- `apps/`
- `assets/`
- `docs/`
- `knowledge_base/`
- `dashboard/`
- 前端 `node_modules`
- 历史 `cache_* / results_* / splits_*`
- 旧实验材料

## 本地打包

在本地手动建一个目录，目录名建议为 `uav_fault_train_min`。只把上面“训练代码最小包”清单里的文件按相同相对路径放进去，然后自己压缩成：

```text
D:\uav_fault_train_min.zip
```

压缩包打开后，第一层应该直接看到：

```text
requirements-autodl-train.txt
scripts/
packages/
```

不要让压缩包解开后变成：

```text
uav_fault_train_min/uav_fault_train_min/scripts/...
```

数据你已经上传到 AutoDL 的 `/root/UAV`，无需重复上传。

## 用 scp 上传

本项目训练文件统一用 `scp` 上传。把 AutoDL 的 SSH 信息替换进去，端口参数是大写 `-P`。

数据已经上传后，只需要上传训练代码最小包。你当前 AutoDL SSH 信息是：

```text
ssh -p 50682 root@connect.westd.seetacloud.com
```

因此本地 PowerShell 上传命令是：

```powershell
scp -P 50682 D:\uav_fault_train_min.zip root@connect.westd.seetacloud.com:/root/
```

如果 AutoDL 上还没有 Buaa/RflySim No Fault extra normal，同步上传：

```powershell
ssh -p 50682 root@connect.westd.seetacloud.com "mkdir -p /root/UAV/extra_normals"
scp -P 50682 D:\UAV-Fault-Diagnosis\runtime\extra_normals\nf_buaa_case_189_1419000188.csv root@connect.westd.seetacloud.com:/root/UAV/extra_normals/
```

## AutoDL 解压

在 AutoDL 终端执行：

```bash
cd /root
rm -rf /root/uav_fault_train_min
mkdir -p /root/uav_fault_train_min
unzip -q /root/uav_fault_train_min.zip -d /root/uav_fault_train_min
```

检查结构：

```bash
ls /root/uav_fault_train_min/scripts/train_pipeline.py
ls /root/uav_fault_train_min/packages/uav_fault/uav_fault/data/build_online_real3_splits.py
ls /root/uav_fault_train_min/requirements-autodl-train.txt
ls /root/UAV/manifest/manifest_v2.csv
ls /root/UAV/processed_SIL_50hz | head
ls /root/UAV/processed_HIL_50hz | head
ls /root/UAV/processed_Real_50hz | head
ls /root/UAV/extra_normals/nf_buaa_case_189_1419000188.csv
```

如果 `unzip` 后多了一层目录，整理到上述路径即可。

## 数据量检查

数据已经在 `/root/UAV`。推荐先用不会被 CSV 内部逗号干扰的关键检查命令：

```bash
wc -l /root/UAV/manifest/manifest_v2.csv
find /root/UAV/processed_SIL_50hz -maxdepth 1 -name "*.csv" | wc -l
find /root/UAV/processed_HIL_50hz -maxdepth 1 -name "*.csv" | wc -l
find /root/UAV/processed_Real_50hz -maxdepth 1 -name "*.csv" | wc -l
awk -F, 'NR==1{for(i=1;i<=NF;i++){if($i=="platform") p=i; if($i=="label_main_6class") l=i; if($i=="quality_flag") q=i}next} $q!="bad" && ($l=="normal" || $l=="motor" || $l=="sensor"){a[$p","$l]++; total++} END{print "total",total; for(k in a) print k,a[k]}' /root/UAV/manifest/manifest_v2.csv
```

当前已确认 `/root/UAV` 的关键数据量正确：

```text
manifest 行数: 13270
processed_SIL_50hz: 3055
processed_HIL_50hz: 3056
processed_Real_50hz: 492

Real3 eligible total: 4193
HIL: normal 240, motor 921, sensor 690
SIL: normal 240, motor 920, sensor 690
Real: normal 51, motor 245, sensor 196
```

注意：不要用普通 `awk -F,` 统计完整 `platform` 或 `label_main_6class` 全量分布，manifest 里有带逗号的数组字段，普通 awk 会把 CSV 拆错。上面的 Real3 检查已经足够判断训练主线数据是否完整。

## 训练依赖

RTX 4090 + PyTorch 2.1.2 + CUDA 11.8 + Python 3.10 可以跑这套训练。AutoDL 镜像已带 PyTorch 时，不要重复安装 torch，只安装训练所需补充依赖：

```bash
cd /root/uav_fault_train_min
python -m pip install -U pip
python -m pip install -r requirements-autodl-train.txt
```

`pyproject.toml` 是依赖唯一事实来源。`requirements-autodl-train.txt` 选择不重复安装 PyTorch 的 `training-autodl` 分组，补充依赖包括：

```text
numpy
pandas
scikit-learn
matplotlib
tabulate
```

如果镜像没有预装 PyTorch，改用 `python -m pip install ".[training]"`。

检查 PyTorch 和 GPU：

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda:", torch.version.cuda)
print("available:", torch.cuda.is_available())
print("device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
PY
```

期望看到 `available: True` 和 RTX 4090。如果这里是 `False`，先处理 AutoDL 镜像或 PyTorch CUDA 环境，不要开始训练。

## 先 dry-run

先打印命令，不写结果：

```bash
cd /root/uav_fault_train_min
python -B scripts/train_pipeline.py \
  --task both \
  --data-root /root/UAV \
  --run-name real3_buaa_nf_v1 \
  --online-extra-normal-dir /root/UAV/extra_normals \
  --online-extra-normal-glob 'nf_*.csv'
```

确认输出包含：

```text
online/splits real3 binary
online/cache binary fewshot windows
online/train backbone suite
```

在线 cache 命令应包含：

```text
--protocol-name rflymad_online_real3_binary_v1 --label-space real3 --source-fault-classes motor sensor
```

## 正式执行

确认 dry-run 无误后执行：

```bash
cd /root/uav_fault_train_min
python -B scripts/train_pipeline.py \
  --task both \
  --data-root /root/UAV \
  --run-name real3_buaa_nf_v1 \
  --online-extra-normal-dir /root/UAV/extra_normals \
  --online-extra-normal-glob 'nf_*.csv' \
  --execute
```

默认执行：

1. 离线三分类 split。
2. 离线三分类 cache。
3. 离线 `cnn_lstm_v2_normalproto` 训练。
4. 离线结果汇总。
5. 在线 Real3 二分类 split。
6. 在线短窗口 cache。
7. 在线 `kg_mstca/mstca/cnn1d` 训练和汇总。

输出目录：

```text
/root/UAV/training_runs/real3_buaa_nf_v1/
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

只生成 split/cache：

```bash
python -B scripts/train_pipeline.py \
  --task both \
  --data-root /root/UAV \
  --run-name real3_buaa_nf_v1 \
  --online-extra-normal-dir /root/UAV/extra_normals \
  --online-extra-normal-glob 'nf_*.csv' \
  --stages splits cache \
  --execute
```

已有 split/cache 后只训练和汇总：

```bash
python -B scripts/train_pipeline.py \
  --task both \
  --data-root /root/UAV \
  --work-root /root/UAV/training_runs/real3_buaa_nf_v1 \
  --stages train summary \
  --execute \
  --force
```

只跑在线：

```bash
python -B scripts/train_pipeline.py \
  --task online \
  --data-root /root/UAV \
  --run-name real3_online_buaa_nf_v1 \
  --online-extra-normal-dir /root/UAV/extra_normals \
  --online-extra-normal-glob 'nf_*.csv' \
  --execute
```

只跑离线：

```bash
python -B scripts/train_pipeline.py \
  --task offline \
  --data-root /root/UAV \
  --run-name transfer3_offline_v1 \
  --execute
```

## 参数建议

第一轮不要急着改 batch，先跑通 baseline：

```text
offline: epochs=120, batch_size=32, lr=3e-4
online : epochs=100, batch_size=512, lr=3e-4
online models: kg_mstca, mstca, cnn1d
```

如果后续要补 backbone 对比：

```bash
python -B scripts/train_pipeline.py \
  --task online \
  --data-root /root/UAV \
  --run-name real3_online_backbone_full \
  --online-models kg_mstca mstca cnn1d bilstm transformer \
  --online-epochs 120 \
  --execute
```

## 训练监控

```bash
nvidia-smi -l 5
```

保存日志：

```bash
mkdir -p /root/UAV/training_runs/logs
python -B scripts/train_pipeline.py \
  --task both \
  --data-root /root/UAV \
  --run-name real3_buaa_nf_v1 \
  --online-extra-normal-dir /root/UAV/extra_normals \
  --online-extra-normal-glob 'nf_*.csv' \
  --execute 2>&1 | tee /root/UAV/training_runs/logs/real3_buaa_nf_v1.log
```

## 结果检查

离线结果：

```text
/root/UAV/training_runs/real3_buaa_nf_v1/offline_transfer3/results/transfer3_cnnlstm_normalproto/
├─ best.pt
├─ config.json
├─ metrics_val.json
├─ metrics_test.json
└─ metrics_external_test.json
```

在线结果：

```text
/root/UAV/training_runs/real3_buaa_nf_v1/online_real3_binary/models/
├─ online_detector_kg_mstca_rfly/
├─ online_detector_mstca_rfly/
└─ online_detector_cnn1d_rfly/
```

汇总文件：

```text
/root/UAV/training_runs/real3_buaa_nf_v1/offline_transfer3/summary/transfer3_backbone_ranking.md
/root/UAV/training_runs/real3_buaa_nf_v1/online_real3_binary/summary/online_backbone_ranking.md
```

快速查看指标：

```bash
python - <<'PY'
import json
from pathlib import Path

run = Path("/root/UAV/training_runs/real3_buaa_nf_v1")
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

检查在线模型训练合同是否正确写入：

```bash
python - <<'PY'
import json
from pathlib import Path

run = Path("/root/UAV/training_runs/real3_buaa_nf_v1")
for model_dir in sorted((run / "online_real3_binary/models").iterdir()):
    if not model_dir.is_dir():
        continue
    cfg = json.loads((model_dir / "config.json").read_text())
    stats = json.loads((model_dir / "norm_stats.json").read_text())
    print(model_dir.name, {
        "task": cfg.get("task"),
        "protocol_name": cfg.get("protocol_name"),
        "label_space": cfg.get("label_space"),
        "source_fault_classes": cfg.get("source_fault_classes"),
        "extra_normal_dir": stats.get("extra_normal_dir"),
        "extra_normal_max_windows_per_file": stats.get("extra_normal_max_windows_per_file"),
    })
PY
```

## 回传结果

只回传训练结果，不回传 cache。cache 体积大且可以重建。

AutoDL 上打包：

```bash
cd /root/UAV/training_runs
tar -czf real3_buaa_nf_v1_results.tar.gz \
  real3_buaa_nf_v1/manifest \
  real3_buaa_nf_v1/offline_transfer3/results \
  real3_buaa_nf_v1/offline_transfer3/summary \
  real3_buaa_nf_v1/online_real3_binary/models \
  real3_buaa_nf_v1/online_real3_binary/summary
```

Windows PowerShell 下载：

```powershell
scp -P 50682 root@connect.westd.seetacloud.com:/root/UAV/training_runs/real3_buaa_nf_v1_results.tar.gz D:\UAV-Data\training_runs\
```

## 常见问题

`ModuleNotFoundError: No module named 'uav_fault'`：确认是在 `/root/uav_fault_train_min` 下运行，且 `packages/uav_fault/uav_fault` 存在。`scripts/train_pipeline.py` 会自动设置 `PYTHONPATH`。

`No module named sklearn`：没有安装训练最小依赖，执行 `python -m pip install -r requirements-autodl-train.txt`。

`torch.cuda.is_available() == False`：PyTorch/CUDA 环境有问题，不要继续训练。

`manifest missing source_csv_path`：上传的不是 `manifest_v2.csv`，或者 manifest 不完整。

`FileNotFoundError: Case_xxx.csv`：通常是数据解压路径不对。确认 `processed_SIL_50hz`、`processed_HIL_50hz`、`processed_Real_50hz` 位于 `/root/UAV/` 下。

`work root is not empty`：同名 run 已存在。换 `--run-name`，或确认复用后加 `--force`。

## 下一步

第一轮建议只跑默认主线：

```bash
python -B scripts/train_pipeline.py \
  --task both \
  --data-root /root/UAV \
  --run-name real3_buaa_nf_v1 \
  --online-extra-normal-dir /root/UAV/extra_normals \
  --online-extra-normal-glob 'nf_*.csv' \
  --execute
```

拿到结果后再决定是否扩展 backbone、重复随机种子或重新设计模型结构。
