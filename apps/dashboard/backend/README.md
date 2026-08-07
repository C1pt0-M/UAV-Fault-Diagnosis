# Dashboard Backend

Flask + SQLite 后端，提供认证、资产管理、离线诊断任务、实时监测、诊断报告和 AI 助手 API。

## 目录

```text
apps/dashboard/backend/
├─ uav_dashboard_api/   # Flask package
├─ tests/               # backend tests
├─ wsgi.py              # WSGI entrypoint
└─ runtime/             # local runtime output, ignored by Git
```

## 安装

Python 依赖的唯一事实来源是仓库根目录的 `pyproject.toml`。

```powershell
cd D:\UAV-Fault-Diagnosis
python -m venv .\apps\dashboard\backend\.venv
.\apps\dashboard\backend\.venv\Scripts\python.exe -m pip install -e ".[dashboard,test]"
Copy-Item .\apps\dashboard\backend\.env.example .\apps\dashboard\backend\.env -Force
```

## 启动

```powershell
cd D:\UAV-Fault-Diagnosis
.\apps\dashboard\backend\.venv\Scripts\python.exe -m uav_dashboard_api
```

也可以使用：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\start_dashboard_backend.ps1
```

默认地址：

- `http://localhost:5000`
- `http://localhost:5000/health`
- API prefix: `/api`

## 关键配置

- `APP_ENV=development|production|testing`：选择实际加载的配置 Profile。
- `ENABLE_SETUP_API=True|False`：控制 `/api/setup/*` 是否注册；首次初始化后应关闭。
- `ENABLE_PUBLIC_REGISTER=False`：V1.0 默认采用管理员控制的账号访问，不开放公开注册。
- `MODEL_ROOT=assets/models`
- `OFFLINE_NORM_STATS=assets/models/transfer3_cnnlstm_normalproto/norm_stats.json`
- `AI_KNOWLEDGE_DIR=knowledge_base`
- `RUNTIME_DIR=runtime`
- `TELEMETRY_ALLOWED_BACKENDS=replay,rfly`
- `RFLY_PYTHON_BIN=D:\PX4PSP\Python38\python.exe`
- `RFLY_PX4MAVCTRL_PATH=D:\PX4PSP\RflySimAPIs\RflySimSDK\ctrl\PX4MavCtrlV4.py`
- `RFLY_BRIDGE_MODE=attach`：连接已启动的 RflySim/PX4。
- `RFLY_BRIDGE_MODE=buaa`：由 `BuaaAutoTestPlatform` bat 启动仿真并执行复现 case。
- `RFLY_BUAA_AUTOTEST_DIR=D:\BuaaAutoTestPlatForm\AutoTestAPI`
- `RFLY_BUAA_UNIQUE_INDEX=1419000188`

运行时权重和 metadata 在 `assets/models/`，由 `assets/models/manifest.json` 控制注册清单。模型结构代码仍在 `uav_fault.models`。

## RflySim 运行边界

- Dashboard backend/frontend 从 Windows PowerShell 启动。
- RflySim GUI 层在 Windows 侧运行，包括 `CopterSim.exe`、`RflySim3D.exe` 和 `QGroundControl.exe`。
- PX4 SITL 可由 `scripts/start_rflysim_sitl_qgc.ps1` 拉起到 `RflySim-20.04` WSL 发行版；不要在 WSL 里直接启动 GUI 仿真层。
- QGC 是人工操控入口。脚本会为 QGC 写入 UDP auto link，默认端口 `14600`；Dashboard 的 MAVLink 补充监听默认使用 `14031`。
- 前端选择 `RflySim 实时链路` 后，后端按 `RFLY_BRIDGE_MODE` 决定 attach 已有仿真，或通过 BuaaAutoTestPlatform 自动复现 case。

## Rfly/Buaa 采集后复核

前端采集完成后，可对生成的 `TS-*.csv` 做字段健康检查和离线 replay 对齐：

```powershell
cd D:\UAV-Fault-Diagnosis
.\apps\dashboard\backend\.venv\Scripts\python.exe .\scripts\eval\eval_telemetry_live.py `
  --csv-dir .\apps\dashboard\backend\runtime\telemetry_sessions `
  --glob TS-416582.csv `
  --backend replay `
  --model-dir .\assets\models\online_detector_kg_mstca_rfly `
  --every 50 `
  --out .\runtime\rfly_replay_check.csv
```

重点看输出中的 `input_contract_status`、`input_contract_detection_blocked`、`mean_fault_prob`、`max_fault_prob` 和事件级指标。

## 实时概率偏高诊断

如果 No Fault 或起飞前出现高故障概率，先跑 schema drift 诊断，定位是输入合同被阻断、字段单位/别名错误，还是真实分布超出训练集：

```powershell
cd D:\UAV-Fault-Diagnosis
.\apps\dashboard\backend\.venv\Scripts\python.exe .\scripts\eval\diagnose_live_schema_drift.py `
  --csv .\apps\dashboard\backend\runtime\telemetry_sessions\TS-416582.csv `
  --model-dir .\assets\models\online_detector_kg_mstca_rfly `
  --backend replay `
  --every 50 `
  --out .\runtime\live_schema_drift_summary.csv `
  --windows-out .\runtime\live_schema_drift_windows.csv
```

重点看 `input_contract_detection_blocked`、`full_file_top_z`、`max_window_top_z_feature`。起飞前地面静止窗口属于当前训练集未覆盖的 operating envelope，后端会通过 input contract 暂停检测，不把这类窗口送入模型。

本地 `.env`、`runtime/`、`logs/`、`.venv/` 均不应提交。

首次部署建议：复制 `.env.example`，保持 `ENABLE_SETUP_API=True` 完成管理员初始化，
随后改为 `False` 并重启后端。生产模式不会再静默使用 development 配置。
