# UAV-Fault-Diagnosis

基于 RflyMAD / RflySim 数据、在线遥测流和 Dashboard 的无人机故障检测与诊断项目。

当前仓库按 monorepo 方式组织，主要包含三条线：

- 离线故障诊断：面向日志资产的多分类诊断任务。
- 在线故障检测：面向实时遥测流的在线检测、告警和证据汇总。
- Dashboard 系统：提供资产管理、诊断中心、报告管理、AI 助手和实时监测能力。

## 项目结构

```text
UAV-Fault-Diagnosis/
├─ apps/
│  └─ dashboard/
│     ├─ backend/          # Flask API，包名 uav_dashboard_api
│     └─ frontend/         # Vue 3 + Vite 前端
├─ packages/
│  ├─ uav_fault/           # 核心 ML、特征、在线检测、telemetry 代码
│  └─ rfly_bridge/         # RflySim/PX4 实时桥接 CLI
├─ assets/
│  └─ models/              # 当前系统运行必需模型与 manifest
├─ docs/
│  └─ migration-monorepo.md
├─ scripts/
│  ├─ data/                # 数据检查、准备、输入门禁脚本
│  ├─ dev/                 # 本地开发辅助脚本
│  ├─ eval/                # 评估、指标、审计脚本
│  └─ plots/               # 绘图脚本
├─ deploy/                 # 部署示例
└─ tests/                  # 核心库和迁移约束测试
```

根目录保留 `uav_dashboard_api/`、`uav_fault/`、`rfly_bridge/` shim 包，用于从仓库根目录直接 import 和 `python -m ...`。

## 快速启动

后端：

```powershell
cd D:\UAV-Fault-Diagnosis
python -m venv .\apps\dashboard\backend\.venv
.\apps\dashboard\backend\.venv\Scripts\python.exe -m pip install -r .\apps\dashboard\backend\requirements.txt
Copy-Item .\apps\dashboard\backend\.env.example .\apps\dashboard\backend\.env -Force
.\apps\dashboard\backend\.venv\Scripts\python.exe -m uav_dashboard_api
```

前端：

```powershell
cd D:\UAV-Fault-Diagnosis\apps\dashboard\frontend
npm install
npm run frontend
```

一键启动前后端：

```powershell
cd D:\UAV-Fault-Diagnosis
powershell -ExecutionPolicy Bypass -File .\scripts\start_dashboard.ps1
```

默认地址：

- Backend: `http://localhost:5000`
- Frontend: `http://localhost:5173`
- Health: `http://localhost:5000/health`

默认账号：`admin / admin123`

## RflySim 接入

实时接入默认支持 `replay`。需要接入 RflySim 时，在 `apps/dashboard/backend/.env` 中配置：

```powershell
TELEMETRY_ALLOWED_BACKENDS=rfly,replay
RFLY_PX4MAVCTRL_PATH=D:\PX4PSP\RflySimAPIs\RflySimSDK\ctrl\PX4MavCtrlV4.py
```

桥接入口：

```powershell
python -m rfly_bridge --px4mavctrl D:\PX4PSP\RflySimAPIs\RflySimSDK\ctrl\PX4MavCtrlV4.py
```

## 模型资产

运行时权重和 metadata 位于 `assets/models/`，清单位于 `assets/models/manifest.json`。默认保留：

- `transfer3_cnnlstm_normalproto`
- `online_detector_kg_mstca_rfly`
- `online_detector_mstca_rfly`
- `online_detector_cnn1d_rfly`

模型结构、训练和推理代码仍在 `packages/uav_fault/uav_fault/models/`，包名继续是 `uav_fault.models`。不要把 `uav_fault.models` 和 `assets/models` 混用：前者是代码，后者是运行资产。

如果本地 `.env` 仍配置 `MODEL_ROOT=models`，后端会在旧路径不存在且新路径存在时自动迁移到 `assets/models`。

当前训练主线已整理到 `scripts/train_pipeline.py`，说明见 `docs/model_training_pipeline.md`。默认 dry-run 只打印命令，确认后再加 `--execute` 跑离线三分类和在线 Real3 二分类流水线。

## 脚本入口

常用启动脚本仍保留在 `scripts/` 顶层：

- `scripts/start_dashboard.ps1`
- `scripts/start_dashboard_backend.ps1`
- `scripts/start_dashboard_frontend.ps1`

其他 Python 脚本已按用途分层到 `scripts/data/`、`scripts/dev/`、`scripts/eval/`、`scripts/plots/`。旧顶层 Python 脚本暂时保留 shim，会提示新路径并转发执行。

## 验证命令

```powershell
cd D:\UAV-Fault-Diagnosis
python -m compileall -q uav_dashboard_api uav_fault rfly_bridge apps\dashboard\backend\uav_dashboard_api packages\uav_fault\uav_fault packages\rfly_bridge\rfly_bridge scripts tests
python -m pytest tests\test_monorepo_layout.py tests\test_decision_engine.py -q
.\apps\dashboard\backend\.venv\Scripts\python.exe -m pytest apps\dashboard\backend\tests\test_rfly_backend.py -q
```

前端：

```powershell
cd D:\UAV-Fault-Diagnosis\apps\dashboard\frontend
npm run test
npm run build
```
