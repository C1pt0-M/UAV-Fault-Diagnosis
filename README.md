# UAV-Fault-Diagnosis

基于 RflyMAD / RflySim 数据、在线遥测流和 Dashboard 的无人机故障检测与诊断项目。

## 当前进度与文档入口

**先看 [《项目进程与后续计划》](项目进程.md)** 了解项目目标、管理决策和结项安排；再看 [《文档索引与实验台账》](docs/文档索引与实验台账.md) 查询全部重要文档、实验是否做过、冻结判定和待办状态。实验结果与协议以各自原始报告/配置为准，索引负责导航和状态汇总。本文继续用于安装、启动和仓库结构说明。

2026-11-01 归档入口：[近期改动与历史MSTCA对比](docs/research/2026-11-01/github_handoff.md)、[提交前验证及已知失败](docs/research/2026-11-01/github_submission_verification.md)。源码、协议、原创报告和轻量证据纳入Git；原始数据、缓存、研究权重新增副本及第三方全文保留本地，不应把GitHub快照视为完整实验盘备份。

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

Python package 只保留 `apps/` 与 `packages/` 下的 canonical 实现；开发环境通过 editable install 暴露这些 package。

## 快速启动

后端：

```powershell
cd D:\UAV-Fault-Diagnosis
python -m venv .\apps\dashboard\backend\.venv
.\apps\dashboard\backend\.venv\Scripts\python.exe -m pip install -e ".[dashboard,test]"
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

首次使用请确认 `.env` 中 `ENABLE_SETUP_API=True`，打开
`http://localhost:5173/setup` 初始化管理员账号。初始化完成后，将该配置改为
`False` 并重启后端；公开注册默认关闭，后续账号由管理员维护。

`APP_ENV` 是后端配置 Profile 的选择入口：

- `development`：本地开发，默认允许 setup，支持 replay/Rfly 调试。
- `production`：正式或比赛部署，默认关闭 setup、公开注册和盲测，只开放显式配置的能力。
- `testing`：自动化测试使用，业务代码不应手工选择。

未知的 `APP_ENV` 会直接阻止应用启动，避免静默落入错误配置。

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

已有训练入口为 `scripts/train_pipeline.py`，操作参考见 `docs/model_training_pipeline.md`；默认 dry-run 只打印命令。该流水线仍包含旧 Real3 few-shot 协议，**不能直接视为本轮“目标域正常数据适配”的新实验方案**。执行训练前先核对 [项目进程](项目进程.md) 中的协议、数据和当前任务。

## 脚本入口

常用启动脚本仍保留在 `scripts/` 顶层：

- `scripts/start_dashboard.ps1`
- `scripts/start_dashboard_backend.ps1`
- `scripts/start_dashboard_frontend.ps1`

其他 Python 脚本按用途从 `scripts/data/`、`scripts/dev/`、`scripts/eval/`、`scripts/plots/` 的 canonical 路径运行。

## 验证命令

`pyproject.toml` 是 Python packaging、依赖分组和 pytest discovery 的唯一事实来源。AutoDL bundle 单独保留 `requirements-autodl-train.txt`，用于选择不重复安装 PyTorch 的依赖分组。

```powershell
cd D:\UAV-Fault-Diagnosis
.\apps\dashboard\backend\.venv\Scripts\python.exe .\scripts\verify.py
```

仅验证 Python：

```powershell
.\apps\dashboard\backend\.venv\Scripts\python.exe .\scripts\verify.py --skip-frontend
```
