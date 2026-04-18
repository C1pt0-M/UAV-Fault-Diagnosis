# UAV-Fault-Diagnosis

基于 RflyMAD 数据、实时遥测与 Dashboard 的无人机故障检测与诊断项目。

当前仓库主要覆盖三条主线：

- 离线故障诊断：面向日志资产的多分类诊断任务
- 在线故障检测：面向实时遥测流的在线检测与告警
- Dashboard 系统：提供日志资产管理、诊断中心、报告管理、AI 助手与实时监测能力

## 项目结构

```text
UAV-Fault-Diagnosis/
├─ dashboard/
│  ├─ backend/      # Flask + SQLite 后端
│  └─ frontend/     # Vue3 + Vite 前端
├─ models/          # 当前运行默认使用的模型文件
├─ knowledge_base/  # AI 助手使用的补充背景资料
├─ scripts/         # 启动、处理与辅助脚本
├─ src/             # 训练、推理与在线检测代码
├─ realtime/        # 实时桥接与运行脚本
├─ README.md
└─ requirements.txt
```

## 当前运行方式

当前默认运行方式是 Windows 本地启动 Dashboard：

- Backend：`dashboard/backend`
- Frontend：`dashboard/frontend`
- 默认后端地址：`http://localhost:5000`
- 默认前端地址：`http://localhost:5173`
- 实时数据源：支持 `RflySim` 仿真接入与 `CSV` 回放；本地接入 `RflySim` 时需要在后端配置 `RFLY_PX4MAVCTRL_PATH`

## 环境要求

- Python 3.10
- Node.js 18+
- npm

## 模型说明

仓库已保留当前运行所需模型，位于 `models/`：

- `models/transfer3_cnnlstm_normalproto/`
  用于离线诊断主流程
- `models/online_detector_cnn1d_rfly/`
  用于在线检测主流程
- `models/online_detector_CNN/`
  作为在线检测兼容模型保留

如果只想运行当前系统，不需要重新训练模型。

## 快速启动

首次运行前，先准备后端虚拟环境与前端依赖：

```powershell
cd D:\UAV-Fault-Diagnosis
python -m venv .\dashboard\backend\.venv
.\dashboard\backend\.venv\Scripts\python.exe -m pip install -r .\dashboard\backend\requirements.txt
Copy-Item .\dashboard\backend\.env.example .\dashboard\backend\.env -Force

cd .\dashboard\frontend
npm install
cd ..\..
```

如果要启用 `RflySim` 实时接入，确认 `dashboard/backend/.env` 中至少包含：

```powershell
TELEMETRY_ALLOWED_BACKENDS=rfly,replay
RFLY_PX4MAVCTRL_PATH=D:\BuaaAutoTestPlatForm\AutoTestAPI\PX4MavCtrlV4.py
```

先启动官方 `RflySim` 环境，再在 Dashboard 的“在线监测”页选择 `RflySim` 并启动采集。

一键启动前后端：

```powershell
cd D:\UAV-Fault-Diagnosis
powershell -ExecutionPolicy Bypass -File .\scripts\start_dashboard.ps1
```

分开启动：

```powershell
cd D:\UAV-Fault-Diagnosis
powershell -ExecutionPolicy Bypass -File .\scripts\start_dashboard_backend.ps1
```

```powershell
cd D:\UAV-Fault-Diagnosis
powershell -ExecutionPolicy Bypass -File .\scripts\start_dashboard_frontend.ps1
```

如果已经进入对应目录，也可以直接运行：

```powershell
cd D:\UAV-Fault-Diagnosis\dashboard\backend
.\.venv\Scripts\python.exe run.py
```

```powershell
cd D:\UAV-Fault-Diagnosis\dashboard\frontend
npm run frontend
```

## 默认账号

- `admin / admin123`

## AI 助手说明

AI 助手当前以“报告助手优先”为主：

- 支持通用无人机知识问答
- 支持带 `report_id` 的报告分析
- 可使用 `knowledge_base/` 中的资料作为补充背景知识
- 只有在检索结果确实相关时，才会在回复下方显示“参考资料”

如果更新了 `knowledge_base/` 内容，可以在前端 AI 助手页面点击“重建知识库”，或调用后端重建接口。

## 验证命令

前端：

```powershell
cd D:\UAV-Fault-Diagnosis\dashboard\frontend
npm run test
npm run build
```

后端：

```powershell
cd D:\UAV-Fault-Diagnosis
.\dashboard\backend\.venv\Scripts\python.exe -m pytest dashboard\backend\tests\test_ai_assistant.py dashboard\backend\tests\test_knowledge_base_rag.py -q
```

## 相关文档

- [dashboard/backend/README.md](dashboard/backend/README.md)
- [dashboard/frontend/README.md](dashboard/frontend/README.md)
