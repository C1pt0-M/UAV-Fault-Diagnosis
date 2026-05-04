# Dashboard Backend

Flask + SQLite 后端，提供认证、资产管理、离线诊断任务、实时监测、诊断报告和 AI 助手 API。

## 目录

```text
apps/dashboard/backend/
├─ uav_dashboard_api/   # Flask package
├─ tests/               # backend tests
├─ requirements.txt
├─ run.py               # dev launcher
├─ wsgi.py              # WSGI entrypoint
└─ runtime/             # local runtime output, ignored by Git
```

## 安装

```powershell
cd D:\UAV-Fault-Diagnosis
python -m venv .\apps\dashboard\backend\.venv
.\apps\dashboard\backend\.venv\Scripts\python.exe -m pip install -r .\apps\dashboard\backend\requirements.txt
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

- `MODEL_ROOT=assets/models`
- `OFFLINE_NORM_STATS=assets/models/transfer3_cnnlstm_normalproto/norm_stats.json`
- `AI_KNOWLEDGE_DIR=knowledge_base`
- `RUNTIME_DIR=runtime`
- `TELEMETRY_ALLOWED_BACKENDS=replay,rfly`
- `RFLY_PX4MAVCTRL_PATH=<PX4MavCtrlV4.py path>`

运行时权重和 metadata 在 `assets/models/`，由 `assets/models/manifest.json` 控制注册清单。模型结构代码仍在 `uav_fault.models`。

本地 `.env`、`runtime/`、`logs/`、`.venv/` 均不应提交。
