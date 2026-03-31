# Dashboard Backend

Flask + SQLite 后端，负责：

- 登录鉴权
- 资产管理（ULG / CSV 上传、列表、下载）
- 离线诊断任务调度
- 数据分析接口
- 诊断报告生成与导出

## 目录

```text
dashboard/backend/
├── app/
├── requirements.txt
├── run.py
└── runtime/   # 运行时自动创建
```

## 环境变量

建议从 `.env.example` 复制一份 `.env` 后再按本机环境修改。

常用项：

- `DB_PATH=runtime/dashboard.db`
- `REPORTS_DIR=runtime/reports`
- `MODEL_ROOT=models`

## Windows 主环境启动

先准备虚拟环境：

```powershell
cd D:\UAV-Fault-Diagnosis
python -m venv .\dashboard\backend\.venv
.\dashboard\backend\.venv\Scripts\python.exe -m pip install -r .\dashboard\backend\requirements.txt
Copy-Item .\dashboard\backend\.env.example .\dashboard\backend\.env
```

安装依赖：

```powershell
cd D:\UAV-Fault-Diagnosis
.\dashboard\backend\.venv\Scripts\python.exe -m pip install -r .\dashboard\backend\requirements.txt
```

直接通过仓库根目录脚本启动：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\start_dashboard_backend.ps1
```

如果你已经进入 `dashboard/backend`，也可以直接运行：

```powershell
.\.venv\Scripts\python.exe run.py
```

默认地址：`http://localhost:5000`

## 默认账号

首次启动自动创建：

- `admin / admin123`
