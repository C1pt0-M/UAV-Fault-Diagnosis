# UAV-Fault-Diagnosis

基于 RflyMAD 数据、实时遥测与 Dashboard 的无人机故障诊断项目。

当前仓库主要覆盖三条主线：

- 离线四分类诊断：`normal / motor / propeller / sensor`
- 在线二分类检测：`normal / fault`
- Dashboard 系统：提供日志管理、离线诊断、数据分析与诊断报告能力

如果本地已经准备好训练好的模型目录 `models/` 与待测数据，现阶段可以直接启动 Dashboard 或命令行推理，不需要重新训练。

## 项目结构

```text
UAV-Fault-Diagnosis/
├── dashboard/                # Flask + Vue3 Dashboard
│   ├── backend/
│   └── frontend/
├── data/                     # 数据与缓存
├── docs/                     # 项目文档
├── models/                   # 训练好的模型
├── scripts/                  # 启动、评估与辅助脚本
├── src/                      # 训练与推理代码
└── README.md
```

## 当前环境策略

- 主开发环境：Windows
- Dashboard 启动方式：PowerShell
- WSL2：保留给 PX4 / Linux 兼容链路，当前不是主运行环境

## 环境要求

- Python 3.10
- Node.js 18+ 与 npm
- `dashboard/backend` 使用独立虚拟环境 `.venv`

## Dashboard 快速启动

首次启动前，先准备后端虚拟环境与前端依赖：

```powershell
cd D:\UAV-Fault-Diagnosis
python -m venv dashboard\backend\.venv
.\dashboard\backend\.venv\Scripts\python.exe -m pip install -r .\dashboard\backend\requirements.txt
Copy-Item .\dashboard\backend\.env.example .\dashboard\backend\.env
cd .\dashboard\frontend
npm install
cd ..\..
```

## 启动命令总览

以下命令默认都在仓库根目录 `D:\UAV-Fault-Diagnosis` 下执行。

### 1. 创建后端虚拟环境

```powershell
cd D:\UAV-Fault-Diagnosis
python -m venv .\dashboard\backend\.venv
```

### 2. 安装后端依赖

```powershell
cd D:\UAV-Fault-Diagnosis
.\dashboard\backend\.venv\Scripts\python.exe -m pip install -r .\dashboard\backend\requirements.txt
```

### 3. 生成后端环境变量文件

```powershell
cd D:\UAV-Fault-Diagnosis
Copy-Item .\dashboard\backend\.env.example .\dashboard\backend\.env -Force
```

### 4. 安装前端依赖

```powershell
cd D:\UAV-Fault-Diagnosis\dashboard\frontend
npm install
```

一键启动前后端：

```powershell
powershell -ExecutionPolicy Bypass -File .\\scripts\\start_dashboard.ps1
```

分开启动：

```powershell
powershell -ExecutionPolicy Bypass -File .\\scripts\\start_dashboard_backend.ps1
powershell -ExecutionPolicy Bypass -File .\\scripts\\start_dashboard_frontend.ps1
```

只启动后端：

```powershell
cd D:\UAV-Fault-Diagnosis
powershell -ExecutionPolicy Bypass -File .\\scripts\\start_dashboard_backend.ps1
```

只启动前端：

```powershell
cd D:\UAV-Fault-Diagnosis
powershell -ExecutionPolicy Bypass -File .\\scripts\\start_dashboard_frontend.ps1
```

如果已经进入后端目录，也可以直接启动后端：

```powershell
cd D:\UAV-Fault-Diagnosis\dashboard\backend
.\.venv\Scripts\python.exe run.py
```

如果已经进入前端目录，也可以直接启动前端：

```powershell
cd D:\UAV-Fault-Diagnosis\dashboard\frontend
npm run frontend
```

如果需要从 Windows 宿主桥接启动 WSL 内的 Dashboard：

```powershell
powershell -ExecutionPolicy Bypass -File .\\scripts\\start_dashboard_wsl.ps1
```

启动后访问：

- Backend：`http://localhost:5000`
- Frontend：`http://localhost:5173`

默认账号：

- `admin / admin123`

## 说明

- 当前阶段不要求在 Windows 端重新训练模型，优先复用已有 `models/` 目录。
- 数据集与大体积中间产物不建议放进 WSL，按需保留在 Windows 主环境即可。
- 若后续需要接 PX4 仿真，可再通过 WSL2 单独维护 Linux 运行链路。

## 相关文档

- [dashboard/backend/README.md](dashboard/backend/README.md)
- [dashboard/frontend/README.md](dashboard/frontend/README.md)
- [docs/project_status_and_windows_migration.md](docs/project_status_and_windows_migration.md)
