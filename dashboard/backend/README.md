# Dashboard Backend

Flask + SQLite 后端，负责：

- 登录鉴权
- 日志资产管理（ULG / CSV 上传、列表、下载、删除、预处理）
- 离线诊断任务调度
- 实时监测与遥测会话管理
- 诊断报告生成与导出
- AI 助手与知识库检索

## 目录

```text
dashboard/backend/
├─ app/
├─ requirements.txt
├─ run.py
└─ runtime/          # 运行时自动生成
```

## 环境变量

建议从 `.env.example` 复制：

```powershell
Copy-Item .\dashboard\backend\.env.example .\dashboard\backend\.env -Force
```

常用项包括：

- `DB_PATH=runtime/dashboard.db`
- `REPORTS_DIR=runtime/reports`
- `MODEL_ROOT=models`
- `AI_KNOWLEDGE_DIR=knowledge_base`

## 安装依赖

```powershell
cd D:\UAV-Fault-Diagnosis
python -m venv .\dashboard\backend\.venv
.\dashboard\backend\.venv\Scripts\python.exe -m pip install -r .\dashboard\backend\requirements.txt
```

## 启动

从仓库根目录启动：

```powershell
cd D:\UAV-Fault-Diagnosis
powershell -ExecutionPolicy Bypass -File .\scripts\start_dashboard_backend.ps1
```

如果已经进入后端目录：

```powershell
cd D:\UAV-Fault-Diagnosis\dashboard\backend
.\.venv\Scripts\python.exe run.py
```

默认地址：

- `http://localhost:5000`
- 健康检查：`http://localhost:5000/health`
- API 根路径：`http://localhost:5000/api`

## 默认账号

首次启动会自动创建：

- `admin / admin123`

## 运行时目录说明

以下目录在运行过程中自动生成，不建议提交到 Git：

- `runtime/uploads`
- `runtime/derived`
- `runtime/reports`
- `runtime/runs`
- `runtime/ai`

## AI 助手说明

AI 助手后端支持：

- 通用问答
- 报告分析
- 本地知识库检索
- `bm25 / hybrid / semantic` 三种检索模式

当 `knowledge_index.json` 中引用的历史 `run/report` 已失效时，系统会自动重建索引，避免引用已清理的旧任务记录。
