# Dashboard Frontend

Vue 3 + Vite + Element Plus 前端。
当前页面模块包括：
- 登录
- 总览
- 实时监测
- 诊断中心
- 日志资产
- 诊断报告
- AI 助手

## 安装依赖

```powershell
cd D:\UAV-Fault-Diagnosis\apps\dashboard\frontend
npm install
```

## 启动开发环境
从仓库根目录启动：
```powershell
cd D:\UAV-Fault-Diagnosis
powershell -ExecutionPolicy Bypass -File .\scripts\start_dashboard_frontend.ps1
```

如果已经进入前端目录，也可以直接运行：
```powershell
cd D:\UAV-Fault-Diagnosis\apps\dashboard\frontend
npm run frontend
```

默认端口：
- `http://localhost:5173`

默认会将 `/api` 和 `/health` 代理到：

- `http://localhost:5000`

也可以通过环境变量覆盖：
- `VITE_BACKEND_TARGET`

## 常用命令

```powershell
npm run frontend
npm run test
npm run build
```

## 当前前端结构

```text
src/
|-- app/
|-- layouts/
|-- modules/
|   |-- ai/
|   |-- assets/
|   |-- auth/
|   |-- diagnosis/
|   |-- monitoring/
|   |-- overview/
|   `-- reports/
|-- shared/
`-- styles/
```
