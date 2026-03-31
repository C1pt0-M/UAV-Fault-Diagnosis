# Dashboard Frontend

Vue3 + Vite + Element Plus 前端，当前包含：

- 登录
- 首页
- 日志管理
- 故障检测
- 数据分析
- 诊断报告

## 开发依赖

```powershell
cd D:\UAV-Fault-Diagnosis\dashboard\frontend
npm install
```

## Windows 主环境启动

从仓库根目录启动：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\start_dashboard_frontend.ps1
```

如果先安装依赖，建议完整命令如下：

```powershell
cd D:\UAV-Fault-Diagnosis\dashboard\frontend
npm install
cd ..\..
powershell -ExecutionPolicy Bypass -File .\scripts\start_dashboard_frontend.ps1
```

如果你已经进入 `dashboard/frontend`，也可以直接运行：

```powershell
npm run frontend
```

默认端口：`5173`

## 构建

```powershell
npm run build
```
