# 项目现状与 Windows 运行说明

更新时间：2026-08-01

## 1. 文档定位

本文描述当前仓库的真实主线、运行方式和下一步工作。出现口径冲突时，按以下顺序判断：

1. 当前代码、`assets/models/manifest.json` 与自动化测试；
2. 根目录 `README.md`、`CONTEXT.md` 和 `docs/model_training_pipeline.md`；
3. 本文；
4. `docs/archive/` 下的历史实验记录。

历史文档中的旧 `src/`、`dashboard/`、五分类或六分类描述不再代表当前交付版本。

## 2. 当前产品边界

项目名称统一为“基于数据驱动的无人机故障智能诊断系统”。当前交付围绕飞行日志资产和可重复的数据流分析展开：

- 资产中心：管理 CSV/ULG 输入、预处理状态和来源追溯；
- 在线监测：对 replay 或扩展遥测流进行短窗口故障检测、事件聚合和风险告警；
- 诊断中心：对准备完成的飞行日志执行离线多分类诊断；
- 诊断报告：保存输入资产、模型、诊断结果、证据、风险和建议；
- AI 助手：解释已有报告和知识库内容，不替代人工维护决策；
- 账号访问：首次初始化管理员，默认不开放公开注册。

RflySim/PX4 是扩展接入接口。比赛或交付的基础演示应能仅依赖 CSV replay 完成，避免把外部仿真安装作为系统成立的前提。

## 3. 当前仓库结构

```text
UAV-Fault-Diagnosis/
├─ apps/dashboard/backend/       Flask API 与后台任务
├─ apps/dashboard/frontend/      Vue 3 + Vite 前端
├─ packages/uav_fault/           数据、特征、模型和在线检测核心包
├─ packages/rfly_bridge/         RflySim/PX4 桥接包
├─ assets/models/                运行模型、指标与 manifest
├─ scripts/                      数据、训练、评估、绘图和启动入口
├─ tests/                        跨 package 与工程约束测试
├─ docs/                         当前文档及 archive 历史材料
├─ data/                         本地处理数据，不纳入 Git
└─ runtime/                      本地运行和训练产物，不纳入 Git
```

`apps/` 与 `packages/` 下的实现是唯一 canonical source。旧根 package shim 和旧顶层 Python 转发脚本已经删除。

## 4. 当前算法主线

### 4.1 离线多分类诊断

当前交付任务为：

```text
normal / motor / sensor
```

默认运行模型：

```text
transfer3_cnnlstm_normalproto
```

当前活动命名统一为：

- task：`offline_multiclass`
- bridge：`run_offline_multiclass`
- worker：`_offline_multiclass_worker`
- inference script：`apps/dashboard/backend/scripts/offline_multiclass_infer.py`

旧值 `offline_5class` 只用于识别历史模型 metadata，不作为新任务或新代码入口。

### 4.2 在线二分类监测

在线任务统一为：

```text
normal / fault
```

训练数据范围使用 Real 可验证的 `normal/motor/sensor`：normal 为负类，motor 和 sensor 合并为 fault 正类。

运行模型清单：

- `online_detector_kg_mstca_rfly`：当前默认模型；
- `online_detector_mstca_rfly`；
- `online_detector_cnn1d_rfly`。

Dashboard 只展示 `assets/models/manifest.json` 中实际存在且通过能力检查的模型，不维护另一份手写模型清单。

### 4.3 历史实验

旧五分类、六分类、propeller/wind/load 等实验可以保留为研究记录，但不得在 README、比赛材料或当前界面中描述为已交付能力。相关说明应放在 `docs/archive/`。

## 5. 数据与训练流水线

统一入口：

```powershell
python -B scripts\train_pipeline.py --task both --data-root D:\UAV-Data
```

默认只执行 dry-run。确认输入、split、cache 和输出目录后，再添加 `--execute`。

流水线包含：

- 离线：split → 三分类 sequence cache → Normal Prototype 训练 → backbone 汇总；
- 在线：Real3 split → binary window cache → KG-MSTCA/MSTCA/CNN1D 训练与汇总。

训练结果不会自动覆盖 `assets/models/`。只有完成 test、external test、replay 和典型异常样本验收后，才应手动发布运行模型。

## 6. Dashboard 运行闭环

标准业务流程：

```text
管理员初始化与登录
  → 上传或选择飞行日志资产
  → 准备 CSV 输入
  → 在线 replay 监测或离线多分类诊断
  → 风险评估与证据汇总
  → 生成诊断报告
  → AI 辅助解释
```

在线监测会话检测到可疑风险后，可以把会话生成的 CSV 资产移交到诊断中心进行离线分类，形成 monitoring-to-diagnosis handoff。

## 7. Windows 运行方式

Dashboard、CSV replay 和常规测试以 Windows PowerShell 为主运行环境。项目专用 Python 位于：

```text
apps/dashboard/backend/.venv/Scripts/python.exe
```

一键启动：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\start_dashboard.ps1
```

默认地址：

- Frontend：`http://localhost:5173`
- Backend：`http://localhost:5000`
- Health：`http://localhost:5000/health`

WSL/Linux 或 AutoDL 主要用于大规模数据处理和 GPU 训练，不再作为 Dashboard 正常运行的必需环境。

## 8. 配置 Profile 与首次初始化

`APP_ENV` 现在实际决定后端加载的配置 Profile：

- `development`：本地开发；
- `production`：比赛、部署或正式演示；
- `testing`：自动化测试。

未知值会阻止启动，不会静默回退到 development。

首次部署：

1. 复制 `apps/dashboard/backend/.env.example` 为 `.env`；
2. 设置真实的 `SECRET_KEY`、`JWT_SECRET_KEY` 和允许的 CORS 来源；
3. 保持 `ENABLE_SETUP_API=True`；
4. 启动系统并在 `/setup` 初始化管理员；
5. 将 `ENABLE_SETUP_API=False`，然后重启后端；
6. 保持 `ENABLE_PUBLIC_REGISTER=False`，由管理员维护账号。

模型配置不在设置页维护。离线和在线诊断模型分别由模型 manifest、后端 registry 以及对应业务页面共同选择。设置页只负责 AI 助手的 LLM 接入配置。

## 9. RflySim/PX4 扩展

实时扩展支持两种模式：

- `attach`：连接已运行的 RflySim/PX4；
- `buaa`：通过 BuaaAutoTestPlatform 启动并复现 case。

Windows 侧负责 CopterSim、RflySim3D、QGroundControl 和 Dashboard；WSL 只在需要时承载 PX4 SITL。现场演示前必须单独验证外部安装路径、端口、地图、case 和采集 CSV，不得用单元测试结果替代真实链路验收。

## 10. 验证基线

统一验证入口：

```powershell
.\apps\dashboard\backend\.venv\Scripts\python.exe .\scripts\verify.py
```

该命令依次执行：

1. Python `compileall`；
2. 根测试；
3. Dashboard backend 测试；
4. 核心 online/telemetry/rfly_bridge 测试；
5. Frontend Vitest；
6. Vite production build。

外部 RflySim/Buaa 安装相关测试使用 `external` marker，不属于无外部依赖的默认基线。

## 11. 当前下一步

参赛准备按以下顺序推进：

1. 保持代码、README、开发文档和界面术语一致；
2. 完成比赛开发文档的需求、概要、详细设计、测试、安装使用和项目总结六章；
3. 固化一套无需外部仿真的 CSV replay 演示数据；
4. 单独验收 RflySim/Buaa 扩展演示；
5. 整理模型指标、测试记录、界面截图和故障证据链；
6. 在材料冻结前不进行大规模架构重构。

长期维护重点是拆分 `telemetry_manager.py`、`telemetry_backends.py` 和 `ai_assistant.py`，但这些工作不应阻塞当前比赛材料与演示闭环。
