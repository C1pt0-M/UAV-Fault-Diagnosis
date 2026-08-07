# Monorepo Migration Notes

本说明记录本轮工程化整理后的稳定入口和迁移策略，方便审查和后续维护。

## 路径迁移

| 旧路径 | 新路径 | 说明 |
| --- | --- | --- |
| `dashboard/backend` | `apps/dashboard/backend` | Flask API，包名为 `uav_dashboard_api` |
| `dashboard/frontend` | `apps/dashboard/frontend` | Vue 3 + Vite 前端 |
| `src/*` | `packages/uav_fault/uav_fault/*` | 训练、特征、在线检测、telemetry 核心代码 |
| `realtime/*` | `packages/rfly_bridge/rfly_bridge/*` | RflySim/PX4 实时桥接 |
| `models/*` | `assets/models/*` | 运行必需模型权重和 metadata |

根目录 package shim 已在迁移验收后删除。`apps/` 与 `packages/` 下的 package 是唯一 Implementation，开发环境通过 editable install 使用其公开 Interface。

## 模型资产策略

`assets/models/manifest.json` 是运行模型清单。后端 registry 有 manifest 时只注册清单中的模型，并要求清单里的目录实际存在。未列入 manifest 的实验模型目录不作为运行模型注册。

`uav_fault.models` 是模型结构、训练和推理代码包名；`assets/models` 是权重和 metadata 资产目录。两者不要混用。

## 脚本分层

顶层保留常用启动脚本：

- `scripts/start_dashboard.ps1`
- `scripts/start_dashboard_backend.ps1`
- `scripts/start_dashboard_frontend.ps1`

其他 Python 脚本按用途移动到：

- `scripts/data/`
- `scripts/dev/`
- `scripts/eval/`
- `scripts/plots/`

旧顶层 Python 转发脚本已在仓内调用清零后删除；脚本只从上述 canonical 路径运行。

## 验证命令

Python 依赖和 pytest discovery 已统一到根 `pyproject.toml`，本地和 CI 使用同一入口：

```powershell
.\apps\dashboard\backend\.venv\Scripts\python.exe .\scripts\verify.py
```

## 配置 Profile

后端通过 `APP_ENV` 选择 `development`、`production` 或 `testing` Profile，
`create_app()` 不再固定加载基础配置类。未知 Profile 会直接报错。

`ENABLE_SETUP_API` 现在实际控制 setup Blueprint 是否注册；生产部署首次初始化完成后应关闭。
`ENABLE_PUBLIC_REGISTER` 默认关闭，符合管理员控制账号访问的 V1.0 产品边界。

离线推理活动入口统一命名为 `offline_multiclass`：

- bridge：`run_offline_multiclass`
- worker：`_offline_multiclass_worker`
- script：`apps/dashboard/backend/scripts/offline_multiclass_infer.py`

历史任务值 `offline_5class` 仅在模型 registry 中作为旧资产兼容别名保留，不作为新任务写入。

## Remaining Refactors

`telemetry_manager.py`、`telemetry_backends.py` 与 `ai_assistant.py` 的后续深 Module 治理独立进行；`uav_fault.models` 继续作为稳定 package Interface。
