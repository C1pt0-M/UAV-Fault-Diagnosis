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

根目录保留 `uav_dashboard_api/`、`uav_fault/`、`rfly_bridge/` shim 包，保证从仓库根目录直接 import 和 `python -m ...` 可用。

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

旧顶层 Python 脚本暂时保留 shim，打印迁移提示并转发到新路径。后续确认没有外部依赖后，可以删除这些 shim。

## 验证命令

```powershell
python -m compileall -q uav_dashboard_api uav_fault rfly_bridge apps\dashboard\backend\uav_dashboard_api packages\uav_fault\uav_fault packages\rfly_bridge\rfly_bridge scripts tests
python -m pytest tests\test_monorepo_layout.py tests\test_decision_engine.py -q
.\apps\dashboard\backend\.venv\Scripts\python.exe -m pytest apps\dashboard\backend\tests\test_model_registry.py apps\dashboard\backend\tests\test_runtime_portability.py -q
cd apps\dashboard\frontend
npm run test
npm run build
```

## Deferred Refactors

本轮不继续拆分 `telemetry_manager.py`、`ai_assistant.py`、`dashboard.py`，也不把 `uav_fault.models` 改名为 `modeling` 或 `nn`。这些适合在当前目录迁移合并后作为独立 refactor 处理。
