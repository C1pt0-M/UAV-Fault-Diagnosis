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

`assets/models/manifest.json` 是运行模型清单。后端 registry 有 manifest 时只注册清单中的模型；清单必须声明 `version: 1`，`models` 必须为列表。每项使用 `MODEL_ROOT` 内的相对路径，不接受绝对路径、上级路径穿越或解析后越界的目录；名称须与目录名一致，不得重复，目录必须有 `config.json` 和 `best.pt`。旧清单项仍可只提供 name 或 path，可省略 runtime；声明 runtime 时只能为 online／offline，且须与模型任务相符。清单结构错误、缺文件或任务矛盾会明确报错，不悄悄跳过。未列入 manifest 的实验模型目录不作为运行模型注册；无清单时保留旧目录扫描模式。

在线与离线分别使用公共输入合同校验，但不合并预处理。当前离线入口仍为 31 通道、2048 帧的整日志处理；模型内归一化文件优先于显式配置和历史缓存候选。只有缺文件才继续回退，所有候选缺失时仍用 `sample_zscore`；已有文件损坏、数值非有限、维度或声明的通道顺序不匹配时直接报错。旧统计文件可省略 columns；离线零／微小非负 std 仍按原规则替换为 1。归一化缺失回退不是训练统计已获验证的证明。

离线注册表与推理共用 `offline_label_names`：缺省标签仍支持旧 3／4／5 类约定，标签前后空格按原规则去除，多余标签仍只取输出类数对应的前 N 项；已提供的标签必须为非空字符串，数量足够，且截取、去空格后不重复。不再将残缺标签静默替换为另一套默认含义。已知离线架构不因错误标签被改判为在线任务。

在线与离线共用特征声明校验：声明的 feature_count 必须匹配实际通道数，已知 feature_schema 必须匹配有序列名，protocol_name／feature_schema 必须为非空且无首尾空格的字符串。离线 config 与实际选中的 norm_stats 若均声明 schema／协议名称，两者必须一致；外部候选同样检查，矛盾不触发换源。旧字段缺省、单侧声明、自定义 schema 名称仍兼容；自定义名称只做声明一致性检查，不认证语义。离线不把在线 window_size 当成整日志重采样长度，也不把协议名称相同当成实验成员／数据来源已验证。

离线子进程从 `packages/uav_fault` 导入 canonical package，不再注入过时的 src 或包内部目录；仍保留独立 Python 环境与子进程隔离。权重优先以 `weights_only=True` 读取，保留旧 Torch 兼容路径；这些检查不验证权重来源、哈希或实验协议成员。

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

历史任务值 `offline_5class` 在模型 registry 与离线输入合同中作为旧资产兼容别名保留，不作为新任务写入。旧 3／4／5 类标签缺省以及未写 num_classes 时的 5 类缺省仍兼容。

## Remaining Refactors

`telemetry_manager.py`、`telemetry_backends.py` 与 `ai_assistant.py` 的后续深 Module 治理独立进行；`uav_fault.models` 继续作为稳定 package Interface。
