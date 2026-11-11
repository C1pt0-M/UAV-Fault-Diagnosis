# Dashboard Backend

Flask + SQLite 后端，提供认证、资产管理、离线诊断任务、实时监测、诊断报告和 AI 助手 API。

## 目录

```text
apps/dashboard/backend/
├─ uav_dashboard_api/   # Flask package
├─ tests/               # backend tests
├─ wsgi.py              # WSGI entrypoint
└─ runtime/             # local runtime output, ignored by Git
```

## 安装

Python 依赖的唯一事实来源是仓库根目录的 `pyproject.toml`。

```powershell
cd D:\UAV-Fault-Diagnosis
python -m venv .\apps\dashboard\backend\.venv
.\apps\dashboard\backend\.venv\Scripts\python.exe -m pip install -e ".[dashboard,test]"
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

- `APP_ENV=development|production|testing`：选择实际加载的配置 Profile。
- `ENABLE_SETUP_API=True|False`：控制 `/api/setup/*` 是否注册；首次初始化后应关闭。
- `ENABLE_PUBLIC_REGISTER=False`：V1.0 默认采用管理员控制的账号访问，不开放公开注册。
- `MODEL_ROOT=assets/models`
- `OFFLINE_NORM_STATS=assets/models/transfer3_cnnlstm_normalproto/norm_stats.json`
- `AI_KNOWLEDGE_DIR=knowledge_base`
- `RUNTIME_DIR=runtime`
- `TELEMETRY_ALLOWED_BACKENDS=replay,rfly`
- `RFLY_PYTHON_BIN=D:\PX4PSP\Python38\python.exe`
- `RFLY_PX4MAVCTRL_PATH=D:\PX4PSP\RflySimAPIs\RflySimSDK\ctrl\PX4MavCtrlV4.py`
- `RFLY_BRIDGE_MODE=attach`：连接已启动的 RflySim/PX4。
- `RFLY_BRIDGE_MODE=buaa`：由 `BuaaAutoTestPlatform` bat 启动仿真并执行复现 case。
- `RFLY_BUAA_AUTOTEST_DIR=D:\BuaaAutoTestPlatForm\AutoTestAPI`
- `RFLY_BUAA_UNIQUE_INDEX=1419000188`

运行时权重和 metadata 在 `assets/models/`，由 `assets/models/manifest.json` 控制注册清单。模型结构代码仍在 `uav_fault.models`。

## 离线诊断执行

`uav_dashboard_api.services.offline_diagnosis` 是离线诊断 Module，公开
`submit_offline_diagnosis(...)` Interface，负责输入校验、监测会话承接、创建和提交任务，以及后台推理、严重度与决策拼装。调用需要 Flask application context；成功返回 `RunRecord` 表示任务已提交，最终结果通过现有任务查询获取。

路由负责请求解析和 HTTP 响应。共享资产输入逻辑位于
`services/flight_log_asset_input.py`，供资产、分析与诊断使用。提交前检查和后台执行前复查使用相同输入规则，保留排队期间文件或模型发生变化的失败语义。

`task_manager` 继续管理 executor、任务状态和结果落库，`bridge` 保留推理子进程 Adapter，报告由成功诊断结果单独生成。编排回归测试位于
`tests/dashboard_flows/test_offline_diagnosis.py`：使用临时数据库和 CSV、真实规则与决策计算，仅替代耗时推理，并覆盖真实 executor。

## AI 知识检索

`knowledge_base.retrieve_for_interpretation(query, candidate_limit=5, max_items=2)`
返回同一次查询的 `KnowledgeContextResult`：`hits`、`actual_mode` 和
`index_embedding_enabled`。AI 解释流程使用这些命中和元数据，不再自行解释检索分数或再次读取索引状态；原 `search(query, top_k=5)` 继续返回候选列表。

筛选保持原有顺序：检索侧执行意图约束、来源限额、四位分数舍入、排序与候选截断，再以
`max(0.35, top_score * 0.6)` 筛选分数、检查标题与 snippet 质量，最后应用上下文上限。计算最高分时仍包含随后可能因质量被拒绝的候选，过滤后不额外补足候选。

`used_context.knowledge_retrieval_mode` 记录本次实际模式，因此语义查询降级到 BM25 后即使零命中也记录为 `bm25`；知识库关闭、空查询或空索引未执行检索时为 `null`。
`knowledge_embedding_enabled` 仍表示索引能力，`fallback_used` 仍仅表示回答模型的降级。

`tests/test_knowledge_retrieval_context.py` 使用合成索引与固定 embedding/chat Adapter，对照重构前快照验证命中、引用和提示词上下文，并覆盖降级、零命中及并发查询元数据。测试不调用真实外部提供方。

## RflySim 运行边界

- Dashboard backend/frontend 从 Windows PowerShell 启动。
- RflySim GUI 层在 Windows 侧运行，包括 `CopterSim.exe`、`RflySim3D.exe` 和 `QGroundControl.exe`。
- PX4 SITL 可由 `scripts/start_rflysim_sitl_qgc.ps1` 拉起到 `RflySim-20.04` WSL 发行版；不要在 WSL 里直接启动 GUI 仿真层。
- QGC 是人工操控入口。脚本会为 QGC 写入 UDP auto link，默认端口 `14600`；Dashboard 的 MAVLink 补充监听默认使用 `14031`。
- 前端选择 `RflySim 实时链路` 后，后端按 `RFLY_BRIDGE_MODE` 决定 attach 已有仿真，或通过 BuaaAutoTestPlatform 自动复现 case。

## Rfly/Buaa 采集后复核

前端采集完成后，可对生成的 `TS-*.csv` 做字段健康检查和离线 replay 对齐：

```powershell
cd D:\UAV-Fault-Diagnosis
.\apps\dashboard\backend\.venv\Scripts\python.exe .\scripts\eval\eval_telemetry_live.py `
  --csv-dir .\apps\dashboard\backend\runtime\telemetry_sessions `
  --glob TS-416582.csv `
  --backend replay `
  --model-dir .\assets\models\online_detector_kg_mstca_rfly `
  --every 50 `
  --out .\runtime\rfly_replay_check.csv
```

重点看输出中的 `input_contract_status`、`input_contract_detection_blocked`、`mean_fault_prob`、`max_fault_prob` 和事件级指标。

## 实时概率偏高诊断

如果 No Fault 或起飞前出现高故障概率，先跑 schema drift 诊断，定位是输入合同被阻断、字段单位/别名错误，还是真实分布超出训练集：

```powershell
cd D:\UAV-Fault-Diagnosis
.\apps\dashboard\backend\.venv\Scripts\python.exe .\scripts\eval\diagnose_live_schema_drift.py `
  --csv .\apps\dashboard\backend\runtime\telemetry_sessions\TS-416582.csv `
  --model-dir .\assets\models\online_detector_kg_mstca_rfly `
  --backend replay `
  --every 50 `
  --out .\runtime\live_schema_drift_summary.csv `
  --windows-out .\runtime\live_schema_drift_windows.csv
```

重点看 `input_contract_detection_blocked`、`full_file_top_z`、`max_window_top_z_feature`。起飞前地面静止窗口属于当前训练集未覆盖的 operating envelope，后端会通过 input contract 暂停检测，不把这类窗口送入模型。

本地 `.env`、`runtime/`、`logs/`、`.venv/` 均不应提交。

首次部署建议：复制 `.env.example`，保持 `ENABLE_SETUP_API=True` 完成管理员初始化，
随后改为 `False` 并重启后端。生产模式不会再静默使用 development 配置。
