# WSL 环境下 Codex 后续处理交接说明

## 背景

当前项目在 Windows PowerShell 下处理包含中文文案的 Python/Vue/Markdown 文件时，出现了明显的编码风险。问题不只是终端显示乱码，而是部分 PowerShell 文本读写操作可能把 UTF-8 文件重新写坏，导致中文字符串变成 mojibake，甚至产生 `SyntaxError: unterminated string literal`。

因此，后续建议切换到 WSL/Linux 环境继续修复，避免继续用 PowerShell 对源码文件做写回操作。

## WSL 环境前置检查

进入仓库后先确认编码环境：

```bash
pwd
locale
python3 - <<'PY'
import locale
print(locale.getpreferredencoding(False))
PY
```

期望输出包含 `UTF-8`。如果不是，先执行：

```bash
export LANG=C.UTF-8
export LC_ALL=C.UTF-8
```

后续文件修改优先使用 `apply_patch`。如果必须用脚本处理文件，必须显式指定 UTF-8，并在写回前确认目标范围。

## 当前未完成目标

用户要求修复前两个风险点：

1. 修复后端测试文件损坏、旧路径、旧包名导致的测试无法运行问题。
2. 降低后端超大模块维护风险。

第三个风险点暂缓，第四个 Rfly/PX4 实机集成仍在测试中，不作为本轮重点。

## 切换到 WSL 后 Codex 的任务单

进入 WSL 后，Codex 不需要重新做项目全量探索，直接按以下顺序继续：

1. 确认 Linux/WSL 环境是 UTF-8，并停止使用 PowerShell 对中文源码做写回。
2. 确认 `telemetry_manager.py` 已恢复到未污染状态，且 `py_compile` 通过。
3. 处理未完成拆分产生的新增文件：如果不立即完成并验证 `telemetry_payloads.py` 拆分，就删除这两个新增文件，避免半成品进入测试面。
4. 修复后端测试文件的语法错误、旧 import、旧路径和 mojibake 断裂字符串。
5. 先让后端测试目录全部 `compileall` 通过，再跑重点 pytest。
6. 测试稳定后，才重新评估第二风险点，即大模块拆分。拆分必须小步、纯函数优先、有测试覆盖。

本轮不要优先处理：

- 第三个风险点。
- Rfly/PX4 实机链路。
- 大范围重构。
- 为了让测试通过而修改生产行为。

## 修改文件造成的报错与处理方式

### 1. PowerShell 写回导致 UTF-8 文件污染

已发生现象：

- `apps/dashboard/backend/uav_dashboard_api/services/telemetry_manager.py` 曾被 PowerShell 写回污染。
- 中文字符串和注释出现 mojibake。
- Python 编译阶段出现无关的 `SyntaxError`，典型形式是 `unterminated string literal`。

当前处理状态：

- 已在 Windows 侧用 `git checkout -- apps/dashboard/backend/uav_dashboard_api/services/telemetry_manager.py` 恢复过该文件。
- 已在 Windows 侧确认该文件 `py_compile` 通过。

WSL 后仍必须再次验证：

```bash
git diff -- apps/dashboard/backend/uav_dashboard_api/services/telemetry_manager.py
python3 -m py_compile apps/dashboard/backend/uav_dashboard_api/services/telemetry_manager.py
```

期望：

- `git diff` 为空。
- `py_compile` 无输出并返回成功。

如果仍有 diff，先不要继续改该文件；需要先判断 diff 是否来自本轮有效修改，还是编码污染残留。

### 2. 未完成拆分引入半成品文件

已发生现象：

- 曾尝试从 `telemetry_manager.py` 抽出 helper 到 `telemetry_payloads.py`。
- 拆分未完成，且测试尚未稳定。
- 工作区出现两个 untracked 文件：

```text
apps/dashboard/backend/tests/test_telemetry_payloads.py
apps/dashboard/backend/uav_dashboard_api/services/telemetry_payloads.py
```

处理方式：

- 如果 WSL 后不继续完成该拆分，应删除这两个文件。
- 如果继续该拆分，必须先保证它只包含纯 payload 转换逻辑，并通过独立测试。

验证命令：

```bash
python3 -m py_compile apps/dashboard/backend/uav_dashboard_api/services/telemetry_payloads.py
python3 -m pytest -q apps/dashboard/backend/tests/test_telemetry_payloads.py
```

不能快速通过时，删除半成品：

```bash
rm apps/dashboard/backend/tests/test_telemetry_payloads.py
rm apps/dashboard/backend/uav_dashboard_api/services/telemetry_payloads.py
```

### 3. 测试文件被修复到一半

已发生现象：

- 多个测试文件已从旧包名、旧路径迁移到新包名、新 monorepo 路径，但尚未完整验证。
- 部分测试中的中文字符串断裂可能仍会造成语法错误。
- 部分测试可能还保留旧 frontend contract，例如 `Telemetry.vue`、`FaultDetection.vue`。

处理方式：

- 先用 `compileall` 找语法错误。
- 再用 `rg` 找旧 contract。
- 修复测试时优先改断言目标，不要保留乱码字符串断言。

检查命令：

```bash
python3 -m compileall -q apps/dashboard/backend/tests tests packages/uav_fault/uav_fault/online
rg -n "app\.services|app\.routes|Telemetry\.vue|FaultDetection\.vue|assert True|parents\[3\]" apps/dashboard/backend/tests
```

旧包名迁移规则：

```text
app.services... -> uav_dashboard_api.services...
app.routes... -> uav_dashboard_api.routes...
online.fault_detector -> uav_fault.online.fault_detector
```

## 当前工作树状态

切换 WSL 后第一步执行：

```bash
git status --short
```

目前已知存在大量测试文件修改，包括：

```text
apps/dashboard/backend/tests/test_ai_assistant.py
apps/dashboard/backend/tests/test_ai_assistant_answer_style.py
apps/dashboard/backend/tests/test_ai_assistant_context_summarization.py
apps/dashboard/backend/tests/test_ai_assistant_knowledge_relevance.py
apps/dashboard/backend/tests/test_ai_assistant_query_enrichment.py
apps/dashboard/backend/tests/test_ai_assistant_report_context_summary.py
apps/dashboard/backend/tests/test_ai_assistant_scene_guidance.py
apps/dashboard/backend/tests/test_ai_assistant_timeout_guard.py
apps/dashboard/backend/tests/test_dashboard_startup_scripts.py
apps/dashboard/backend/tests/test_diagnosis_report_pdf.py
apps/dashboard/backend/tests/test_knowledge_base_quality_filter.py
apps/dashboard/backend/tests/test_knowledge_base_rag.py
apps/dashboard/backend/tests/test_knowledge_base_source.py
apps/dashboard/backend/tests/test_knowledge_index_config_staleness.py
apps/dashboard/backend/tests/test_lazy_online_imports.py
apps/dashboard/backend/tests/test_main_flows.py
apps/dashboard/backend/tests/test_severity_recommendation.py
apps/dashboard/backend/tests/test_telemetry_manager_runtime.py
apps/dashboard/backend/tests/test_validators.py
```

还有两个新增文件来自未完成的大模块拆分尝试：

```text
apps/dashboard/backend/tests/test_telemetry_payloads.py
apps/dashboard/backend/uav_dashboard_api/services/telemetry_payloads.py
```

这两个新增文件需要重新评估。建议先不要继续扩大拆分，优先恢复测试可运行。如果拆分没有完成并验证通过，应删除这两个新增文件。

## 已经发生过的问题

曾尝试从 `telemetry_manager.py` 抽出 helper 到 `telemetry_payloads.py`。过程中 PowerShell 写回文件造成 `telemetry_manager.py` 编码污染，出现大量无关 `SyntaxError`。

已在 Windows 侧执行过：

```bash
git checkout -- apps/dashboard/backend/uav_dashboard_api/services/telemetry_manager.py
```

并确认：

```bash
python -m py_compile apps/dashboard/backend/uav_dashboard_api/services/telemetry_manager.py
```

通过。

WSL 下仍需再次确认该文件没有残留修改：

```bash
git diff -- apps/dashboard/backend/uav_dashboard_api/services/telemetry_manager.py
python3 -m py_compile apps/dashboard/backend/uav_dashboard_api/services/telemetry_manager.py
```

期望：没有 diff，编译通过。

## 第一优先级：修复测试可运行

先不要做生产代码重构。执行：

```bash
python3 -m compileall -q apps/dashboard/backend/tests tests packages/uav_fault/uav_fault/online
```

如果有语法错误，逐个修复测试文件。重点关注：

- `apps/dashboard/backend/tests/test_main_flows.py`
- `apps/dashboard/backend/tests/test_telemetry_manager_runtime.py`
- 其他 AI/RAG/PDF 测试文件中的 mojibake 断裂字符串

修复原则：

- 断裂中文字符串如果只是 message 文案断言，可改成结构断言，例如 `status_code`、`success`、`data`、`message` 非空。
- 不要用乱码字符串继续做断言。
- 不要为了过测试改生产行为。
- 旧包名需要迁移：
  - `app.services...` -> `uav_dashboard_api.services...`
  - `app.routes...` -> `uav_dashboard_api.routes...`
  - `online.fault_detector` -> `uav_fault.online.fault_detector`
- 旧前端路径需要迁移：
  - 旧：`apps/dashboard/frontend/src/views/Telemetry.vue`
  - 现：`apps/dashboard/frontend/src/modules/monitoring/...`
  - 旧：`apps/dashboard/frontend/src/views/FaultDetection.vue`
  - 现：`apps/dashboard/frontend/src/modules/diagnosis/...`

## 重点检查 test_main_flows.py

该文件已做过大量临时修复，但需要在 WSL 下重新审查：

```bash
python3 -m py_compile apps/dashboard/backend/tests/test_main_flows.py
rg -n "app\.services|app\.routes|Telemetry\.vue|FaultDetection\.vue|assert True|parents\[3\]" apps/dashboard/backend/tests/test_main_flows.py
```

如果看到被三引号包起来的旧 frontend contract 测试块，确认是否保留。更干净的处理方式是删除旧块，只保留当前 `modules/*` 结构的有效 contract 测试。

## 重点检查 test_telemetry_manager_runtime.py

执行：

```bash
python3 -m py_compile apps/dashboard/backend/tests/test_telemetry_manager_runtime.py
rg -n "app\.services|assert True|unterminated|online\.fault_detector" apps/dashboard/backend/tests/test_telemetry_manager_runtime.py
```

确认所有 monkeypatch 目标都指向当前包名：

```text
uav_dashboard_api.services.telemetry_manager
uav_dashboard_api.services.detection_runner
```

## 第二优先级：大模块风险处理

在测试恢复稳定前，不建议继续拆 `telemetry_manager.py`。

更稳妥的处理顺序：

1. 先让测试全部可编译。
2. 跑核心测试。
3. 若要降低大模块风险，先做无行为变化的小拆分，并新增独立测试。
4. 每次只拆一个纯函数/纯转换模块。

建议优先拆分候选：

- `telemetry_manager.py` 中纯 payload 转换函数。
- 纯 recommendation 构造函数。
- 不依赖 Flask app、线程、进程、socket 的 helper。

不要优先拆：

- PX4/Rfly 启停流程。
- 线程生命周期。
- `_on_backend_frame`。
- 数据库持久化流程。

如果继续拆分 `telemetry_payloads.py`，必须先确认：

```bash
python3 -m py_compile apps/dashboard/backend/uav_dashboard_api/services/telemetry_manager.py
python3 -m py_compile apps/dashboard/backend/uav_dashboard_api/services/telemetry_payloads.py
python3 -m pytest -q apps/dashboard/backend/tests/test_telemetry_payloads.py
```

如果无法快速完成，应删除未完成新增文件：

```bash
rm apps/dashboard/backend/tests/test_telemetry_payloads.py
rm apps/dashboard/backend/uav_dashboard_api/services/telemetry_payloads.py
```

## 建议验证命令

先编译：

```bash
python3 -m compileall -q apps/dashboard/backend/uav_dashboard_api apps/dashboard/backend/tests tests packages/uav_fault/uav_fault
```

再跑重点测试：

```bash
python3 -m pytest -q apps/dashboard/backend/tests/test_main_flows.py
python3 -m pytest -q apps/dashboard/backend/tests/test_telemetry_manager_runtime.py
python3 -m pytest -q apps/dashboard/backend/tests/test_lazy_online_imports.py
```

如果项目使用 backend venv，优先使用对应解释器，例如：

```bash
./apps/dashboard/backend/.venv/bin/python -m pytest -q apps/dashboard/backend/tests/test_main_flows.py
```

Windows venv 不能直接在 WSL 中使用。如果没有 Linux venv，需要在 WSL 下重新建环境。

## 禁止事项

- 不要在 PowerShell 中继续对中文源码文件执行写回操作。
- 不要用 `Get-Content` / `Set-Content` / `[System.IO.File]::WriteAllLines` 批量改源码。
- 不要盲目全局替换中文乱码。
- 不要在测试仍不可编译时继续做生产代码拆分。
- 不要把生产文件的中文注释/文案改成乱码。

## 完成标准

本轮至少应完成：

1. `telemetry_manager.py` 无修改或可解释的最小修改，且编译通过。
2. 后端测试文件全部 `compileall` 通过。
3. 旧包名和旧前端路径断言清理完成。
4. 重点 pytest 至少覆盖：
   - `test_main_flows.py`
   - `test_telemetry_manager_runtime.py`
   - `test_lazy_online_imports.py`
5. 如果第二风险点暂缓，需要明确记录原因：先恢复测试稳定性，再拆大模块。
