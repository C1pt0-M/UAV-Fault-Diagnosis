# RflyMAD 数据工作流

## 当前结论
- 当前训练/验证直接使用 `data/processed/` 下的 processed CSV。
- 当前唯一需要长期保留的训练输入目录是 `/home/jack/UAV-Fault-Diagnosis/data/processed/`。
- 原始数据目录 `/home/jack/RflyMAD` 只在需要重新处理时才需要，平时可以删除；要复现流程时再重新准备一份干净原始数据。
- 当前 processed 结果为 **4950 / 4984**，缺失 **34** 个，且**全部位于 Real**。
- 当前训练策略保持为：**SIL + HIL 训练，Real 验证**。

## 最小可复用流程

### 输入与输出
- 原始数据输入：`/home/jack/RflyMAD`
- 项目内重组输出：`/home/jack/UAV-Fault-Diagnosis/data/`
- 最终训练输入：`/home/jack/UAV-Fault-Diagnosis/data/processed/`
- 官方处理工具：`/home/jack/Data_processing_tools/Rflytool_main.py`

### 推荐复现顺序
1. 准备一份干净的原始 RflyMAD 数据到 `/home/jack/RflyMAD`。
2. 运行项目脚本把原始目录重组为官方工具可识别的结构：
   ```bash
   python /home/jack/UAV-Fault-Diagnosis/scripts/reorganize_rflymad.py
   ```
3. 使用官方处理工具或项目封装脚本生成 processed CSV：
   ```bash
   bash /home/jack/UAV-Fault-Diagnosis/scripts/process_all_final.sh
   ```
4. 处理完成后，以 `data/processed/` 作为训练/验证输入。
5. 查看结果总结文件确认缺失情况：
   - `data/processed/final_missing_summary.txt`
   - `data/processed/failed_real_cases.txt`

## 当前最终状态
- Processed CSV count: **4950 / 4984**
- Total missing: **34**
- 缺失全部在 `Real`
- `SIL`: complete
- `HIL`: complete
- `Real`: 缺 34
- 缺失详情与已确认坏样本不要手工复制维护，直接以这两个文件为准：
  - `data/processed/final_missing_summary.txt`
  - `data/processed/failed_real_cases.txt`

## 官方编号与 processed 文件命名

### 1. CLI 参数编号（官方工具输入）
依据 `Data_processing_tools/get_parse.py`：

#### Sub-dataset
| CLI 值 | 含义 |
|---|---|
| 1 | SIL |
| 2 | HIL |
| 3 | Real |

#### Flight status
| CLI 值 | 含义 |
|---|---|
| 1 | hover |
| 2 | waypoint |
| 3 | velocity |
| 4 | circling |
| 5 | acce |
| 6 | dece |

#### Fault type
| CLI 值 | 含义 |
|---|---|
| 1 | motor |
| 2 | propeller |
| 6 | accelerometer |
| 7 | gyroscope |
| 8 | magnetometer |
| 9 | barometer |
| 10 | GPS |
| 11 | no_fault |

> 备注：`3=low_voltage`、`4=wind_affect`、`5=load_lose` 存在于官方工具中，但当前项目四分类训练不使用。

### 2. processed 文件内部编号（输出文件名）
依据 `Data_processing_tools/get_parse.py` 中 `flight_id_dict()` 与 `fault_id_dict()`，processed 文件名中的 `caseID` 采用下面的内部编码：

```text
caseID = data_type * 1e9 + flight_mode * 1e8 + fault_type * 1e6 + case_num
```

其中：

#### Sub-dataset file-id
| file-id | 含义 |
|---|---|
| 1 | SIL |
| 2 | HIL |
| 3 | Real |

#### Flight file-id
| file-id | 含义 |
|---|---|
| 0 | hover |
| 1 | waypoint |
| 2 | velocity |
| 3 | circling |
| 4 | acce |
| 5 | dece |

#### Fault file-id
| file-id | 含义 |
|---|---|
| 0 | motor |
| 1 | propeller |
| 5 | accelerometer |
| 6 | gyroscope |
| 7 | magnetometer |
| 8 | barometer |
| 9 | GPS |
| 10 | no_fault |

### 3. 示例：`Case_3109000005.csv`
按上述规则拆分：
- `3` → Real
- `1` → waypoint
- `09` → GPS
- `000005` → case index 5

因此 `Case_3109000005.csv` 表示：**Real / waypoint / GPS / case 5**。

## 易混点
- `CLI fault_type=1` 表示 `motor`，但 processed 文件里的 `motor` 编码是 `0`。
- `CLI flight_status=2` 表示 `waypoint`，但 processed 文件里的 `waypoint` 编码是 `1`。
- 因此：**CLI 参数编号 ≠ processed 文件名内部编号**。
- processed 文件中的 `case_num` 是该故障-飞行状态组合内的案例序号，不应直接等同于原始目录名称中的编号语义。
- `Real` 当前没有 `propeller` 数据；相关批处理脚本也按此约束运行。

## 项目四分类映射
当前训练标签统一为四分类：

| 原始故障 | 训练标签 |
|---|---|
| no_fault | 0 |
| motor | 1 |
| propeller | 2 |
| accelerometer | 3 |
| gyroscope | 3 |
| magnetometer | 3 |
| barometer | 3 |
| GPS | 3 |

说明：
- 传感器相关故障统一并入 `sensor fault`（标签 3）。
- 当前项目显式排除：`low_voltage`、`wind_affect`、`load_lose`。
- 训练使用 `SIL + HIL`，`Real` 主要用于跨分布验证，因此 Real 缺少 34 个 processed 文件对当前主训练流程影响可接受。

## 脚本分层说明

### 核心可复用脚本

#### `scripts/reorganize_rflymad.py`
- 用途：把原始 RflyMAD 目录重组为官方工具要求的 `SIL/HIL/Real/flight/fault/case` 结构。
- 输入：`/home/jack/RflyMAD`
- 输出：`/home/jack/UAV-Fault-Diagnosis/data/`
- 实现方式：大量使用符号链接，不复制 case 目录内容。
- 风险/限制：脚本会先删除旧的 `data/` 目录再重建，因此重跑前要确认该目录没有其他需要保留的内容。

### 可复用但带上下文的处理脚本

#### `scripts/process_all_final.sh`
- 用途：按 fault 粗粒度批量调用 `Rflytool_main.py` 生成 processed CSV，是当前最接近“最终可复现入口”的处理脚本。
- 输入：`/home/jack/UAV-Fault-Diagnosis/data/`
- 输出：`/home/jack/UAV-Fault-Diagnosis/data/processed/`
- 依赖：
  - `/home/jack/Data_processing_tools/`
  - Conda 环境 `UAV-Fault-Diagnosis`
- 特点：
  - 使用 `--flight_status 0`，一次覆盖某 sub-dataset + fault 下的全部飞行状态。
  - 处理后会删除原始目录中自动转出的中间 CSV，以缓解磁盘压力。
- 限制：
  - 假定 `Real` 无 `propeller`。
  - 会覆盖已存在的 processed CSV，但这不影响最终结果统计。

### 历史补跑 / 清理脚本
以下脚本保留用于复现历史补救路径，但**默认不作为第一次处理入口**：

#### `scripts/process_missing_only.sh`
- 用途：面向“缺失部分”做定向补跑。
- 适用场景：已有一轮处理结果后，只想重试缺失组合。
- 限制：依赖当时的缺失分布假设，默认不适合作为全量首跑脚本。

#### `scripts/process_remaining.sh`
- 用途：按 `SIL → HIL → Real` 对剩余组合逐项处理，并给单次任务加 5 分钟超时。
- 适用场景：早期全量/半全量重试。
- 限制：更偏历史过程脚本，不如 `process_all_final.sh` 简洁。

#### `scripts/process_and_cleanup.sh`
- 用途：边处理边清理中间 CSV，适合磁盘空间紧张时使用。
- 适用场景：磁盘吃紧时的分批处理。
- 限制：脚本中的批次划分与当时处理进度强相关，属于历史补救方案。

#### `scripts/cleanup_rflymad_csv.sh`
- 用途：清理原始 RflyMAD 目录下由 ULog 转出的中间 CSV，只保留后续仍需的少量 CSV。
- 风险：会删除中间产物，属于显式清理脚本；使用前应确认 processed 输出已保留。

## 依赖说明
- 官方工具文档：`/home/jack/Data_processing_tools/README.md`
- 官方参数定义：`/home/jack/Data_processing_tools/get_parse.py`
- 官方处理主入口：`/home/jack/Data_processing_tools/Rflytool_main.py`

## 使用建议
- 日常训练/验证时，**不要再依赖原始数据目录**，直接从 `data/processed/` 读取。
- 只有在以下场景才需要重新准备 `/home/jack/RflyMAD`：
  - 重新全量处理
  - 修改官方处理参数
  - 重新选择 topic / frequency
- 如果只是查看当前缺失样本与坏样本，直接读：
  - `data/processed/final_missing_summary.txt`
  - `data/processed/failed_real_cases.txt`
