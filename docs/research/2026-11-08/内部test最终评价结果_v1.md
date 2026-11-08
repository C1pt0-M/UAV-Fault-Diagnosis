# 内部test最终评价结果 v1

## 1. 结论：评分前环境失败，未评价

**唯一一次正式启动已停止，没有模型推理、没有任何test性能指标。不能给出CONFIRMED或NOT_CONFIRMED。未重启、未修依赖后重跑。**

冻结协议：`configs/experiments/internal_test_final_v1.json`，冻结提交`e475a42c094c3e2afa5fc047a10fc58b417d691e`，配置原始SHA256：

```text
7293a62e00f71b98c5fae892148ded486d0438aa832f180d5b32e2e156b5d51d
```

本报告如实记录第二次、最后一次test使用的失败尝试，不把环境失败伪装成科学“不确认”，也不以未产生指标为理由重新打开test。

## 2. 启动检查与一致性门

三个资产SHA全部匹配：

|资产|实际SHA256|
|---|---|
|GBT全量冻结模型|`df468524e2047205dd8422ab3e486ce7692f0ac312276e20394d6f88a1bb6774`|
|M0部署基线|`ac8596cd2e762a4b50845b79e312b1588e68956839a0e9c3573f88a05b5fc6cb`|
|已有窗口预测|`45360dbf5e136dc30e881bf9989cb0307683fe1cd6ff19ce8cdfde886411e8f8`|

- test任务553个，GPS48个，mode全部可查。
- 排除集10个（SIL/HIL各5个），在评分前按hover/GPS/mode4固定名单。
- 真实association_group为277组，任务分组与34802条监督窗口台账一致，没有自造group。
- 两个目标输出目录启动时都不存在。旧锁存在、只读，其SHA和mtime均未变。
- 一致性门**未运行**：预期34802窗；实际对齐数、GBT最大差、M0最大差均为**不可用/null**，不是0。
- 新`TEST_EVALUATED.lock`没有写入，不能伪造一致性门通过。

## 3. 正式失败与根因

正式命令：

```text
D:\python3.12.4\python.exe -X utf8 -I -B scripts/research/internal_test_final_v1.py
```

失败阶段`FRAME_SCORING`，原始异常：

```text
ModuleNotFoundError: No module named 'threadpoolctl'
```

`score_cases`导入`score_ens_frames.load_m0`时，模板第11行导入`sklearn.ensemble`，再沿`sklearn.utils.parallel`导入`threadpoolctl`失败。尚未调用`load_m0`、`pickle.load`或任何案例循环，因此没有反序列化模型、读取test案例payload或执行推理。

`threadpoolctl.py`实际存在于用户site-packages：

```text
C:\Users\23119\AppData\Roaming\Python\Python312\site-packages\threadpoolctl.py
```

但本轮选择的`-I`隔离模式使其不可见。这是实现者选择隔离运行却未预检完整导入链的遗漏，不能笼统归咎为系统没装包。合成测试用假模型，未覆盖该导入边界。

外部评分目录`D:/UAV-Data/internal_test_final_v1_scores/`为空，保留现场。canonical输出目录也保留，脚本会拒绝重入。没有安装依赖、切换环境、调整模型/批量/数值路线或重新评分。

## 4. F1、F2：均未计算

|判据|GBT|M0|判定|
|---|---|---|---|
|F1 独立正常误报事件/有效曝光小时|未计算|未计算|不可判定|
|F2 去排除集后的检出案例数|未计算|未计算|不可判定|

科学结论为空。不是`NOT_CONFIRMED`：那一结论只能在完成注册指标后给出，不能用于替代基础设施故障。

## 5. F3与F4：均未计算

|类别|F3 M0/GBT召回|F3通过/失败|F4 M0中位/P90|F4 GBT中位/P90|慢>2s标记|
|---|---|---|---|---|---|
|GPS|不可用|未评价|不可用|不可用|未评价|
|accelerometer|不可用|未评价|不可用|不可用|未评价|
|barometer|不可用|未评价|不可用|不可用|未评价|
|gyroscope|不可用|未评价|不可用|不可用|未评价|
|magnetometer|不可用|未评价|不可用|不可用|未评价|
|motor|不可用|未评价|不可用|不可用|未评价|

全部共同检出案例的配对差中位数/P90也不可用。没有绕过一致性门，单独拿已有窗口概率计算F3或正常FP。

## 6. 描述性结果与未生成项

事件误报/曝光、窗口FP、两个bootstrap CI、SIL/HIL性能、按飞行目录性能、hover占用、故障前误报、spans_onset、排除集30s前后检出、后段mode4窗口召回和案例级检出均未计算。

协议已知ENS正常FP342只作为历史引用，没有重算，不冒充本轮结果。

没有逐帧npz，因此没有评分文件SHA；没有`scoring_receipt.json`或`events_per_case.csv`。不制造空事件表来暗示零事件。`parity_gate.json`和`results.json`仅记录未运行/未评价的行政状态，指标全null。

## 7. 测试、完整偏差与现场保全

- 合成测试一次：**26 passed，0 failed，0 skipped**。访问审计拒绝真实数据/旧test资产目录，实际访问尝试0；这不能替代真实评分依赖预检。
- 正式评价只启动一次，失败一次，无重启/恢复/重算。失败前没有任何指标，也未计算其他工作点。
- 进程detached运行，观察器未轮询；观察时进程已退出，Windows OpenProcess返回87，无法再取得OS退出码。该观察记录不是模型重启，也不虚报捕获了退出码。
- PI中“不读train/val以外的其他留出数据”的字面与本轮授权冲突，按冻结协议仅访问明确授权test元数据/已有预测/资产；未读Real或其他留出payload。
- 窗口台账末行为exclusive；拟定帧对齐为start+149=end−1，F3排除则按协议字面使用window_end_row×0.02>30。
- F4事先实现按完整故障总体及全部共同检出；F2/F3各用注册排除。事件bootstrap helper中的M0-tuned别名固定为同一0.60，不新增工作点。上述真实指标均未执行。
- 原始`receipt.json`、`run.log`及完整traceback不改写；8个受保护文件（旧锁、原模板、协议及用户已有脏脚本/日志）SHA和mtime不变，三个资产SHA也仍匹配。没有更改golden、部署工作点、冻结协议、结项报告或实验台账。
- docs/artifacts仅对明确新文件`git add -f`，不改忽略规则、不stage用户旧改动。

完整逐项记录见`artifacts/internal_test_final_v1/execution_notes.md`；完整原始异常见`failure_traceback.log`。没有隐藏重跑或额外test性能分析。

## 8. 交付

新增`internal_test_final_v1.py`、合成测试、本文、失败现场/回执/行政状态/审计。逐帧输出外部空目录保留，不入库。提交和push确认在最终回复提供，不把自身commit hash循环写入本文。
