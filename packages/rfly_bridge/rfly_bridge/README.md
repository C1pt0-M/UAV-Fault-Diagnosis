# Rfly Bridge

`rfly_bridge` 负责把 RflySim / PX4 官方控制对象转换为项目内部 telemetry JSON 帧，并通过 UDP 转发给 Dashboard 后端。

## 入口

```powershell
cd D:\UAV-Fault-Diagnosis
python -m rfly_bridge `
  --px4mavctrl D:\PX4PSP\RflySimAPIs\RflySimSDK\ctrl\PX4MavCtrlV4.py `
  --udp-port 20100 `
  --udp-mode 2 `
  --host 127.0.0.1 `
  --port 15550
```

## 核心对象

- `RflyLiveBridge`：构建 canonical telemetry frame 并转发。
- `PymavStateSource`：可选 MAVLink 补充字段监听。
- `RflyLiveEmitter`：按固定频率从 `mav` / `pymav` 读取并发送。
- `RflyBridgeRuntime`：附着官方 `PX4MavCtrlV4.py` 的 CLI runtime。

## 常用参数

- `--px4mavctrl`：官方 `PX4MavCtrlV4.py` 路径。
- `--host` / `--port`：Dashboard telemetry receiver 地址。
- `--emit-rate`：发送频率，默认 `50`。
- `--no-pymav`：关闭 MAVLink 补充字段。
- `--duration`：运行秒数，`0` 表示持续运行。
