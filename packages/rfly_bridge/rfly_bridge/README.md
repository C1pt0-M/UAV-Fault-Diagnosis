# Rfly Bridge

`rfly_bridge` 负责把 RflySim / PX4 官方控制对象转换为项目内部 telemetry JSON 帧，并通过 UDP 转发给 Dashboard 后端。

## 入口

运行边界：

- 在 Windows PowerShell 中启动 Dashboard 和 RflySim GUI。
- `CopterSim.exe`、`RflySim3D.exe`、`QGroundControl.exe` 均在 Windows 侧运行。
- PX4 SITL 使用 `RflySim-20.04` WSL 发行版承载；GUI 仿真层不从 WSL 启动。
- QGC 负责人工操控，Dashboard 只负责采集、桥接和在线检测。

Attach 到已启动的 RflySim/PX4：

```powershell
cd D:\UAV-Fault-Diagnosis
python -m rfly_bridge `
  --px4mavctrl D:\PX4PSP\RflySimAPIs\RflySimSDK\ctrl\PX4MavCtrlV4.py `
  --udp-port 20100 `
  --udp-mode 2 `
  --host 127.0.0.1 `
  --port 15550
```

BuaaAutoTestPlatform 自动复现链路：

```powershell
cd D:\UAV-Fault-Diagnosis
python .\packages\rfly_bridge\rfly_bridge\run_buaa_realtime_bridge.py `
  --autotest-dir D:\BuaaAutoTestPlatForm\AutoTestAPI `
  --unique-index 1419000188 `
  --host 127.0.0.1 `
  --port 15550 `
  --duration 60
```

## 核心对象

- `RflyLiveBridge`：构建 canonical telemetry frame 并转发。
- `PymavStateSource`：可选 MAVLink 补充字段监听。
- `RflyLiveEmitter`：按固定频率从 `mav` / `pymav` 读取并发送。
- `RflyBridgeRuntime`：附着官方 `PX4MavCtrlV4.py` 的 CLI runtime。
- `BuaaRealtimeBridgeRunner`：启动 BuaaAutoTestPlatform bat、控制序列和实时 telemetry 转发。

## 常用参数

- `--px4mavctrl`：官方 `PX4MavCtrlV4.py` 路径。
- `--host` / `--port`：Dashboard telemetry receiver 地址。
- `--emit-rate`：发送频率，默认 `50`。
- `--no-pymav`：关闭 MAVLink 补充字段。
- `--duration`：运行秒数，`0` 表示持续运行。
- `--autotest-dir`：BuaaAutoTestPlatform 的 `AutoTestAPI` 目录。
- `--unique-index`：复现 case 的 `UniqueIndex`。
