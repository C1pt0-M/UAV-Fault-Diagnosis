# Realtime Bridge

当前目录用于放置 `RflySim` 官方环境到项目实时检测链的桥接实现。

## 当前状态

- 已落地第一版桥接骨架：
  - [rfly_live_bridge.py](D:\UAV-Fault-Diagnosis\realtime\rfly_live_bridge.py)
- 已覆盖最小合同测试：
  - [test_rfly_live_bridge.py](D:\UAV-Fault-Diagnosis\tests\test_rfly_live_bridge.py)

当前 bridge 文件内已包含三个核心对象：

- `RflyLiveBridge`
  - 负责把官方环境状态拼成 telemetry JSON 帧
- `PymavStateSource`
  - 负责启动/停止 `pymavlink` 补采线程，并提供 snapshot
- `RflyLiveEmitter`
  - 负责周期性读取 `mav` 对象和 `pymav` 状态，并发到 UDP

同时已提供附着式运行入口：

- [rfly_live_attach.py](D:\UAV-Fault-Diagnosis\realtime\rfly_live_attach.py)
  - 负责只读附着到官方 `PX4MavCtrlV4`
- [run_rfly_live_bridge.py](D:\UAV-Fault-Diagnosis\realtime\run_rfly_live_bridge.py)
  - 负责把附着会话、`pymav` 补采和 emitter 串起来

## 第一版范围

第一版只固定最小 live 字段合同和 JSON 发包骨架，不直接启动官方环境。

最小字段包括：

- `seq`
- `time_s`
- `motor1..4`
- `roll/pitch/yaw`
- `rollspeed/pitchspeed/yawspeed`
- `accel_x/y/z`
- `gyro_x/y/z`
- `vx/vy/vz`
- `pos_z`

bridge 还支持叠加第二批增强字段：

- `baro_*`
- `mag_*`
- `actuator_ctrl_*`
- `rates_sp_*`
- `thrust_z`
- `battery_warning/system_status`
- `vibration/esc_rpm`

## 后续步骤

1. 接入官方 `PX4MavCtrlV4` 对象实例
2. 增加一条 `pymavlink` 补采线程
3. 周期性输出 JSON 到 `127.0.0.1:15550`
4. 接入现有 `telemetry.receiver -> telemetry_manager` 主链

## 当前推荐入口

当官方环境已经跑起来之后，推荐使用：

```powershell
python realtime/run_rfly_live_bridge.py `
  --px4mavctrl D:\BuaaAutoTestPlatForm\AutoTestAPI\PX4MavCtrlV4.py `
  --udp-port 20100 `
  --udp-mode 2 `
  --host 127.0.0.1 `
  --port 15550
```

说明：

- 这是“附着式 bridge”，不改官方代码。
- 默认附着 `CopterID=1` 的 `20100` 系列端口。
- 默认同时启用 `pymavlink` 补采；如需只测最小字段，可追加 `--no-pymav`。


## ???????

?????????????????? `mav` ?????????

```powershell
python realtime/run_buaa_realtime_bridge.py `
  --autotest-dir D:\BuaaAutoTestPlatForm\AutoTestAPI `
  --unique-index 1419000188 `
  --udp-port 20100 `
  --host 127.0.0.1 `
  --port 15550
```

???

- ??????????? `PX4MavCtrlV4 + Off_ctrl_connect_copter`?
- ??????????????????? `mav` ???? bridge?
- ???????? listener?????????????????????????? live ???
- ????? `pymavlink` ????? `--no-pymav`?
