# RflySim 官方环境实时桥接方案草稿

## 1. 目标

本说明用于固定当前结论：

- 仿真环境继续使用 `RflySim` 官方环境。
- 故障注入语义继续沿用 `D:\BuaaAutoTestPlatForm`。
- 实时检测链路由项目自行搭建，不直接复用 `BuaaAutoTestPlatForm` 的离线目录产出流程。
- 目标不是复刻离线数据集生成链，而是在运行时尽量生成更接近 `RflyMAD` 分布的 live telemetry。

一句话概括：

> `RflySim` 负责飞和注入故障，项目负责实时取数、在线检测和 Dashboard 展示。

## 2. 为什么不能直接把官方复现工具当实时入口

`D:\BuaaAutoTestPlatForm` 的主职责是：

1. 启动仿真环境
2. 按 `ControlSequence` 跑测试用例
3. 注入故障
4. 保存原始 case 目录（`Log/`、`TrueData/`、`TestInfo.csv` 等）

它更接近“原始数据采集与复现工具”，而不是实时检测服务。

因此：

- 它适合生成官方风格的离线原始样本。
- 它不适合作为在线 detector 的直接输入链路。
- 实时检测必须在运行时从官方环境直接取数，而不是等待 `TestCase_*` 落盘后再处理。

## 3. 实时线路的推荐拆分

推荐拆成两条并行线路：

### 3.1 离线数据生成线

```text
RflySim + BuaaAutoTestPlatForm
  -> TestCase_* 原始目录
  -> Data_processing_tools / Rflytool_main.py
  -> Case_*.csv
  -> manifest/cache/train
```

### 3.2 实时检测线

```text
RflySim 官方环境运行中
  -> rfly_live_bridge
  -> UDP JSON (127.0.0.1:15550)
  -> telemetry.receiver
  -> telemetry_manager canonical 主链
  -> online detector
  -> event aggregation
  -> Dashboard
```

当前要优先推进的是 **3.2 实时检测线**。

## 4. 必须保留的官方元素

为了让实时数据尽量接近 `RflyMAD` 的生成机制，以下官方元素必须尽量保留。

### 4.1 FaultType 编码体系

继续沿用官方 fault id，不重新发明一套编码。

当前已确认的主要编码包括：

- `123450` ~ `123459`
- `123540` ~ `123549`
- `125340`、`125341`

其中：

- `123549` 当前被用作 `No Fault`

### 4.2 Subsystem 体系

继续沿用官方子系统命名，用于 case 标注、实时事件展示和风险分组：

- `Power subsystem`
- `Sensor subsystem`
- `GPS subsystem`
- `Integrated subsystem`
- `Environment subsystem`
- `Pod subsystem`

### 4.3 ControlSequence 语法

继续沿用 `db.json + command.py` 里的控制序列语义，不另起一套动作描述语言。

需要保留的核心动作包括：

- `2,1,...`：起飞流程
- `1,1,...`：等待
- `2,3,...`：位置控制
- `2,4,...`：速度控制
- `2,9,...`：航点轨迹
- `2,10,...`：加减速控制
- `2,11,...`：绕圈
- `2,6,...`：故障注入

### 4.4 FaultModel 故障模型

尽量继续使用官方 `FaultModel`，不要退化成“纯 PX4 原生注入”。

关键路径：

- `D:\BuaaAutoTestPlatForm\FaultModel\FaultModel.slx`
- `D:\BuaaAutoTestPlatForm\FaultModel\FaultModel450d_new.dll`
- `D:\BuaaAutoTestPlatForm\FaultModel\GenerateModelDLLFile.p`

### 4.5 官方 topic 保留集合

当前官方 `db.json.testdata` 保留的 topic 应继续作为实时 bridge 的核心观测面：

- `actuator_outputs`
- `estimator_status`
- `vehicle_command`
- `sensor_combined`
- `vehicle_magnetometer`
- `estimator_attitude`
- `vehicle_gps_position`
- `vehicle_global_position`

## 5. rfly_live_bridge 的定位

新增一个独立的 bridge 层，建议命名为：

- `rfly_live_bridge.py`

职责只包括：

1. 附着到官方环境运行时
2. 读取官方环境中的 live 状态
3. 补充少量 MAVLink 消息字段
4. 组装成当前项目的 telemetry JSON
5. 通过 UDP 发给 `127.0.0.1:15550`

它 **不负责**：

- 生成 `TestCase_*`
- 目录重组
- `Case_*.csv` 落盘
- 训练数据处理

## 6. bridge 第一版最小字段集

第一版的目标是先把实时链路跑通，不追求完整覆盖。

### 6.1 必须字段

```text
seq
time_s
motor1 motor2 motor3 motor4
roll pitch yaw
rollspeed pitchspeed yawspeed
accel_x accel_y accel_z
gyro_x gyro_y gyro_z
vx vy vz
pos_z
```

### 6.2 第一版目标

第一版只要做到：

- `receiver` 能持续收包
- `telemetry_manager` 能 canonicalize
- 31 通道能基本补齐
- 在线 detector 能连续输出 `fault_prob`

## 7. bridge 第二版增强字段集

第二版用于拉近和 `RflyMAD` 的分布距离。

```text
baro_alt_meter
baro_temp
baro_pressure_pa
mag_x mag_y mag_z
battery_voltage_v
battery_remaining
actuator_ctrl_1..4
rates_sp_roll rates_sp_pitch rates_sp_yaw
thrust_z
accel_vibration_metric
gyro_vibration_metric
esc_rpm_1..4
nav_state
battery_warning
sys_status_health
system_status
pymav_highres_imu_seen
pymav_attitude_target_seen
pymav_actuator_control_seen
```

## 8. 字段来源分工

### 8.1 直接来自 PX4MavCtrlV4 的字段

这些字段可直接从官方环境现成状态变量读取：

- `roll pitch yaw` <- `uavAngEular`
- `rollspeed pitchspeed yawspeed` <- `uavAngRate`
- `vx vy vz` <- `uavVelNED`
- `pos_z` <- `uavPosNED[2]`
- `accel_x accel_y accel_z` <- `trueAccB`
- `motor1..4` <- `trueMotorRPMS[0..3]`
- `time_s` <- `trueTimeStmp` 或 wall clock 差值
- `seq` <- bridge 内部自增

### 8.2 由 bridge 自己补算的字段

这些字段不需要额外消息源，可以在 bridge 内部按现有 `state.py` 逻辑补算：

- `ax ay az`
- `speed_xy_m_s`
- `speed_3d_m_s`
- `motor_cross_mean`
- `motor_cross_std`
- `acc_norm`
- `motor_effort`
- `rate_err_roll/pitch/yaw`（前提是已有 `rates_sp_*`）

### 8.3 必须额外用 pymavlink 补的字段

建议直接沿用项目已有 `pymav_supplement.py` 的消息选择：

- `HIGHRES_IMU`
  - `baro_alt_meter`
  - `baro_temp`
  - `baro_pressure_pa`
  - `mag_x/y/z`
- `BATTERY_STATUS`
  - `battery_warning`
- `SYS_STATUS`
  - `sys_status_health`
- `HEARTBEAT`
  - `system_status`
- `ATTITUDE_TARGET`
  - `rates_sp_roll/pitch/yaw`
  - `thrust_z`
- `ACTUATOR_CONTROL_TARGET`
  - `actuator_ctrl_1..4`
- `VIBRATION`
  - `accel_vibration_metric`
  - `gyro_vibration_metric`
- `ESC_STATUS`
  - `esc_rpm_1..4`

## 9. 为什么这条路线优于“纯 PX4 live”

用户已经明确验证过：

- 纯 PX4 live 字段与官方数据集差异较大
- 很多关键字段语义和分布都偏离 `RflyMAD`

因此继续用纯 PX4 路线做实时检测，风险很高。

而当前方案保留了：

- 官方故障模型
- 官方 case 语义
- 官方故障编码
- 官方 topic 选择偏好

这会让 live 数据更接近 `RflyMAD` 的生成机制，降低训练域与运行域的偏移。

## 10. 推荐实施顺序

### Phase A

固定不动的部分：

- `FaultType`
- `Subsystem`
- `ControlSequence`
- `FaultModel`
- 官方核心 topic 集

### Phase B

实现 `rfly_live_bridge` 第一版：

- 最小字段集
- 持续 UDP 发包
- 接入现有 `receiver`

### Phase C

接入现有 backend 主链：

- 走 `telemetry_manager`
- 不走旧的 raw realtime path 作为主入口

### Phase D

补第二版增强字段：

- `baro_*`
- `mag_*`
- `rates_sp_*`
- `actuator_ctrl_*`
- `thrust_z`
- `battery / status / vibration / esc`

### Phase E

再引入故障 case 做在线告警验证。

## 11. 当前结论

当前阶段的正确工程判断是：

- `RflySim` 继续作为官方仿真环境
- `BuaaAutoTestPlatForm` 继续作为官方故障/场景驱动参考
- 实时检测链路需要项目自行搭建
- 但必须保留官方故障生成语义，否则数据分布会明显偏离 `RflyMAD`

一句话总结：

> 实时链路要自己搭，但不能脱离官方故障模型、case 语义和 topic 选择习惯，否则生成出来的 live 数据与官方数据集差距会过大，现有模型验证效果会显著变差。
