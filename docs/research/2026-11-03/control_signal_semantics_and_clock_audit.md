# Control Signal Semantics and Clock Audit

**Date:** 2026-11-03  
**Scope:** Read-only source and raw-data audit of PX4/RflySim signal semantics and clock relationships in frozen SIL/HIL Train/Val motor/no-fault data. No model training, inference, or changes to source data, labels, models, or experiment results.

## Findings at a glance

| Item | Result | Confidence |
|---|---|---|
| `actuator_controls_0.control[0..3]` | PX4 group-0 control vector: roll, pitch, yaw, throttle. It is a controller-side control-demand vector, not four motor channels. | **Confirmed** by PX4 message definition, exact logged firmware version family, and ULog schema. |
| `actuator_outputs.output[0..3]` | First four entries of the post-mixer/output-driver topic, in output-native units. These are actuator command values, not actuator feedback. | **Confirmed** as output-command semantics. Units and exact motor mapping are configuration-dependent. |
| output 0..3 to motors 1..4 | Cannot be established for both project domains from the available project configuration. The observed HIL firmware reports `SYS_AUTOSTART=1001` and logs a `quad_x.main.mix` boot message; SIL reports a different airframe ID (`3010`). ULog contains no per-output `Function` mapping. | **Partially confirmed** for HIL's quad-X mixer; **not confirmed** as a four-index-to-four-RPM mapping, nor as SIL/HIL-identical. |
| `TrueState_data.motorRPMs[1..4]` | RflySim ground-truth/simulator actuator-state RPM channels, not PX4 controller logs and not physical tachometer/ESC measurements. RflySim docs identify `MotorRPMS` as rotor speed in RPM. | **Confirmed** as simulation ground-truth RPM semantics; exact dataset-generation executable path was unavailable. |
| `rfly_ctrl_lxl` event | Dataset documentation describes it as carrying fault ID/parameters; `123450` is the motor fault ID in the RflyMAD flight-information documentation. It records a command/parameter event, not independent proof of a physical failure transition. | **Confirmed** as a logged injection/configuration marker; physical-effect time is only approximately cross-checked. |
| `fault_state` | Dataset documentation says 0 means no fault and 1 means fault; sampled TrueState/UAVState flags transition near the same-log custom event after clock conversion. The source condition/code that sets the column was not available. | **Partially confirmed** as a dataset label/flag; implementation condition and exact transition semantics are unconfirmed. |
| Common time axis | There is useful empirical synchronization among ULog publication time, `uavTime`, and `trueTime` at approximately 0.1 s event resolution in the fixed 16-case subset. This is stronger than mere existence of each file, but the toolkit's constant start-offset method is still an assumption, not an independent calibration. `TestInfo` remains an unverified elapsed-time metadata clock. | **Partially confirmed**, suitable for coarse event association, not precise latency ordering. |

## 1. Control signal semantics

### PX4 version and message definition

A representative HIL ULog carries firmware hash `a41c356e08c65682e4e5daec58e41a9534fd8b8d`, release `1.12.3`, hardware `PX4_FMU_V5`, `SYS_HITL=1`, and `SYS_AUTOSTART=1001`. A representative SIL ULog has the same firmware hash but `SYS_HITL=0` and `SYS_AUTOSTART=3010`. These values come from the original ULog metadata and initial parameters, not an adapter-generated CSV.

PX4's `actuator_controls` message defines index constants `INDEX_ROLL=0`, `INDEX_PITCH=1`, `INDEX_YAW=2`, `INDEX_THROTTLE=3`; it also contains `timestamp_sample` and an eight-element `control` array. The corresponding message definition is available in the [PX4 v1.13 message docs](https://docs.px4.io/v1.13/en/msg_docs/actuator_controls) and [PX4 v1.13 `actuator_controls.msg`](https://github.com/PX4/PX4-Autopilot/blob/release/1.13/msg/actuator_controls.msg). PX4 v1.12 uses this legacy control-group model; the exact firmware commit is recorded in the sample ULog. Therefore the first four fields are roll, pitch, yaw, and throttle components of group 0. They do not mean motor 1 through motor 4. In a multicopter the mixer combines these control components across several motors.

The project RflyMAD toolkit's [HIL topic-selection JSON](../../../D:/RflyMAD/Data_processing_tools/data_HIL_PX4.json) includes `control[0..3]` from `_actuator_controls_0_0`; the [SIL JSON](../../../D:/RflyMAD/Data_processing_tools/data_SIL_PX4.json) selects the same fields. This confirms which fields the toolkit extracts, not by itself their semantics. Semantics come from the PX4 message definition.

### `actuator_outputs`

PX4's `ActuatorOutputs` message defines `timestamp`, `noutputs`, and `output[16]`, with output data in **natural output units**. See the [PX4 ActuatorOutputs message documentation](https://docs.px4.io/main/en/msg_docs/ActuatorOutputs). In legacy PX4, the output driver/mixer transforms group control demands into output-channel command values. For a PWM output these can be pulse-width-like values; for simulated output drivers PX4 may normalize values. They are not measured actuator positions or motor RPM.

This is directly visible in the sampled HIL ULog: the first entries of `actuator_outputs` are around 1000 before takeoff and rise to roughly 1400-1500 later, while `noutputs=12`. In SIL, the raw topic reports `noutputs=6` in the audited case. The distinct counts and different `SYS_AUTOSTART` values are evidence against assuming that output indexes 0..3 necessarily carry identical mappings in SIL and HIL.

The HIL ULog boot-console text includes `Mixer: /etc/mixers/quad_x.main.mix on /dev/pwm_output0`. The PX4 v1.12 mixer docs explain that a multirotor mixer maps roll/pitch/yaw/thrust to motor outputs and that output order follows mixer/device configuration. The firmware uses an SD-card mixer path, and this ULog alone does not include the contents of that deployed mixer or a complete per-output function map. `SYS_AUTOSTART=3010` in SIL is not the HIL `1001` airframe identifier. Consequently:

- It is confirmed that HIL has a quad-X mixer loaded in its boot log.
- It is **not** confirmed that logged `output[0]` through `[3]` equal RflySim `motorRPMs[1]` through `[4]` in numbering/order.
- It is **not** confirmed that SIL and HIL output indexes map to motors in the same order.
- Output values are controller/driver commands delivered toward the simulated plant or hardware path, not measured motor response.

The RflyMAD configuration JSON includes `actuator_outputs.output[0..3]`, but this is an extraction schema; it does not declare the actuator function assignment. TLog ESC RPM/current is not a substitute: in the preceding 48-case audit the exported RPM/current fields were finite zeros throughout the sample.

## 2. RPM, fault event, and fault-state semantics

### TrueState RPM

RflyMAD's [dataset documentation](https://rfly-openha.github.io/documents/4_resources/dataset.html) says Ground Truth Data is generated by RflySim and contains kinematics, fault states, and motor speeds. RflySim's [ground-truth receiver documentation](https://rflysim.com/doc/en/4.DevRef/slx-tool/rfly-model/udp-30100-truesim-recv.html) defines `MotorRPMS` as actuator status and, for rotorcraft, motor RPM in RPM; its payload also includes a simulation `runnedTime` beginning at zero. The RflyMAD [sample schemas](../../../D:/RflyMAD/Data_processing_tools/data_HIL_GTD.json) select `motorRPMs[1..4]` and exclude `[5..8]` for the four-motor aircraft.

In the fixed 48-case source sample, all four fields are finite in all sampled TrueState workbooks. In a representative HIL case, `trueTime` has 5,880 rows, positive median increment 0.020 s (nominal 50 Hz), and the RPM values are numeric; duplicate timestamps are common because the workbook contains more rows than distinct time steps. These fields are credible as **simulator-reported internal rotor-speed state** for SIL/HIL analysis. They are not physical RPM sensor readings, ESC telemetry, or real-flight measurements. The exact RflySim executable/model code that calculates these values was not present with the dataset, so the audit cannot verify whether the reported state is a modeled shaft speed, a state derived from actuator dynamics, or another simulator output implementation.

This distinction matters: the RflyMAD documentation identifies them as RPM, so they are more than an arbitrary feature column; but their provenance remains simulation ground truth, and the index correspondence to PX4 outputs is not established by the files alone.

### `rfly_ctrl_lxl`, `TestInfo`, and `fault_state`

RflyMAD's [dataset description](https://rfly-openha.github.io/documents/4_resources/dataset.html#16-when-faults-occur-in-rflymad) says the ULog `rfly_ctrl_lxl` topic contains the fault ID and parameters, and that the signal values change to indicate the fault. The [flight-information documentation](https://rfly-openha.github.io/documents/4_resources/flight_information.html) defines control-sequence command class `2`, function `6` as fault injection, and identifies motor fault ID `123450`. Its fault-parameter section says four motor parameters are health factors in [0,1], with 0 representing complete damage and 1 normal. The RflySim motor-fault module documentation describes an efficiency fault as output motor speed equal to input speed times a health factor. This describes the declared simulation mechanism; it does not turn the ULog event marker into a sensor observation of the plant response.

For the sampled HIL Val Hover case `TestCase_2_2000000001`, `TestInfo.csv` contains `Fault ID=[123450.]`, `Fault Parameter=[1. 0.85 0.85 0.85 0.9]`, and a control-sequence command containing `2,6,123450,1,0.85,0.85,0.85,0.9`. Its `Fault injection time` literal is 20. In the raw ULog, the first ID 123450 record has timestamp 175,852,416 us and `controls[0..3]` approximately `[0.85, 0.85, 0.85, 0.9]`. This internally corroborates that the topic records the fault command/parameters. It does not reveal the exact instant at which the plant's RPM/torque response physically changed.

The dataset documentation says `fault_state` in `TrueState_data.xlsx` and `UAVState_data.xlsx` is 0 outside the fault and 1 when a fault occurs. The sampled case's two flags transition at `trueTime=40.262 s` and `uavTime=175806 ms`. The RflyMAD processing package includes no code that generates these workbooks; its `data_HIL_GTD.json` and `data_SIL_GTD.json` only describe which columns to read. Thus the flag's documented role is confirmed, while its exact producer-side trigger condition, delay, and whether it denotes command dispatch versus modeled fault activation remain unverified.

The upstream [Data_processing_tools `fileprocess.py`](https://github.com/lerlis/Data_processing_tools/blob/main/fileprocess.py) function `RFLY_fault_id_check` treats IDs 123450-123459 and 123540-123549 as fault IDs, and 0/1500 as default/no-fault values. `RFLY_info_adjust` repairs values after resampling by carrying neighboring default/fault topic values. That operation is an export post-process, not raw event generation, and should not be used as an independent fault timestamp. The current audit used original ULog values.

## 3. Clock relationship

### What the files actually contain

- **ULog:** PX4 `timestamp` is microseconds since PX4 system start. For the HIL sample, `actuator_outputs` runs from 155.326 s to 212.035 s on that boot-time counter; `rfly_ctrl_lxl` and output topics share the PX4 publication clock. The custom topic is relatively sparse (about 9.5 Hz in the earlier sample audit), so event time is bounded by its update cadence.
- **TrueState:** `trueTime` is seconds in the workbook. It starts at 21.282 s in the HIL sample, has median positive step 0.020 s, and first `fault_state=1` at 40.262 s.
- **UAVState:** `uavTime` is milliseconds in a separate workbook, starts at 156,826 ms in that sample, median positive step 20 ms, and first `fault_state=1` at 175,806 ms.
- **TestInfo:** `Fault injection time` is a scalar elapsed-time value stored with the flight/control-sequence metadata. The file provides no clock-origin field or PX4 boot-time timestamp.

### Processing convention and empirical check

The RflyMAD [Data_processing_tools `Rflytool_main.py`](https://github.com/lerlis/Data_processing_tools/blob/main/Rflytool_main.py) reads the first `uavTime` value, multiplies it by 1000 to get microseconds, reads the first `trueTime` value in seconds, then constructs `TimeBridgeofPX4andGTD(gtd_t=T_start_time, px4_t=start_time)` at lines 141-157. The [bridge in `timetrans.py`](https://github.com/lerlis/Data_processing_tools/blob/main/timetrans.py) maps each source to a shared elapsed offset from those starts. The code then resamples ULog data on a PX4-derived grid and TrueState/UAVState on corresponding grids. This is a documented constant-start-offset convention; it does not read a common event timestamp or estimate drift, delivery latency, or residual alignment error.

However, the current audit adds an independent empirical check of the paired raw files, using only same-case event anchors and no outcome-based offset adjustment. For the fixed 16-case temporal subset (Train/Val × Hover/Velocity × motor/no-fault × SIL/HIL):

- All **8 motor cases** have a same-ULog `rfly_ctrl_lxl` ID 123450-123459 marker and an in-workbook `fault_state` transition.
- Comparing ULog event timestamp relative to ULog start with `uavTime` fault timestamp relative to the UAVState start gives absolute differences from **32 ms to 105 ms** across those eight cases.
- Comparing the same ULog event to `trueTime` elapsed from the TrueState start gives absolute differences from **46 ms to 115 ms**.
- In the same cases, the TrueState and UAVState fault flags occur at elapsed times from their own workbook starts that differ by no more than **23 ms**.
- `TestInfo` injection literals differ from the observed source-clock-relative fault/event positions; the example Val HIL Hover entry says 20 s, while its ULog event is 20.538 s after the ULog start and its UAVState flag is 18.980 s after that workbook's start. Treating the metadata literal as sharing either file's start would create a false exact alignment.

These measurements support a **coarse cross-file event association at about 0.1 s resolution** for this fixed sample. They do not prove an exact affine clock relationship at millisecond precision: event-topic update intervals are about 0.1 s, workbook rows contain repeated timestamps, and no transmitted common clock marker, per-row capture timestamp, or independent synchronization pulse was found. The constant-offset method is therefore not sufficient to substantiate controller-leading-state latency claims finer than its observed residual/event-cadence bound. It is more accurate to say that common simulation progression is empirically supported at coarse event scale than to say that the sources have no relation at all.

No result here establishes that the `TestInfo` scalar time uses the same origin as ULog, `uavTime`, or `trueTime`. It remains metadata for the planned test/control sequence unless an original orchestrator timestamp/clock definition is supplied.

## Conclusion

- **What do the four control values mean?** `control[0..3]` are PX4 group-0 roll, pitch, yaw, and throttle control-demand components. They are not four per-motor commands.
- **What do the four output values mean?** `output[0..3]` are the first four post-mixer/output-driver command values in device-native units. They are not feedback measurements and not thrust. Exact motor assignment is dependent on the loaded mixer/output-function configuration.
- **Are the outputs mapped motor 1-4 identically in SIL and HIL?** This is **not established**. HIL logs `quad_x.main.mix` but lacks a verified saved four-channel function-to-motor map; SIL uses a different airframe ID and has a different output count (6 vs 12 in the sampled logs). The channel mapping must not be asserted from array indices alone.
- **Can four RPM values be used as trusted simulation state?** Yes, with a scope limit: they are documented RflySim simulated ground-truth motor RPM state and all four are finite in the 48 audited cases. They are not physical RPM sensor values, not ULog controller values, and their exact model-generation path and PX4-output channel correspondence are not independently verified.
- **Does a fault marker locate the fault accurately?** `rfly_ctrl_lxl` ID/parameters identify the logged fault-injection command; `fault_state` is documented as the ground-truth active-fault flag. In the fixed 16 cases those markers cross-correlate within roughly 0.1 s. Neither marker independently resolves the physical plant onset to a millisecond or establishes exact `TestInfo` clock origin.
- **Is a reliable shared time base available?** **Partly.** Raw ULog and the two simulation workbook clocks are empirically associated at coarse event scale through paired starts and fault markers; the upstream package implements a start-offset assumption. No independent synchronization pulse or quantified drift calibration was found. `TestInfo` remains unaligned metadata.
- **Is the data sufficient for control-compensation and early-diagnosis research?** It is sufficient to continue **coarse, exploratory simulation-only study** of PX4 control-demand/output commands and simulator RPM/state around event flags. It is not yet sufficient to claim motor-specific output mapping across SIL/HIL, physical actuator feedback, or sub-0.1-second early diagnosis that distinguishes compensation from normal maneuvering. This audit does not validate a diagnostic model or the compensation hypothesis.

## Evidence locations

- Local signal selection: `D:/RflyMAD/Data_processing_tools/data_HIL_PX4.json`, `data_SIL_PX4.json`, `data_HIL_GTD.json`, `data_SIL_GTD.json`.
- Local clock bridge: `D:/RflyMAD/Data_processing_tools/Rflytool_main.py:141`, `:149`, `:157`, `:178`, `:211`, `:213`; `D:/RflyMAD/Data_processing_tools/timetrans.py:52` onward.
- Local event-ID post-processing: `D:/RflyMAD/Data_processing_tools/fileprocess.py:322`, `:351`, `:461`.
- Raw example: `D:/RflyMAD/HIL/hover/motor/TestCase_2_2000000001/Log/log_1_2023-7-24-15-54-16.ulg` (ULog firmware hash/version, topic fields, event marker, HIL and mixer boot metadata).
- Raw companion metadata: `D:/RflyMAD/HIL/hover/motor/TestCase_2_2000000001/TestInfo.csv` and `TrueData/TrueState_data.xlsx`, `TrueData/UAVState_data.xlsx`.
- Fixed 16-case results from prior read-only audit: `artifacts/control_compensation_feasibility_audit_final_v6/clock_alignment_examples.csv` and `artifacts/control_compensation_feasibility_audit_sampled_v1/within_clock_timeline_summary.csv`.
- PX4 definitions: [actuator_controls (PX4 v1.13)](https://docs.px4.io/v1.13/en/msg_docs/actuator_controls), [ActuatorOutputs](https://docs.px4.io/main/en/msg_docs/ActuatorOutputs), [Mixing and Actuators (PX4 v1.12)](https://docs.px4.io/v1.12/en/concept/mixing).
- RflySim definitions: [RflyMAD fault markers](https://rfly-openha.github.io/documents/4_resources/dataset.html), [flight information and command sequence](https://rfly-openha.github.io/documents/4_resources/flight_information.html), [ground-truth `MotorRPMS`](https://rflysim.com/doc/en/4.DevRef/slx-tool/rfly-model/udp-30100-truesim-recv.html), [PX4/RflySim protocol summary](https://rflysim.com/doc/en/4.DevRef/protocol.html).
