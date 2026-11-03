# RflyMAD Control-Compensation Feasibility Audit

**Date:** 2026-11-03  
**Scope:** Read-only feasibility audit of frozen SIL/HIL Train/Val motor-fault and no-fault source cases. No training, inference, Test/Real data, model changes, or threshold/feature selection.

## Executive conclusion

RflyMAD is **partially suitable for exploratory analysis** of controller-side and actuator-output changes around simulated motor-fault events. It is **not sufficient, on the evidence audited here, to establish closed-loop control compensation as an early, fault-specific diagnostic signal**.

The fixed source sample contains `actuator_controls_0`, `actuator_outputs`, Rfly custom control/event records, and four finite `TrueState_data.xlsx` RPM channels. The RPM channels are simulation TrueState fields, not verified ESC or physical tachometer measurements. HIL TLog ESC RPM and current fields are present but all finite values in the sampled export are zero. The record rates and channel layouts differ substantially between SIL and HIL.

Most importantly, the RflyMAD processing toolkit aligns the sources by anchoring the beginning of TrueState to the beginning of the PX4 log and applying a constant offset. That is a documented processing convention, not an independently verified synchronization marker. `TestInfo` fault-injection seconds, ULog publication timestamps, Excel `trueTime`, and `uavTime` therefore remain distinct time sources for causal-order claims. The audit cannot verify that a controller signal precedes an external motion-state anomaly.

## Questions answered

| Question | Assessment | Evidence-bounded answer |
|---|---|---|
| Is the dataset sufficient to study closed-loop compensation? | **Partially verified** | It contains useful simulated controller/output and TrueState RPM streams in sampled Train/Val SIL/HIL cases. Signal semantics, motor mapping, physical RPM provenance, and a validated common clock are missing, preventing a mechanism-level conclusion. |
| Do internal control signals precede external motion-state changes? | **Not established** | The data-processing toolkit uses a start-offset bridge, but the audit found no independent synchronization event that validates that offset at sample level or ties it to `TestInfo` injection time. Do not infer lead time from these files. |
| Can normal maneuvers be distinguished from fault compensation? | **Not established** | The fixed sample is descriptive, not a powered or background-matched discrimination study. A sampled normal Velocity case shows a strong single-channel RPM/output change, demonstrating that asymmetry/change alone is not fault-specific. |
| What is worth pursuing next? | **Partially supported** | A narrowly scoped SIL/HIL study of raw controller/output response around the same-ULog `rfly_ctrl_lxl` event is feasible as an exploratory signal audit. It requires verified actuator mapping, event semantics, synchronization, maneuver-matched normal cases, and independent case-level replication before testing early-diagnostic claims. |

## Data and sampling

The frozen Train/Val ledger contains **1,967 motor/no-fault source cases**: Train 1,283 motor and 334 no-fault; Val 278 motor and 72 no-fault. These are ledger case counts, not independent physical flights. The ledger reports the same counts of association groups for these strata; SIL/HIL association-group matches are paired records and are not counted as independent repeated flights.

The raw audit used **48 fixed cases**: split × six flight folders (`hover`, `velocity`, `acce`, `dece`, `waypoint`, `circling`) × fault class (`motor`, `no_fault`) × domain (SIL/HIL). Selection was deterministic from the frozen ledger, using a SIL/HIL association-group pair where available. It was not randomized and was not selected by looking at outcomes. It gives one pair per cell, not a representative estimate of full-corpus raw-signal availability. The 16-case temporal subset covers Train/Val × Hover/Velocity × fault class × domain.

All selected cases had a parseable ULog, TrueState workbook, UAVState workbook, and TLog CSV. The source streams are not uniformly useful simply because a file or field exists.

## Signal audit

- **TrueState RPM:** `motorRPMs[1]` through `[4]` are present and finite in all 48 selected workbooks. In the fixed 16-case temporal subset, `trueTime` advances by 0.020 s at its median positive step (nominal 50 Hz); repeated timestamps also occur. Across the four channels in all 48 cases, the mean zero fraction by case is about 0.85%, ranging from about 0.34% to 1.68%. These fields are from simulated TrueState, not verified motor-RPM sensor telemetry. Their channel-to-motor mapping and exact generation semantics remain unknown.
- **UAVState:** the separate `uavTime` clock and state workbook are present. In checked cases it advances in milliseconds with a median positive step of 20 ms, while repeated timestamps occur. It must not be silently merged with `trueTime`; the two are separate workbook timelines.
- **ULog controller fields:** `actuator_controls_0.control[0..3]` are present and finite in all 48 sampled ULogs. Median record rate is about 250 Hz for HIL and 20 Hz for SIL. These remain PX4 topic values; this audit did not prove that each channel is a motor command or identify its units/axis semantics.
- **ULog output fields:** `actuator_outputs.output[0..3]` are present and finite in all 48 sampled ULogs. Median record rate is about 250 Hz for HIL and 10 Hz for SIL. The median `noutputs` is 12 in HIL and 6 in SIL. Channel order, motor mapping, physical unit, and whether values represent realized actuator motion are not established. They are not treated as thrust.
- **Custom event topic:** `rfly_ctrl_lxl` exists in all sampled ULogs at about 9.5 Hz median. Motor-fault samples commonly contain IDs in the `123450` range. This is evidence of a logged custom event/control parameter, not proof of physical fault onset or closed-loop compensation.
- **TLog ESC telemetry:** ESC RPM and current columns are finite in sampled TLog records, but all audited RPM/current values are zero: 10,204 finite channel-cells per group, with zero nonzero cells. A finite zero is not missing, but it is not usable evidence of measured RPM/current response. The TLog is low-rate (roughly 1 Hz in checked records), unsuitable for millisecond-scale early ordering.
- **Adapter/merged values excluded:** adapter defaults, merger-filled zeros, and RPM synthesized from actuator outputs are not treated as original measurements.

## Timing and event evidence

The upstream `Data_processing_tools` implementation constructs a `TimeBridgeofPX4andGTD` from the first TrueState time and first PX4 time, then maps the streams through a constant start offset. This documents the intended resampling convention. It does not independently validate the offset or account for transport, startup, buffering, or event latency.

In a representative HIL validation hover motor case, `TestInfo` states 20 s, the same-ULog `rfly_ctrl_lxl` motor-event marker occurs about 20.538 s after the ULog start, and the TrueState `fault_state` first becomes 1 at `trueTime` about 40.262 s (workbook begins around 21.282 s). These numbers are reported only in their source clocks. Their differences must not be interpreted as latency because the time origins have not been verified as common. Similar independent-clock discrepancies occur in other cases.

Within ULog, `actuator_controls_0`, `actuator_outputs`, and `rfly_ctrl_lxl` can be compared on publication timestamps. Within TrueState, RPM and kinematics can be compared on `trueTime` using `fault_state` as that workbook's phase marker. A cross-source precedence claim needs an independently validated clock relation.

## Descriptive temporal observations

In sampled HIL/SIL Hover motor examples, the four TrueState RPM channels all shift downward by roughly 120 RPM between the broad pre- and post-`fault_state` segments. The shift is shared across all four channels and does not identify a failed rotor or prove compensatory redistribution. The same examples show ULog `actuator_outputs` rising broadly after the custom event marker; that change is compatible with multiple causes, including a change in operating point or control sequence, and cannot be interpreted as thrust or controller intent from these fields alone.

In sampled HIL Velocity motor examples, two TrueState RPM channels settle near 5,220 RPM and two near 5,375 RPM after the TrueState phase change. The corresponding ULog output channels also become asymmetric. However, a sampled normal Velocity case shows a marked single-channel RPM and output shift without a motor-fault label. Normal maneuver context is therefore a material confound, and these small case examples do not establish separation or specificity.

Pre/post summaries use long segments divided by source-local phase markers; they are descriptive and can include other operating changes. The number of time samples is not treated as the number of independent flights. No significance tests or learned decision rules were applied.

## Severity and coverage limits

`TestInfo` contains literal `Fault Parameter`, `Control Sequence`, and injection-time metadata. Each sampled motor stratum has one literal parameter pattern by construction, but parameter elements are not decoded into physical severity. The sample does not support severity-response curves or matched severity comparisons. The sampled cases span all six folder-level maneuver labels, but this is not evidence of matched trajectory, state, or controller background between normal and fault cases.

A prior attempt to read every raw ULog exceeded the local time budget. The audit therefore does not estimate full-corpus raw-signal validity rates. The 48-case raw signal results must not be extrapolated to all 1,967 ledger cases.

## Overall feasibility and recommended next step

**Feasible now:** exploratory, source-local SIL/HIL comparisons of logged controls/outputs around a same-ULog custom event and TrueState RPM/kinematics around the workbook fault-state flag, with descriptive case-level reporting.

**Not supported now:** a claim that the control signal leads motion-state degradation; a claim that it represents purposeful controller compensation; generalization to real flights; physical motor identification from channel indices; or fault-specific early diagnosis separated from normal maneuvering.

The next worthwhile work is a bounded data-semantics and synchronization study, not model development: verify PX4 allocator/output mapping for each SIL/HIL configuration; establish the meaning of the Rfly custom event and TrueState RPM generation; validate a common timestamp relation using a shared event or original simulator logs; then build maneuver-matched case pairs across multiple association groups and compare full case trajectories. Until these are resolved, the research claim should remain at exploratory simulated signal characterization.

## Reproducibility and deliverables

- Fixed-sample raw evidence and source-clock summaries: `artifacts/control_compensation_feasibility_audit_sampled_v1/`.
- Compact derived tables and figures: `artifacts/control_compensation_feasibility_audit_final_v6/`.
- Read-only scripts: `scripts/research/audit_control_compensation_feasibility.py`, `scripts/research/analyze_control_compensation_timelines.py`, and `scripts/research/summarize_control_compensation_audit.py`.
- The full raw-source run is intentionally not repeated. No original ULog, Excel workbook, cached data, model weight, or prediction file is part of the proposed publication set.
