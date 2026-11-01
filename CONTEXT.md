# 基于数据驱动的无人机故障智能诊断系统

This context defines the product language for the UAV fault diagnosis system. It keeps the current V1.0 software boundary clear for documentation, copyright registration, and future development.

## Language

**Data-driven UAV intelligent fault diagnosis system**:
The V1.0 software boundary centered on UAV operating data, flight log assets, CSV replay, offline fault diagnosis, diagnosis reports, report history, and AI-assisted interpretation.
_Avoid_: RflySim closed-loop diagnosis system, PX4 real-time diagnosis system, report analysis system

**Diagnostic value chain**:
The product value of turning raw UAV operating data into observable monitoring results, risk alerts, offline diagnosis conclusions, reviewable reports, and assisted interpretation.
_Avoid_: Single model prediction, raw log viewer

**Flight log asset**:
A diagnostic input prepared from UAV flight data and used by the system for replay, analysis, and report generation.
_Avoid_: Simulation session, live aircraft connection

**CSV replay**:
A repeatable way to feed prepared flight log data through monitoring or analysis flows without depending on a live simulator or aircraft link.
_Avoid_: Real-time simulation, closed-loop flight test

**Online monitoring session**:
A time-ordered monitoring run that evaluates a replayed telemetry-like data stream, summarizes risk over sliding windows, helps locate suspicious fault intervals, and can be handed off for offline diagnosis. Its core session owns current risk, warning state, and preview timing; Dashboard reads a snapshot rather than maintaining copies of those values. The Dashboard telemetry manager belongs to one Flask application, is looked up through that application, and is closed explicitly rather than at request teardown.
_Avoid_: Final fault classification, RflySim flight, simulator mission

**Online detection policy**:
The explicit inference interval and event aggregation settings used by an online monitoring session. Runtime entries share the implementation but retain their declared defaults; transport queues, arming gates, input-quality assessment cadence, and experimental data roles are separate concerns.
_Avoid_: Training hyperparameters, a replacement for the frozen experiment protocol, one mandatory default for every transport

**Event timing record**:
Supplemental metadata that distinguishes a candidate event's original interval, hysteresis confirmation, and finalized output. Stream-clock times and emitting-host Unix times are separate; neither establishes physical damage onset or browser receipt. Existing event intervals and frozen evaluation metrics are unchanged.
_Avoid_: Physical detection latency, end-to-end notification latency, treating the candidate start as the actual alert time

**Offline diagnosis**:
A follow-up analysis flow that classifies a selected flight log asset or monitoring session output into the current supported fault categories and prepares evidence for reporting.
_Avoid_: Real-time control, simulator validation

**Sensor fault**:
A supported offline diagnosis category covering sensor-related abnormalities such as GPS, accelerometer, barometer, or related sensing data issues.
_Avoid_: All possible avionics failures, exact component-level root cause

**Monitoring-to-diagnosis handoff**:
The product relationship where an online monitoring session that indicates possible risk becomes the input context for offline diagnosis and report generation.
_Avoid_: Two separate demos, manual file-only workflow

**Diagnosis report**:
A reviewable record generated after offline diagnosis that captures the input asset, diagnosis result, confidence, risk level, evidence summary, and recommended follow-up actions.
_Avoid_: Raw model output, automatic maintenance order

**AI-assisted interpretation**:
An auxiliary explanation capability that helps users understand an existing diagnosis report, its evidence, risk level, and recommendations without replacing human review or maintenance decisions.
_Avoid_: Automatic root-cause authority, autonomous repair decision

**System initialization**:
The first-use step that creates the administrator account before protected runtime modules can be accessed.
_Avoid_: Public registration, demo default account, simulator setup

**Administrator account**:
The controlled account used to sign in, review protected system functions, and represent the responsible system operator in V1.0.
_Avoid_: Open self-service user, model trainer identity, public visitor

**Authenticated session**:
The signed-in browser session that proves the current user can access protected runtime function modules until the session expires or the user logs out.
_Avoid_: Permanent device binding, anonymous dashboard access

**Controlled account access**:
The V1.0 access model where users enter the system through initialization-created administrator credentials instead of open public registration.
_Avoid_: Open signup portal, multi-tenant user management

**Runtime function module**:
A user-facing capability of the V1.0 system, such as asset management, online monitoring, offline diagnosis, report management, or AI-assisted interpretation.
_Avoid_: Training experiment, research script

**Runtime deployment capability**:
Whether this Flask application has registered an optional runtime function module. The authenticated profile exposes only AI and telemetry availability for navigation; the browser keeps these declarations in memory and does not enable optional pages when declarations are missing or malformed. Availability does not grant account permissions or certify provider/simulator readiness: endpoint authentication, administrator checks and object access controls remain authoritative. Logout, token replacement and current-session expiry clear the browser declaration. Business shortcuts follow the same availability without removing historical reports or warning statistics. Telemetry source choices are filtered by the server allowlist, separately from readiness checks; the current blind-test API uses PX4 and requires both its switch and PX4 allowance. Existing hidden PX4/MEC selectors are not reopened merely because their backend is available.
_Avoid_: User permissions, provider readiness, a browser-only security control, an experimental model capability

**Model input contract**:
The task, ordered feature names, normalization coefficients, label order, and declared schema/protocol metadata needed to interpret a model input correctly. Model discovery and loading share validation without importing the ML runtime into discovery. Legacy online assets may omit task, labels, schema and window declarations, but supplied declarations must agree. Offline diagnosis keeps its separate 31-channel whole-log preprocessor, normalization floor and missing-file fallback; an existing invalid statistics file must not silently select different statistics. Legacy offline statistics without column names remain positional. Offline discovery and prediction resolve labels identically, keeping defaults, whitespace trimming and legacy extra-label truncation while rejecting incomplete or duplicate labels. Supplied feature counts and known schemas must match the input order, and schema/protocol declarations must agree with the selected statistics; missing declarations stay missing and custom schemas are not semantically certified. The version-1 runtime manifest restricts model directories to MODEL_ROOT and checks identity, required files and declared runtime task. These checks do not certify checkpoint contents or experimental validity.
_Avoid_: A training protocol, checkpoint verification, proof of real-time performance

**Model design and training support**:
The technical foundation that prepares, trains, evaluates, and manages models used by runtime diagnosis and monitoring functions; it is documented as algorithm support rather than a user-facing runtime module.
_Avoid_: Dashboard training module, end-user training workflow

**Professional analysis user**:
A user who understands UAV operation, flight testing, maintenance, or fault diagnosis research and uses the system to review logs, model outputs, evidence, and reports.
_Avoid_: General consumer, aircraft controller

**Industry scenario user**:
A lower-specialization user from agricultural operations, road inspection, park inspection, or low-altitude economy scenarios who uses monitoring results, alerts, reports, and AI-assisted interpretation to understand UAV risk.
_Avoid_: Model trainer, simulator operator

**RflySim/PX4 extension interface**:
An experimental or future integration boundary for simulator or PX4 telemetry sources; it is excluded from V1.0 copyright documentation, screenshots, and source-code excerpt materials.
_Avoid_: Core diagnosis capability, required runtime dependency, V1.0 documented feature
