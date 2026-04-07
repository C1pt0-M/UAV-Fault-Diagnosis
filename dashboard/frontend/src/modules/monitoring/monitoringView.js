export const monitoringBackendOptions = [
  { label: 'PX4 SITL', value: 'px4' },
  { label: 'Replay CSV', value: 'replay' }
]

export const monitoringDurationOptions = [
  { label: '30 秒', value: 30 },
  { label: '60 秒', value: 60 },
  { label: '120 秒', value: 120 }
]

export const monitoringStateLabels = {
  IDLE: '空闲',
  COLLECTING: '采集中',
  DETECTING: '检测中',
  BLIND_TEST: '盲测中'
}

export const monitoringCopy = {
  heroTitle: '实时监测与在线预警工作台',
  heroDescription: '统一呈现遥测采集、在线检测、故障趋势与会话承接，让运行状态一眼可读。',
  heroStatusLabel: '当前状态',
  heroBackendLabel: '数据源',
  heroModelLabel: '在线模型',
  heroBridgeLabel: '会话承接',
  heroBridgeReady: '支持承接到离线诊断',
  heroBridgeIdle: '等待形成可承接会话',

  controlTitle: '监测控制台',
  controlDescription: '在同一处完成数据源选择、在线模型切换与采集 / 检测控制。',
  modelPlaceholder: '选择在线模型',
  replayAssetPlaceholder: '选择回放资产',
  startCollectionLabel: '启动采集',
  startDetectionLabel: '启动检测',
  startBlindTestLabel: '启动盲测',
  stopDetectionLabel: '停止检测',
  stopCollectionLabel: '停止采集',

  alertsTitle: '告警与样本摘要',
  alertsDescription: '同步展示当前告警事件与关键实时字段，帮助快速定位异常时段。',
  alertsEmpty: '当前没有故障告警',
  sampleTitle: '实时样本摘要',
  sampleDescription: '读取最近一批实时数据并展示关键字段。',

  realtimeCurveTitle: '实时曲线主区',
  realtimeCurveDescription: '把故障概率、动力输出与姿态变化集中到主视图中，优先展示系统正在发生什么。',
  realtimeCurveEmpty: '当前会话暂无可绘制的实时曲线',
  motorChartTitle: '动力输出',
  attitudeChartTitle: '姿态角',
  positionChartTitle: '高度轨迹',

  sessionsTitle: '监测会话承接区',
  sessionsDescription: '跟踪采集历史，并在需要时继续承接到离线诊断中心。',
  sessionsEmpty: '暂无监测会话',
  sessionDetailTitle: '会话详情',
  sessionDetailEmpty: '暂无会话详情',
  sessionIdLabel: '会话 ID',
  droneIdLabel: '无人机',
  statusLabel: '状态',
  faultEventsLabel: '故障事件',
  startTimeLabel: '开始时间',
  durationLabel: '时长（秒）',
  detailAction: '查看详情',
  continueAction: '继续诊断',

  unknownFault: '疑似异常',
  pendingSeverity: '待评估',
  noSession: '暂无会话',
  noSelection: '未选择',
  noValue: '暂无',

  backendNotAllowed: '当前环境未开放该后端，请使用 Replay CSV',
  replayAssetRequired: '请选择回放资产',
  startCollectionSuccess: '遥测采集已启动',
  startDetectionSuccess: '在线检测已启动',
  stopDetectionSuccess: '在线检测已停止',
  startBlindTestSuccess: '盲测已启动',
  stopCollectionFallback: '采集已停止',
  stopCollectionConfirmTitle: '是否跳转故障分类',
  stopCollectionConfirmMessage(count) {
    return `当前会话已确认 ${count} 个故障事件，是否跳转到离线诊断中心继续分析？`
  }
}

export function resolveMonitoringState(runStatus = {}) {
  if (runStatus.is_blind_test) return 'BLIND_TEST'
  if (runStatus.is_detecting) return 'DETECTING'
  if (runStatus.is_collecting) return 'COLLECTING'
  return 'IDLE'
}

export function resolveMonitoringBackendOptions() {
  return monitoringBackendOptions
}

export function shouldShowBlindTest(capabilities = {}) {
  return Boolean(capabilities?.blind_test_enabled)
}

export function formatLatestValue(values, formatter = (value) => value) {
  if (!Array.isArray(values) || values.length === 0) {
    return monitoringCopy.noValue
  }
  return formatter(values[values.length - 1])
}
