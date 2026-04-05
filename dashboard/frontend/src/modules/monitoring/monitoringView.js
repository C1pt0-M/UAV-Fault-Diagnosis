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
  controlTitle: '监测控制台',
  controlDescription: '统一执行采集、在线检测、盲测与会话承接。',
  modelPlaceholder: '选择在线模型',
  replayAssetPlaceholder: '选择回放资产',
  alertsTitle: '实时告警',
  alertsDescription: '展示当前运行状态下的故障事件与提示信息。',
  alertsEmpty: '当前没有故障告警',
  sampleTitle: '实时样本摘要',
  sampleDescription: '读取最近一批实时数据并展示关键字段。',
  realtimeCurveTitle: '实时字段曲线',
  realtimeCurveDescription: '按采集时间连续展示关键飞行字段，便于观察动态波动与异常拐点。',
  realtimeCurveFieldPlaceholder: '选择要展示的实时字段',
  realtimeCurveEmpty: '当前会话暂无可绘制的实时曲线',
  sessionsTitle: '监测会话',
  sessionsDescription: '跟踪采集历史，必要时可直接承接到离线诊断中心。',
  sessionsEmpty: '暂无监测会话',
  sessionDetailTitle: '会话详情',
  sessionDetailEmpty: '暂无会话详情',
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
    return `当前会话已确认 ${count} 个故障事件，是否跳转故障分类？`
  }
}

export function resolveMonitoringState(runStatus = {}) {
  if (runStatus.is_blind_test) return 'BLIND_TEST'
  if (runStatus.is_detecting) return 'DETECTING'
  if (runStatus.is_collecting) return 'COLLECTING'
  return 'IDLE'
}

export function resolveMonitoringBackendOptions(capabilities = {}) {
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
