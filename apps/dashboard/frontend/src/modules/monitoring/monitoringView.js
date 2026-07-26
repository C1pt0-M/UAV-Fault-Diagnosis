export const monitoringBackendOptions = [
  { label: 'RflySim 实时链路', value: 'rfly' },
  { label: 'CSV 回放', value: 'replay' }
]

export const monitoringDurationOptions = [
  { label: '30 秒', value: 30 },
  { label: '60 秒', value: 60 },
  { label: '120 秒', value: 120 }
]

export const rflyInferenceLocationOptions = [
  { label: 'MEC 独立进程推理', value: 'mec' },
  { label: '中心平台推理', value: 'center' }
]

export const monitoringStateLabels = {
  IDLE: '空闲',
  COLLECTING: '采集中',
  DETECTING: '检测中',
  BLIND_TEST: '盲测中'
}

export const monitoringCopy = {
  pageTitle: '实时监测',
  pageDescription: '统一管理遥测采集、在线检测、实时告警与会话承接。',
  heroTitle: '实时监测',
  heroDescription: '遥测采集、在线检测、实时告警与会话承接',
  heroStatusLabel: '当前状态',
  heroBackendLabel: '数据源',
  heroModelLabel: '在线模型',

  controlTitle: '监测控制台',
  controlDescription: '在同一处完成数据源选择、在线模型切换与采集 / 检测控制。',
  modelPlaceholder: '选择在线检测模型',
  replayAssetPlaceholder: '选择回放资产',
  rflyPx4MavCtrlPlaceholder: 'PX4MavCtrlV4.py 路径（可选，默认读取后端配置）',
  rflyInferenceLocationLabel: '推理位置',
  startCollectionLabel: '启动采集',
  startDetectionLabel: '启动检测',
  startBlindTestLabel: '启动盲测',
  stopDetectionLabel: '停止检测',
  stopCollectionLabel: '停止采集',

  rflyDiagnosticsTitle: 'RflySim 配置诊断',
  rflyDiagnosticsDesc: '检查后端开放状态、PX4MavCtrlV4.py、bridge script、Python 与端口配置。',
  rflyDiagnosticsAction: '检测配置',
  rflyDiagnosticsReady: 'RflySim 链路配置可用',
  rflyDiagnosticsEmpty: '请选择 RflySim 后点击检测配置。',

  alertsTitle: '最近告警事件',
  alertsDescription: '来自 telemetry fault events 的实时告警。',
  alertsEmpty: '当前没有故障告警',
  sampleTitle: '实时样本摘要',
  sampleDescription: '最近一批实时遥测的关键字段。',

  realtimeCurveTitle: '故障概率趋势',
  realtimeCurveDescription: '主图展示在线检测风险变化；下方保留当前接口已有的关键遥测曲线。',
  realtimeCurveEmpty: '暂无实时曲线；启动采集或选择 CSV 回放后自动更新',
  motorChartTitle: '动力输出',
  attitudeChartTitle: '姿态角',
  positionChartTitle: '高度轨迹',

  sessionsTitle: '监测会话记录',
  sessionsDescription: '跟踪采集历史，并在需要时承接到离线诊断中心。',
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
  refreshAction: '刷新',

  unknownFault: '疑似异常',
  pendingSeverity: '待评估',
  noSession: '暂无会话',
  noSelection: '未选择',
  noValue: '暂无',

  backendNotAllowed: '当前环境未开放该后端，请使用 CSV 回放',
  rflyNotReady: 'RflySim 配置未就绪，请先完成配置诊断',
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

export function resolveMonitoringBackendOptions(capabilities = {}) {
  return monitoringBackendOptions
}

export function shouldShowBlindTest(capabilities = {}) {
  return Boolean(capabilities?.blind_test_enabled)
}

export function formatMonitoringModelName(value) {
  const raw = String(value || '').trim()
  if (!raw) return monitoringCopy.noSelection

  const normalized = raw
    .replace(/在线检测模型/g, '')
    .replace(/^online_detector_/i, '')
    .replace(/_rfly$/i, '')
    .replace(/[_-]+/g, ' ')
    .trim()

  if (/kg\s*mstca/i.test(normalized)) return 'KG-MSTCA'
  if (/cnn\s*1d/i.test(normalized)) return 'CNN1D'
  if (/mstca/i.test(normalized)) return 'MSTCA'

  return raw
}

export function formatLatestValue(values, formatter = (value) => value) {
  if (!Array.isArray(values) || values.length === 0) {
    return monitoringCopy.noValue
  }
  return formatter(values[values.length - 1])
}
