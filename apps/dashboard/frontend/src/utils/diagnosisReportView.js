const LABEL_META = {
  normal: {
    text: '状态正常',
    summary: '当前样本未识别出明显故障特征，飞行状态整体稳定。',
    abnormal: false
  },
  motor: {
    text: '电机故障',
    summary: '动力系统输出或转速一致性出现异常，可能影响姿态稳定和推力响应。',
    abnormal: true
  },
  propeller: {
    text: '螺旋桨故障',
    summary: '螺旋桨平衡性或气动状态存在异常，可能带来持续振动和效率下降。',
    abnormal: true
  },
  sensor: {
    text: '传感器故障',
    summary: 'IMU 或相关传感器数据一致性异常，可能影响姿态估计与控制精度。',
    abnormal: true
  },
  unknown: {
    text: '待确认异常',
    summary: '当前报告未能稳定归类，建议结合原始数据与任务背景进行人工复核。',
    abnormal: true
  }
}

const SEVERITY_META = {
  none: {
    badge: '正常',
    text: '风险可控'
  },
  light: {
    badge: '轻度',
    text: '轻度风险'
  },
  medium: {
    badge: '中度',
    text: '中度风险'
  },
  heavy: {
    badge: '高风险',
    text: '高风险'
  },
  unknown: {
    badge: '待确认',
    text: '待人工确认'
  }
}

const PRIORITY_META = {
  P0: '立即处理',
  P1: '优先处理',
  P2: '安排检查',
  P3: '持续观察'
}

const DATA_QUALITY_META = {
  good: { text: '良好', tone: 'ok' },
  warning: { text: '需关注', tone: 'warn' },
  poor: { text: '较差', tone: 'danger' },
  unknown: { text: '未知', tone: 'muted' }
}

const REPORT_CONFIDENCE_META = {
  high: { text: '高', tone: 'ok' },
  medium: { text: '中', tone: 'warn' },
  low: { text: '低', tone: 'danger' },
  unknown: { text: '未知', tone: 'muted' }
}

const REVIEW_STATUS_META = {
  pending: { text: '待复核', tone: 'warn' },
  confirmed: { text: '已确认', tone: 'ok' },
  false_alarm: { text: '误报', tone: 'muted' },
  ignored: { text: '已忽略', tone: 'muted' }
}

const METRIC_LABELS = {
  motor_imbalance_ratio: '电机输出均衡性',
  esc_rpm_imbalance_ratio: 'ESC 转速一致性',
  battery_remaining_min: '最低剩余电量',
  prediction_confidence: '模型判断置信度'
}

const STATUS_LABELS = {
  used: '已采用',
  missing: '缺失',
  unknown: '未知'
}

export const faultLabelOptions = [
  { label: '状态正常', value: 'normal' },
  { label: '电机故障', value: 'motor' },
  { label: '传感器故障', value: 'sensor' }
]

function toNumber(value) {
  const numeric = Number(value)
  return Number.isFinite(numeric) ? numeric : null
}

function getLabelMeta(label) {
  return LABEL_META[String(label || '').trim()] || LABEL_META.unknown
}

function getSeverityMeta(level) {
  return SEVERITY_META[String(level || '').trim()] || SEVERITY_META.unknown
}

function getPriorityText(priority) {
  return PRIORITY_META[String(priority || 'P2').trim().toUpperCase()] || PRIORITY_META.P2
}

function formatPercent(value, digits = 0) {
  const numeric = toNumber(value)
  if (numeric === null) return '-'
  return `${(numeric * 100).toFixed(digits)}%`
}

function formatMetricValue(value) {
  const numeric = toNumber(value)
  if (numeric === null) {
    return value === 0 ? '0' : '-'
  }
  if (Math.abs(numeric) >= 100) return numeric.toFixed(1)
  if (Math.abs(numeric) >= 1) return numeric.toFixed(3)
  return numeric.toFixed(4)
}

function normalizeRecommendations(report, abnormal) {
  const raw = Array.isArray(report?.summary?.recommendations) ? report.summary.recommendations : []

  const normalized = raw
    .map((item) => {
      if (typeof item === 'string') {
        return {
          title: item,
          priorityText: getPriorityText('P2')
        }
      }

      if (!item || !item.title) return null

      return {
        title: item.title,
        priorityText: getPriorityText(item.priority)
      }
    })
    .filter(Boolean)

  if (normalized.length > 0) return normalized

  if (!abnormal) {
    return [{
      title: '继续执行例行检查并关注飞行前后巡检记录。',
      priorityText: getPriorityText('P2')
    }]
  }

  return [{
    title: '建议尽快结合原始遥测和维护记录进行人工复核。',
    priorityText: getPriorityText('P1')
  }]
}

function buildEvidenceChain(report) {
  const evidence = Array.isArray(report?.summary?.severity?.evidence) ? report.summary.severity.evidence : []
  return evidence.slice(0, 5).map((item) => ({
    metric: item.metric,
    metricText: METRIC_LABELS[item.metric] || String(item.metric || '关键指标'),
    value: formatMetricValue(item.value),
    status: item.status,
    statusText: STATUS_LABELS[String(item.status || '').trim()] || String(item.status || '未知'),
    score: item.score != null ? formatMetricValue(item.score) : '-'
  }))
}

function buildWhyItems(report, confidenceText, evidenceChain) {
  const whyItems = evidenceChain.slice(0, 3).map((item) => ({
    title: item.metricText,
    description: `${item.metricText} 出现异常，当前观测值为 ${item.value}，规则贡献分 ${item.score}。`
  }))

  if (whyItems.length === 0) {
    whyItems.push({
      title: '模型判断',
      description: `当前报告的主结论由离线模型给出，主分类置信度为 ${confidenceText}。`
    })
  } else {
    whyItems.push({
      title: '模型判断',
      description: `当前报告的主结论由离线模型给出，主分类置信度为 ${confidenceText}。`
    })
  }

  return whyItems
}

function buildProbabilityRows(report) {
  const rawTopk = Array.isArray(report?.summary?.topk) ? report.summary.topk : []
  const rows = rawTopk.length
    ? rawTopk
    : Object.entries(report?.probs || {}).map(([label, prob]) => ({ label, prob }))

  return rows.map((item) => ({
    label: item.label,
    labelText: getLabelMeta(item.label).text,
    prob: Number(item.prob || 0),
    probText: formatPercent(item.prob, 0)
  }))
}

function buildDataQualityView(report) {
  const raw = report?.summary?.data_quality
  const dataQuality = raw && typeof raw === 'object' ? raw : {}
  const level = String(dataQuality.level || 'unknown').trim() || 'unknown'
  const meta = DATA_QUALITY_META[level] || DATA_QUALITY_META.unknown

  const metrics = [
    { label: '样本数量', value: dataQuality.sample_count ? String(dataQuality.sample_count) : '待补充' },
    { label: '窗口数量', value: dataQuality.window_count ? String(dataQuality.window_count) : '待补充' },
    { label: '特征数量', value: dataQuality.feature_count ? String(dataQuality.feature_count) : '待补充' },
    {
      label: '缺失值占比',
      value: dataQuality.missing_value_ratio != null ? formatPercent(dataQuality.missing_value_ratio, 1) : '待补充'
    },
    {
      label: '异常值占比',
      value: dataQuality.invalid_value_ratio != null ? formatPercent(dataQuality.invalid_value_ratio, 1) : '待补充'
    }
  ]

  const timeRange = dataQuality.time_range && typeof dataQuality.time_range === 'object'
    ? dataQuality.time_range
    : {}
  if (timeRange.start || timeRange.end) {
    metrics.push({
      label: '时间范围',
      value: `${timeRange.start || '待补充'} ~ ${timeRange.end || '待补充'}`
    })
  }

  const warnings = Array.isArray(dataQuality.missing_columns) && dataQuality.missing_columns.length
    ? [`存在缺失列：${dataQuality.missing_columns.slice(0, 5).join('、')}`]
    : []
  if (Array.isArray(dataQuality.warnings)) {
    warnings.push(...dataQuality.warnings)
  }
  if (!warnings.length) {
    warnings.push('无额外数据质量告警。')
  }

  return {
    level,
    levelText: meta.text,
    tone: meta.tone,
    metrics,
    warnings
  }
}

function buildTraceabilityView(report) {
  const raw = report?.summary?.traceability
  const traceability = raw && typeof raw === 'object' ? raw : {}
  const modelCard = report?.summary?.model_card && typeof report.summary.model_card === 'object'
    ? report.summary.model_card
    : {}

  const classSet = Array.isArray(modelCard.class_set) && modelCard.class_set.length
    ? modelCard.class_set
    : ['normal', 'motor', 'sensor']

  return {
    runId: traceability.run_id ?? report?.run_id ?? '-',
    assetId: traceability.asset_id ?? report?.asset_id ?? '-',
    assetFilename: traceability.asset_filename || report?.summary?.basic?.asset_filename || '-',
    inputCsv: traceability.input_csv || '待补充',
    modelName: traceability.model_name || report?.model_name || '-',
    modelVersion: traceability.model_version || '待补充',
    modelHash: traceability.model_hash || '待补充',
    rulesVersion: traceability.rules_version || 'rule_v1',
    generatedAt: traceability.generated_at || report?.created_at || '-',
    task: modelCard.task || 'offline_multiclass',
    classSet,
    classSetText: classSet.join(' / '),
    intendedUse: modelCard.intended_use || 'RflySim/RflyMAD 离线日志故障诊断辅助分析',
    limitations: Array.isArray(modelCard.limitations) ? modelCard.limitations : []
  }
}

function buildReportConfidenceView(report) {
  const raw = report?.summary?.report_confidence
  const reportConfidence = raw && typeof raw === 'object' ? raw : {}
  const level = String(reportConfidence.level || 'unknown').trim() || 'unknown'
  const meta = REPORT_CONFIDENCE_META[level] || REPORT_CONFIDENCE_META.unknown
  const reasons = Array.isArray(reportConfidence.reasons) && reportConfidence.reasons.length
    ? reportConfidence.reasons
    : ['报告可信度信息待补充，建议结合原始数据人工复核。']

  return {
    level,
    levelText: meta.text,
    tone: meta.tone,
    score: reportConfidence.score != null ? formatMetricValue(reportConfidence.score) : '-',
    reasons
  }
}

function buildReviewView(report) {
  const raw = report?.summary?.review
  const review = raw && typeof raw === 'object' ? raw : {}
  const status = String(review.status || 'pending').trim() || 'pending'
  const meta = REVIEW_STATUS_META[status] || REVIEW_STATUS_META.pending

  return {
    status,
    statusText: meta.text,
    tone: meta.tone,
    reviewer: review.reviewer || '待补充',
    reviewedAt: review.reviewed_at || '待补充',
    conclusion: review.conclusion || '待补充',
    maintenanceNotes: review.maintenance_notes || '待补充'
  }
}

export function buildDiagnosisReportView(report = {}) {
  const labelMeta = getLabelMeta(report.prediction_label)
  const severityMeta = getSeverityMeta(report?.summary?.severity?.level)
  const confidenceText = formatPercent(report.confidence, 0)
  const actionItems = normalizeRecommendations(report, labelMeta.abnormal)
  const probabilityRows = buildProbabilityRows(report)
  const dataQuality = buildDataQualityView(report)
  const traceability = buildTraceabilityView(report)
  const reportConfidence = buildReportConfidenceView(report)
  const review = buildReviewView(report)
  const evidenceChain = buildEvidenceChain(report)

  const headline = labelMeta.abnormal
    ? `本次离线诊断识别为${labelMeta.text}，当前属于${severityMeta.badge}。`
    : '本次离线诊断未发现明显故障，当前状态正常。'

  const riskText = labelMeta.abnormal
    ? `当前结果显示${severityMeta.text}，建议立即复核关键部件与飞行记录。`
    : `当前结果显示${severityMeta.text}，模型对正常状态的判断置信度为 ${confidenceText}。`

  return {
    labelText: labelMeta.text,
    headline,
    riskText,
    actionSummary: actionItems.map((item) => item.title).join('；'),
    whyItems: buildWhyItems(report, confidenceText, evidenceChain),
    actionItems,
    probabilityRows,
    dataQuality,
    traceability,
    reportConfidence,
    review,
    evidenceChain,
    severityBadge: severityMeta.badge,
    severityText: severityMeta.text,
    confidenceText
  }
}
