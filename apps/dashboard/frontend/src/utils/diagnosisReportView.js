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
  good: { badge: '良好', text: '数据质量良好，可支撑当前结论' },
  warning: { badge: '需关注', text: '数据质量存在风险，建议结合人工复核' },
  poor: { badge: '较差', text: '数据质量较差，不足以支持强结论' },
  unknown: { badge: '未知', text: '数据质量信息待补充' }
}

const REPORT_CONFIDENCE_META = {
  high: { badge: '高', text: '报告整体可信度较高' },
  medium: { badge: '中', text: '报告可信度中等，建议结合证据复核' },
  low: { badge: '低', text: '报告可信度偏低，需人工确认' },
  unknown: { badge: '未知', text: '可信度信息待补充' }
}

const REVIEW_STATUS_META = {
  pending: { badge: '待复核', text: '尚未完成人工复核' },
  confirmed: { badge: '已确认', text: '人工已确认诊断结论' },
  false_alarm: { badge: '误报', text: '人工判定为误报' },
  ignored: { badge: '已忽略', text: '人工选择忽略本次告警' }
}

const METRIC_LABELS = {
  motor_imbalance_ratio: '电机输出均衡性',
  esc_rpm_imbalance_ratio: 'ESC 转速一致性',
  battery_remaining_min: '最低剩余电量',
  prediction_confidence: '模型判断置信度'
}

const OFFLINE_CLASS_SET = ['normal', 'motor', 'sensor']

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

function getDataQualityMeta(level) {
  return DATA_QUALITY_META[String(level || '').trim()] || DATA_QUALITY_META.unknown
}

function getReportConfidenceMeta(level) {
  return REPORT_CONFIDENCE_META[String(level || '').trim()] || REPORT_CONFIDENCE_META.unknown
}

function getReviewStatusMeta(status) {
  return REVIEW_STATUS_META[String(status || 'pending').trim()] || REVIEW_STATUS_META.pending
}

function getPriorityText(priority) {
  return PRIORITY_META[String(priority || 'P2').trim().toUpperCase()] || PRIORITY_META.P2
}

function formatPercent(value, digits = 0) {
  const numeric = toNumber(value)
  if (numeric === null) return '-'
  return `${(numeric * 100).toFixed(digits)}%`
}

function displayValue(value, placeholder = '-') {
  if (value === null || value === undefined || value === '') return placeholder
  if (Array.isArray(value)) return value.length ? value.join('、') : placeholder
  return String(value)
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

function buildWhyItems(report, confidenceText) {
  const evidence = Array.isArray(report?.summary?.severity?.evidence) ? report.summary.severity.evidence : []
  const whyItems = []

  evidence.slice(0, 3).forEach((item) => {
    const metricLabel = METRIC_LABELS[item.metric] || String(item.metric || '关键指标')
    const scoreText = item.score != null ? `，规则贡献分 ${item.score}` : ''
    const statusText = item.status ? `，状态 ${item.status}` : ''
    whyItems.push({
      title: metricLabel,
      description: `${metricLabel} 当前观测值为 ${displayValue(item.value)}${scoreText}${statusText}。`
    })
  })

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

function buildDataQualityView(summary) {
  const raw = summary?.data_quality && typeof summary.data_quality === 'object'
    ? summary.data_quality
    : {}
  const meta = getDataQualityMeta(raw.level)
  const timeRange = raw.time_range && typeof raw.time_range === 'object' ? raw.time_range : {}
  const warnings = Array.isArray(raw.warnings) ? raw.warnings : []

  const metrics = [
    { label: '样本数量', value: displayValue(raw.sample_count, '未知') },
    { label: '窗口数量', value: displayValue(raw.window_count, '未知') },
    { label: '特征维度', value: displayValue(raw.feature_count, '未知') }
  ]

  if (timeRange.start || timeRange.end) {
    metrics.push({
      label: '时间范围',
      value: `${displayValue(timeRange.start, '未知')} ~ ${displayValue(timeRange.end, '未知')}`
    })
  }

  if (Array.isArray(raw.missing_columns) && raw.missing_columns.length) {
    metrics.push({ label: '缺失列', value: raw.missing_columns.join('、') })
  }

  if (raw.missing_value_ratio != null) {
    metrics.push({ label: '缺失值比例', value: formatPercent(raw.missing_value_ratio, 1) })
  }

  return {
    badge: meta.badge,
    summary: meta.text,
    metrics,
    warnings
  }
}

function buildTraceabilityView(summary, report) {
  const raw = summary?.traceability && typeof summary.traceability === 'object'
    ? summary.traceability
    : {}
  const modelCard = summary?.model_card && typeof summary.model_card === 'object'
    ? summary.model_card
    : {}
  const classSet = Array.isArray(modelCard.class_set) && modelCard.class_set.length
    ? modelCard.class_set
    : OFFLINE_CLASS_SET

  return {
    items: [
      { label: '任务 ID', value: displayValue(raw.run_id ?? report?.run_id) },
      { label: '资产文件', value: displayValue(raw.asset_filename) },
      { label: '输入文件', value: displayValue(raw.input_csv) },
      { label: '模型名称', value: displayValue(raw.model_name ?? report?.model_name) },
      { label: '模型版本', value: displayValue(raw.model_version, '待补充') },
      { label: '规则版本', value: displayValue(raw.rules_version, 'rule_v1') },
      { label: '任务类型', value: displayValue(modelCard.task, 'offline_multiclass') },
      { label: '类别集', value: classSet.join(' / ') },
      { label: '生成时间', value: displayValue(raw.generated_at ?? summary?.basic?.report_generated_at) }
    ],
    intendedUse: displayValue(modelCard.intended_use, '待补充'),
    limitations: Array.isArray(modelCard.limitations) ? modelCard.limitations : []
  }
}

function buildReportConfidenceView(summary, report) {
  const raw = summary?.report_confidence && typeof summary.report_confidence === 'object'
    ? summary.report_confidence
    : {}
  const meta = getReportConfidenceMeta(raw.level)
  const reasons = Array.isArray(raw.reasons) ? raw.reasons : []
  const score = raw.score != null ? raw.score : report?.confidence

  return {
    badge: meta.badge,
    summary: meta.text,
    scoreText: formatPercent(score, 0),
    reasons
  }
}

function buildReviewView(summary) {
  const raw = summary?.review && typeof summary.review === 'object' ? summary.review : {}
  const meta = getReviewStatusMeta(raw.status)

  return {
    badge: meta.badge,
    summary: meta.text,
    reviewer: displayValue(raw.reviewer, '待补充'),
    reviewedAt: displayValue(raw.reviewed_at, '待补充'),
    conclusion: displayValue(raw.conclusion, '待补充'),
    maintenanceNotes: displayValue(raw.maintenance_notes, '待补充')
  }
}

export function buildDiagnosisReportView(report = {}) {
  const summary = report?.summary && typeof report.summary === 'object' ? report.summary : {}
  const labelMeta = getLabelMeta(report.prediction_label)
  const severityMeta = getSeverityMeta(summary?.severity?.level)
  const confidenceText = formatPercent(report.confidence, 0)
  const actionItems = normalizeRecommendations(report, labelMeta.abnormal)
  const probabilityRows = buildProbabilityRows(report)

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
    whyItems: buildWhyItems(report, confidenceText),
    actionItems,
    probabilityRows,
    dataQuality: buildDataQualityView(summary),
    traceability: buildTraceabilityView(summary, report),
    reportConfidence: buildReportConfidenceView(summary, report),
    review: buildReviewView(summary)
  }
}
