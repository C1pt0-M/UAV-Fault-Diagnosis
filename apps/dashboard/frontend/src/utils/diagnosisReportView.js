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
  motor_failure: {
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
  sensor_failure: {
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

const REPORT_INTENDED_USE = '飞行日志 / CSV 回放离线故障诊断辅助分析'

const METRIC_LABELS = {
  motor_imbalance_ratio: '电机输出均衡性',
  esc_rpm_imbalance_ratio: 'ESC 转速一致性',
  gyro_vibration_p95: '陀螺振动',
  accel_vibration_p95: '加速度振动',
  imbalanced_prop_metric_p95: '螺旋桨失衡特征',
  thrust_saturation_ratio: '推力饱和比例',
  unalloc_thrust_z_p95: '未分配推力',
  vz_down_p95: '下降速度异常',
  battery_remaining_min: '最低剩余电量',
  battery_voltage_drop_ratio: '电压跌落',
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
    value: item.status === 'missing'
      ? '-'
      : item.source === 'model'
        ? formatPercent(item.value, 1)
        : formatMetricValue(item.value),
    status: item.status,
    statusText: STATUS_LABELS[String(item.status || '').trim()] || String(item.status || '未知'),
    source: item.source || 'rule',
    sourceText: item.source === 'model' ? '模型证据' : '规则证据',
    reasonText: item.reason_text || (item.status === 'missing' ? '未从当前 CSV 中提取到对应规则指标。' : ''),
    score: item.score != null ? formatMetricValue(item.score) : '-'
  }))
}

function buildModelProbabilityText(probabilityRows) {
  const rows = Array.isArray(probabilityRows) ? probabilityRows : []
  if (!rows.length) return ''
  const top1 = rows[0]
  if (rows.length === 1) {
    return `模型当前主要判断为${top1.labelText}（${top1.probText}）。`
  }
  const top2 = rows[1]
  return `模型当前主要判断为${top1.labelText}（${top1.probText}），次高可能为${top2.labelText}（${top2.probText}）。`
}

function normalizeReadableEvidenceText(text) {
  return String(text || '').replaceAll('贡献分', '规则得分')
}

function buildWhyItems(report, confidenceText, evidenceChain, readableEvidence, probabilityRows) {
  const modelProbabilityText = buildModelProbabilityText(probabilityRows)
  if (Array.isArray(readableEvidence) && readableEvidence.length) {
    return readableEvidence.slice(0, 5).map((text, index) => ({
      title: index === 0 ? '模型概率' : '关键依据',
      description: index === 0 && modelProbabilityText
        ? modelProbabilityText
        : normalizeReadableEvidenceText(text)
    }))
  }

  const whyItems = evidenceChain.slice(0, 3).map((item) => {
    if (item.status === 'missing') {
      return {
        title: item.metricText,
        description: `${item.sourceText}缺失：${item.reasonText || '未记录可用数据。'}`
      }
    }

    if (item.source === 'model') {
      return {
        title: item.metricText,
        description: `模型给出的主分类置信度为 ${item.value}，用于辅助判断本次风险等级。`
      }
    }

    return {
      title: item.metricText,
      description: `${item.metricText} 当前观测值为 ${item.value}，规则得分 ${item.score}。`
    }
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
    probText: formatPercent(item.prob, 3)
  }))
}

function buildDataQualityView(report) {
  const raw = report?.summary?.data_quality
  const available = raw && typeof raw === 'object'
  if (!available) {
    return {
      available: false,
      level: 'unknown',
      levelText: '未记录',
      tone: 'muted',
      metrics: [],
      warnings: ['历史报告未记录数据质量字段，建议结合原始 CSV 或飞行日志复核。']
    }
  }

  const dataQuality = raw && typeof raw === 'object' ? raw : {}
  const level = String(dataQuality.level || 'unknown').trim() || 'unknown'
  const meta = DATA_QUALITY_META[level] || DATA_QUALITY_META.unknown

  const metrics = []
  if (dataQuality.sample_count != null) metrics.push({ label: 'CSV 行数', value: String(dataQuality.sample_count) })
  if (dataQuality.window_count != null && Number(dataQuality.window_count) > 0) {
    metrics.push({ label: '模型窗口数', value: String(dataQuality.window_count) })
  }
  if (dataQuality.feature_count != null && Number(dataQuality.feature_count) > 0) {
    metrics.push({ label: '特征数量', value: String(dataQuality.feature_count) })
  }
  if (dataQuality.missing_value_ratio != null) {
    metrics.push({ label: '缺失值占比', value: formatPercent(dataQuality.missing_value_ratio, 1) })
  }
  if (dataQuality.invalid_value_ratio != null) {
    metrics.push({ label: '异常值占比', value: formatPercent(dataQuality.invalid_value_ratio, 1) })
  }

  const timeRange = dataQuality.time_range && typeof dataQuality.time_range === 'object'
    ? dataQuality.time_range
    : {}
  if (timeRange.start || timeRange.end) {
    metrics.push({
      label: '时间范围',
      value: `${timeRange.start || '-'} ~ ${timeRange.end || '-'}`
    })
  }

  const warnings = Array.isArray(dataQuality.missing_columns) && dataQuality.missing_columns.length
    ? [`存在缺失列：${dataQuality.missing_columns.slice(0, 5).join('、')}`]
    : []
  if (Array.isArray(dataQuality.warnings)) {
    warnings.push(...dataQuality.warnings)
  }
  if (!warnings.length) {
    warnings.push('未发现明显数据质量告警。')
  }

  return {
    available: true,
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
    inputCsv: traceability.input_csv || '-',
    modelName: traceability.model_name || report?.model_name || '-',
    modelVersion: traceability.model_version || '-',
    modelHash: traceability.model_hash || '-',
    rulesVersion: traceability.rules_version || 'rule_v1',
    generatedAt: traceability.generated_at || report?.created_at || '-',
    task: modelCard.task || 'offline_multiclass',
    classSet,
    classSetText: classSet.join(' / '),
    intendedUse: modelCard.intended_use || REPORT_INTENDED_USE,
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
    : ['历史报告未记录报告可信度字段，建议结合原始数据人工复核。']

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
    reviewer: review.reviewer || '-',
    reviewedAt: review.reviewed_at || '-',
    conclusion: review.conclusion || '-',
    maintenanceNotes: review.maintenance_notes || '-'
  }
}

export function buildDiagnosisReportView(report = {}) {
  const predictionLabel = report.prediction_label || report?.summary?.conclusion?.label
  const labelMeta = getLabelMeta(predictionLabel)
  const severityMeta = getSeverityMeta(report?.summary?.severity?.level)
  const confidenceValue = report.confidence ?? report?.summary?.conclusion?.confidence
  const confidenceText = formatPercent(confidenceValue, 3)
  const actionItems = normalizeRecommendations(report, labelMeta.abnormal)
  const probabilityRows = buildProbabilityRows(report)
  const dataQuality = buildDataQualityView(report)
  const traceability = buildTraceabilityView(report)
  const reportConfidence = buildReportConfidenceView(report)
  const review = buildReviewView(report)
  const evidenceChain = buildEvidenceChain(report)
  const readable = report?.summary?.readable && typeof report.summary.readable === 'object'
    ? report.summary.readable
    : {}
  const readableEvidence = Array.isArray(readable.evidence)
    ? readable.evidence.filter((item) => String(item || '').trim())
    : []
  const readableRecommendations = Array.isArray(readable.recommendations)
    ? readable.recommendations.filter((item) => String(item || '').trim())
    : []
  const readableCautions = Array.isArray(readable.cautions)
    ? readable.cautions.filter((item) => String(item || '').trim())
    : []

  const fallbackHeadline = labelMeta.abnormal
    ? `本次离线诊断识别为${labelMeta.text}，当前属于${severityMeta.badge}。`
    : '本次离线诊断未发现明显故障，当前状态正常。'

  const fallbackRiskText = labelMeta.abnormal
    ? `当前结果显示${severityMeta.text}，建议立即复核关键部件与飞行记录。`
    : `当前结果显示${severityMeta.text}，模型对正常状态的判断置信度为 ${confidenceText}。`

  const actionSummary = readableRecommendations.length
    ? readableRecommendations.join('；')
    : actionItems.map((item) => item.title).join('；')

  return {
    labelText: labelMeta.text,
    headline: readable.conclusion || fallbackHeadline,
    riskText: readable.risk || fallbackRiskText,
    actionSummary,
    whyItems: buildWhyItems(report, confidenceText, evidenceChain, readableEvidence, probabilityRows),
    actionItems,
    probabilityRows,
    dataQuality,
    traceability,
    reportConfidence,
    review,
    evidenceChain,
    readableEvidence,
    readableRecommendations,
    readableCautions,
    severityBadge: severityMeta.badge,
    severityText: severityMeta.text,
    confidenceText
  }
}
