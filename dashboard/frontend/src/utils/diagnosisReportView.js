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

const METRIC_LABELS = {
  motor_imbalance_ratio: '电机输出均衡性',
  esc_rpm_imbalance_ratio: 'ESC 转速一致性',
  battery_remaining_min: '最低剩余电量',
  prediction_confidence: '模型判断置信度'
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
    whyItems.push({
      title: metricLabel,
      description: `${metricLabel} 出现异常，当前观测值为 ${item.value}。`
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

export function buildDiagnosisReportView(report = {}) {
  const labelMeta = getLabelMeta(report.prediction_label)
  const severityMeta = getSeverityMeta(report?.summary?.severity?.level)
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
    probabilityRows
  }
}
