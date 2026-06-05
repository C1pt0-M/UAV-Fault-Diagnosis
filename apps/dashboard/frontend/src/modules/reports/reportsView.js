export const reportLabelOptions = [
  { label: '状态正常', value: 'normal' },
  { label: '电机故障', value: 'motor' },
  { label: '传感器故障', value: 'sensor' }
]

export const reportsCopy = {
  title: '诊断报告中心',
  description: '集中查看诊断结论、概率分布与处置建议，并继续追踪对应资产与任务。',
  assetFilterPlaceholder: '按资产筛选',
  labelFilterPlaceholder: '按标签筛选',
  resetButton: '重置',
  refreshButton: '刷新',
  emptyText: '暂无诊断报告',
  detailTitle: '诊断报告',
  detailAction: '详情',
  aiAction: 'AI 解读',
  downloadAction: '下载',
  removeAction: '删除',
  removeSuccess: '报告删除成功',
  removeConfirmMessage(row) {
    return `确定删除报告 #${row.id} 吗？此操作不可恢复。`
  },
  removeConfirmTitle: '删除确认',
  summarySection: '结论摘要',
  recommendationSection: '处置建议',
  evidenceSection: '关键依据',
  probabilitySection: '概率分布',
  dataQualitySection: '数据质量',
  traceabilitySection: '模型与规则追溯',
  confidenceSection: '报告可信度',
  reviewSection: '人工复核',
  technicalSection: '技术详情',
  generatedAtLabel: '生成时间',
  modelLabel: '模型',
  assetLabel: '资产',
  confidenceLabel: '置信度',
  noReportDetail: '暂无报告详情',
  paginationTotal(total) {
    return `共 ${total || 0} 份报告`
  }
}

const reportLabelTextMap = {
  normal: '状态正常',
  motor: '电机故障',
  sensor: '传感器故障',
  imu: 'IMU 异常',
  battery: '电池异常',
  unknown: '未知故障'
}

const modelDisplayNameMap = [
  { match: 'kg_mstca', label: 'KG-MSTCA' },
  { match: 'mstca', label: 'MSTCA' },
  { match: 'cnnlstm', label: 'CNN-LSTM' },
  { match: 'cnn_lstm', label: 'CNN-LSTM' }
]

export function createDefaultReportFilters() {
  return {
    assetId: undefined,
    label: '',
    page: 1,
    perPage: 20
  }
}

export function formatReportConfidence(value) {
  const numeric = Number(value)
  if (!Number.isFinite(numeric)) return '暂无'
  return `${(numeric * 100).toFixed(2)}%`
}

export function getReportLabelText(label) {
  return reportLabelTextMap[label] || label || '暂无'
}

export function getReportModelDisplayName(modelName) {
  const rawName = String(modelName || '').trim()
  if (!rawName) return '暂无'

  const lowered = rawName.toLowerCase()
  const matched = modelDisplayNameMap.find((item) => lowered.includes(item.match))
  if (matched) return matched.label

  return rawName.replace(/_/g, ' ')
}

export function createReportSummaryItems(reports = [], pagination = {}) {
  const reportCount = Number(pagination.total ?? reports.length)
  const pageReportCount = reports.length
  const confidenceValues = reports
    .map((item) => Number(item.confidence))
    .filter((value) => Number.isFinite(value))
  const avgConfidence = confidenceValues.length
    ? confidenceValues.reduce((sum, value) => sum + value, 0) / confidenceValues.length
    : null

  return [
    { label: '报告总数', value: reportCount, tone: 'primary' },
    { label: '当前页报告', value: pageReportCount, tone: 'primary' },
    { label: '平均置信度', value: avgConfidence == null ? '暂无' : formatReportConfidence(avgConfidence), tone: 'ok' },
    { label: '可 AI 解读', value: pageReportCount, tone: 'ai' }
  ]
}

export function createReportActionItems() {
  return [
    { key: 'detail', label: reportsCopy.detailAction, type: 'primary' },
    { key: 'ai', label: reportsCopy.aiAction, type: 'warning' },
    { key: 'download', label: reportsCopy.downloadAction, type: 'success' },
    { key: 'remove', label: reportsCopy.removeAction, type: 'danger' }
  ]
}
