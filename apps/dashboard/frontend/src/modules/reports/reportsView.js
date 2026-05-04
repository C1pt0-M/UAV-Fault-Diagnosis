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
  technicalSection: '技术详情',
  generatedAtLabel: '生成时间',
  modelLabel: '模型',
  assetLabel: '资产',
  confidenceLabel: '置信度',
  noReportDetail: '暂无报告详情'
}

export function createDefaultReportFilters() {
  return {
    assetId: undefined,
    label: ''
  }
}
