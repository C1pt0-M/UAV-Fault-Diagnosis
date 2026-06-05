export const reportLabelOptions = [
  { label: '正常', value: 'normal' },
  { label: '电机故障', value: 'motor' },
  { label: '传感器故障', value: 'sensor' }
]

export const reportsCopy = {
  title: '诊断报告',
  description: '筛选、查看和下载离线诊断报告，回溯对应资产与模型。',
  assetFilterPlaceholder: '按资产筛选',
  labelFilterPlaceholder: '按标签筛选',
  resetButton: '重置',
  refreshButton: '刷新',
  emptyText: '暂无诊断报告',
  detailTitle: '报告详情',
  detailAction: '详情',
  aiAction: 'AI 解读',
  downloadAction: '下载',
  removeAction: '删除',
  removeSuccess: '报告删除成功',
  removeConfirmMessage(row) {
    return `确定删除报告 #${row.id} 吗？此操作不可恢复。`
  },
  removeConfirmTitle: '删除确认'
}

export function createDefaultReportFilters() {
  return {
    assetId: undefined,
    label: ''
  }
}
