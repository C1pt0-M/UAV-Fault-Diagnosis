export const assetKindTabs = [
  { label: '全部', value: 'all' },
  { label: 'ULG', value: 'ulg' },
  { label: 'CSV', value: 'csv' }
]

export const assetsCopy = {
  title: '日志资产',
  description: '集中管理 ULG / CSV 资产，支撑回放、诊断与报告追溯。',
  uploadButton: '上传资产',
  emptyText: '暂无日志资产',
  detailTitle: '资产详情',
  preparingDone: '已完成',
  preparingPending: '未完成',
  downloadAction: '下载',
  prepareAction: '预处理',
  detailAction: '详情',
  removeAction: '删除',
  prepareSuccess: '资产预处理完成',
  prepareAlreadyDone: '资产已经完成预处理',
  removeSuccess: '资产删除成功',
  removeConfirmMessage(row) {
    return `确定要删除资产 #${row.id} ${row.filename} 吗？此操作不可恢复。`
  },
  removeConfirmTitle: '删除确认'
}

export function formatAssetSize(bytes) {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  if (bytes < 1024 * 1024 * 1024) return `${(bytes / 1024 / 1024).toFixed(1)} MB`
  return `${(bytes / 1024 / 1024 / 1024).toFixed(1)} GB`
}
