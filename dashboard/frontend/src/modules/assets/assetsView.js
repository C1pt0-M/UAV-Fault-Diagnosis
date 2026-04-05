export const assetKindTabs = [
  { label: '全部', value: 'all' },
  { label: 'ULG', value: 'ulg' },
  { label: 'CSV', value: 'csv' }
]

export const assetsCopy = {
  title: '日志资产',
  description: '集中管理 ULG / CSV 资产，支持回放、诊断与报告追踪。',
  uploadButton: '上传资产',
  reportsCenterAction: '进入报告页',
  emptyText: '暂无日志资产',
  detailTitle: '资产详情',
  preparingDone: '已完成',
  preparingPending: '未完成',
  downloadAction: '下载',
  reportAction: '查看报告',
  prepareAction: '预处理',
  detailAction: '详情',
  removeAction: '删除',
  prepareSuccess: '资产预处理完成',
  prepareAlreadyDone: '资产已经完成预处理',
  removeSuccess: '资产删除成功',
  removeConfirmMessage(row) {
    return `确定要删除资产 #${row.id} ${row.filename} 吗？此操作不可恢复。`
  },
  removeConfirmTitle: '删除确认',
  analysisTitle: '字段曲线分析',
  analysisDescription: '选择日志资产与字段，按时间窗口绘制曲线，支持快速定位异常波动。',
  analysisAssetPlaceholder: '选择要分析的资产',
  analysisFieldsPlaceholder: '选择字段（可多选）',
  analysisRefreshFields: '刷新字段',
  analysisLoadSeries: '加载曲线',
  analysisNoSeries: '当前筛选条件下没有可展示的曲线数据',
  analysisLoadFieldsFailed: '加载字段失败',
  analysisLoadSeriesFailed: '加载曲线失败'
}

export function formatAssetSize(bytes) {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  if (bytes < 1024 * 1024 * 1024) return `${(bytes / 1024 / 1024).toFixed(1)} MB`
  return `${(bytes / 1024 / 1024 / 1024).toFixed(1)} GB`
}
