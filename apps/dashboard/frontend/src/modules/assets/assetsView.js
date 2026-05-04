export const assetKindTabs = [
  { label: '全部资产', value: 'all' },
  { label: 'ULG 日志', value: 'ulg' },
  { label: 'CSV 数据', value: 'csv' }
]

export const assetsCopy = {
  title: '日志资产管理',
  description: '集中管理 ULG / CSV 资产，支持在线回放、诊断分析与报告追踪。',
  uploadButton: '上传新资产',
  reportsCenterAction: '前往报告中心',
  emptyText: '暂无日志资产，请点击上方按钮开始上传',
  detailTitle: '资产详细信息',
  preparingDone: '已就绪',
  preparingPending: '待处理',
  downloadAction: '下载',
  reportAction: '查看报告',
  prepareAction: '执行预处理',
  detailAction: '详情',
  removeAction: '删除',
  prepareSuccess: '资产预处理已提交并在后台运行',
  prepareAlreadyDone: '该资产已完成预处理，无需重复执行',
  removeSuccess: '资产已成功移除',
  removeConfirmMessage(row) {
    return '确定要永久删除资产 #' + row.id + ' (' + row.filename + ') 吗？该操作无法撤销。'
  },
  removeConfirmTitle: '资产删除确认',
  analysisTitle: '多维度字段分析',
  analysisDescription: '选择日志资产与特定传感器字段，按时间窗口实时绘制变化曲线，支持快速定位异常波动。',
  analysisAssetPlaceholder: '请搜索并选择要分析的资产',
  analysisFieldsPlaceholder: '可检索并选择多个传感器字段',
  analysisRefreshFields: '同步字段列表',
  analysisLoadSeries: '请求曲线数据',
  analysisNoSeries: '当前筛选条件下无法匹配到有效的时序数据',
  analysisLoadFieldsFailed: '同步资产字段元数据失败',
  analysisLoadSeriesFailed: '请求时序曲线失败，请检查后端服务状态'
}

export function formatAssetSize(bytes) {
  if (bytes < 1024) return bytes + ' B'
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB'
  if (bytes < 1024 * 1024 * 1024) return (bytes / 1024 / 1024).toFixed(1) + ' MB'
  return (bytes / 1024 / 1024 / 1024).toFixed(1) + ' GB'
}
