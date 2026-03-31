export const diagnosisCopy = {
  sessionBannerTitle: '当前诊断来自监测会话承接',
  sessionBannerDescription(sessionId) {
    return `会话 ${sessionId} 已从实时监测页带入，可直接补做离线诊断。`
  },
  formTitle: '创建诊断任务',
  formDescription: '选择日志资产，或直接承接监测会话进入离线诊断。',
  assetLabel: '日志资产',
  assetPlaceholder: '请选择日志资产',
  modelLabel: '诊断模型',
  modelPlaceholder: '请选择诊断模型',
  submitButton: '提交诊断',
  refreshButton: '刷新资产与模型',
  statusTitle: '任务状态',
  statusDescription: '查看当前诊断任务进度，并在完成后生成报告。',
  resultTitle: '诊断结果',
  resultDescription: '展示预测标签、置信度和概率分布。',
  emptyRun: '尚未提交诊断任务',
  emptyProbability: '暂无概率分布',
  taskIdLabel: '任务 ID',
  taskTypeLabel: '任务类型',
  statusLabel: '状态',
  createdAtLabel: '创建时间',
  predictionLabel: '预测标签',
  confidenceLabel: '置信度',
  reportButton: '生成诊断报告',
  reportLoadingButton: '正在生成诊断报告',
  createRunFailed: '诊断任务创建失败',
  runExecutionFailed: '诊断任务执行失败',
  reportCreated: '诊断报告已生成',
  noModelWarning: '请先选择诊断模型',
  noAssetWarning: '请选择日志资产，或从监测会话自动承接'
}

export function formatPercentage(value) {
  return `${(Number(value || 0) * 100).toFixed(2)}%`
}

export function normalizeDiagnosisResult(payload) {
  if (!payload) return null
  if (payload.prediction) return payload
  return payload.result || payload
}
