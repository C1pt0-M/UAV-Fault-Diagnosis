const LABEL_META = {
  normal: {
    text: '状态正常',
    summary: '当前样本未发现明显故障迹象，可继续按例行巡检流程复核。',
    accent: 'stable'
  },
  motor: {
    text: '电机故障',
    summary: '动力输出与转速一致性存在异常，建议优先检查电机与 ESC 联动状态。',
    accent: 'alert'
  },
  sensor: {
    text: '传感器故障',
    summary: '姿态感知链路出现异常，建议尽快排查 IMU 与相关传感器数据质量。',
    accent: 'alert'
  },
  unknown: {
    text: '待确认异常',
    summary: '当前结果尚未形成稳定归类，建议结合原始数据与任务背景做人工复核。',
    accent: 'pending'
  }
}

const STAGE_META = {
  idle: {
    key: 'idle',
    label: '待开始',
    description: '选择日志资产与模型后即可启动本次离线诊断。',
    cue: '开始离线诊断'
  },
  running: {
    key: 'running',
    label: '诊断进行中',
    description: '任务正在处理中，系统会持续刷新状态并在完成后自动展示结果。',
    cue: '等待模型完成推理'
  },
  completed: {
    key: 'completed',
    label: '任务已完成',
    description: '结果已就绪，建议立即生成报告并进入后续分析闭环。',
    cue: '生成报告 / 查看报告中心'
  }
}

export const diagnosisCopy = {
  sessionBannerTitle: '当前诊断来自监测会话承接',
  sessionBannerDescription(sessionId) {
    return `会话 ${sessionId} 已从实时监测页带入，可直接补做离线诊断。`
  },
  commandTitle: '诊断任务指挥台',
  commandDescription: '围绕离线诊断、报告生成与 AI 解读建立更具展示冲击感的任务闭环。',
  stageLabel: '当前阶段',
  assetSummaryLabel: '当前资产',
  modelSummaryLabel: '当前模型',
  linkedSessionLabel: '承接会话',
  unselectedAsset: '尚未选择资产',
  unselectedModel: '尚未选择模型',
  formTitle: '开始离线诊断',
  formDescription: '锁定日志资产与诊断模型后，立即启动本次离线诊断任务。',
  assetLabel: '日志资产',
  assetPlaceholder: '请选择日志资产',
  modelLabel: '诊断模型',
  modelPlaceholder: '请选择诊断模型',
  submitButton: '开始离线诊断',
  refreshButton: '刷新资产与模型',
  statusTitle: '任务状态总览',
  statusDescription: '将当前任务的进度、阶段和关键元信息聚焦为中枢状态面板。',
  resultTitle: '结果摘要',
  resultDescription: '用更适合展示与答辩的方式呈现结论、风险和后续动作。',
  emptyRun: '尚未提交诊断任务',
  emptyProbability: '暂无概率分布',
  taskIdLabel: '任务 ID',
  taskTypeLabel: '任务类型',
  statusLabel: '状态',
  createdAtLabel: '创建时间',
  predictionLabel: '预测标签',
  confidenceLabel: '置信度',
  resultActionTitle: '后续动作',
  reportButton: '生成诊断报告',
  reportLoadingButton: '正在生成诊断报告',
  reportReadyButton: '报告已生成',
  reportCenterButton: '查看报告中心',
  aiActionButton: 'AI 解读',
  probabilityTitle: 'Top-3 概率',
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

export function resolveDiagnosisStage({ currentRun, resultData }) {
  if (resultData) return STAGE_META.completed
  if (currentRun) return STAGE_META.running
  return STAGE_META.idle
}

export function getDiagnosisLabelMeta(label) {
  return LABEL_META[String(label || '').trim()] || LABEL_META.unknown
}
