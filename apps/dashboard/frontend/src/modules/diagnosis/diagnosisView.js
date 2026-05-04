const LABEL_META = {
  normal: {
    text: '状态正常',
    summary: '当前样本未识别出明显故障特征，系统建议继续保持例行巡检与任务记录。',
    accent: 'stable',
    riskBadge: '风险可控',
    actionTitle: '继续保持任务闭环观察',
    actionSummary: '建议保留本次结果作为正常样本，对后续飞行任务继续进行例行检查。',
    actionItems: ['保留当前日志作为正常对照样本', '继续执行例行巡检与飞前检查']
  },
  motor: {
    text: '电机故障',
    summary: '动力输出与转速一致性出现异常，可能影响姿态稳定和推力响应。',
    accent: 'alert',
    riskBadge: '高风险',
    actionTitle: '优先检查电机与 ESC 协同状态',
    actionSummary: '建议立即复核电机、ESC 与供电链路，并结合原始日志确认异常时段。',
    actionItems: ['重点检查电机 / ESC 联动状态', '复核供电与推力波动是否同步出现']
  },
  sensor: {
    text: '传感器故障',
    summary: '姿态感知链路出现异常，可能影响飞控估计与姿态控制精度。',
    accent: 'warning',
    riskBadge: '中高风险',
    actionTitle: '优先复核 IMU 与关键传感器质量',
    actionSummary: '建议尽快检查 IMU、姿态估计链路和相关传感器数据一致性。',
    actionItems: ['核查 IMU 数据漂移与噪声', '结合原始遥测复核传感器链路稳定性']
  },
  unknown: {
    text: '待确认异常',
    summary: '当前结果尚未形成稳定归类，建议结合原始数据与任务背景进行人工复核。',
    accent: 'pending',
    riskBadge: '待确认',
    actionTitle: '需要进一步人工研判',
    actionSummary: '请结合原始日志、飞行场景与历史记录继续确认故障成因。',
    actionItems: ['补充上下文信息后重新分析', '结合报告与原始数据开展人工复核']
  }
}

const STAGE_META = {
  idle: {
    key: 'idle',
    label: '待开始',
    description: '锁定日志资产与诊断模型后，即可发起本次离线诊断任务。',
    cue: '准备启动离线诊断'
  },
  running: {
    key: 'running',
    label: '诊断进行中',
    description: '系统正在执行模型推理与结果整理，并持续刷新任务状态。',
    cue: '等待模型完成推理'
  },
  completed: {
    key: 'completed',
    label: '任务已完成',
    description: '结果已就绪，建议立即生成报告并进入后续解释与复核环节。',
    cue: '生成报告并进入 AI 解读'
  }
}

export const diagnosisCopy = {
  sessionBannerTitle: '当前诊断来自监测会话承接',
  sessionBannerDescription(sessionId) {
    return `会话 ${sessionId} 已从实时监测页带入，可直接补做离线诊断。`
  },
  commandTitle: '离线诊断任务指挥台',
  commandDescription: '把任务发起、状态跟踪、结果展示与报告出口收拢为一块统一的诊断工作台。',
  stageLabel: '当前阶段',
  assetSummaryLabel: '当前资产',
  modelSummaryLabel: '当前模型',
  linkedSessionLabel: '承接会话',
  unselectedAsset: '尚未选择资产',
  unselectedModel: '尚未选择模型',
  formTitle: '任务发起区',
  formDescription: '选择日志资产与诊断模型后，即可启动离线故障分类任务。',
  assetLabel: '日志资产',
  assetPlaceholder: '请选择日志资产',
  modelLabel: '诊断模型',
  modelPlaceholder: '请选择诊断模型',
  submitButton: '启动离线诊断',
  refreshButton: '刷新资产与模型',
  statusTitle: '任务状态雷达',
  statusDescription: '用更直观的方式展示阶段推进、进度百分比和关键任务元信息。',
  resultTitle: '诊断结果主舞台',
  resultDescription: '将故障结论、风险判断、处置建议与报告出口收束为一块核心展示区。',
  emptyRun: '尚未提交诊断任务',
  emptyProbability: '暂无概率分布',
  taskIdLabel: '任务 ID',
  taskTypeLabel: '任务类型',
  statusLabel: '状态',
  createdAtLabel: '创建时间',
  predictionLabel: '诊断结论',
  confidenceLabel: '置信度',
  resultActionTitle: '后续动作',
  reportButton: '生成诊断报告',
  reportLoadingButton: '正在生成诊断报告',
  reportReadyButton: '报告已生成',
  reportCenterButton: '查看报告中心',
  aiActionButton: 'AI 解读',
  probabilityTitle: 'Top-K 概率分布',
  recommendationTitle: '处置建议',
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
