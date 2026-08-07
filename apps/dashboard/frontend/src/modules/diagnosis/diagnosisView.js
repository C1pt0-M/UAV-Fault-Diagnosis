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
    text: '电机绕组异常',
    summary: '动力输出与转速一致性出现异常，可能影响姿态稳定和推力响应。',
    accent: 'alert',
    riskBadge: '高风险',
    actionTitle: '优先检查电机与 ESC 协同状态',
    actionSummary: '建议立即复核电机、ESC 与供电链路，并结合原始日志确认异常时段。',
    actionItems: ['重点检查电机 / ESC 联动状态', '复核供电与推力波动是否同步出现']
  },
  motor_failure: {
    text: '电机绕组异常',
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
    label: '诊断完成',
    description: '结果已就绪，建议生成报告并进入后续解释与复核环节。',
    cue: '生成报告并进入 AI 解读'
  },
  error: {
    key: 'error',
    label: '诊断失败',
    description: '离线诊断任务执行失败，请查看错误原因并重新提交。',
    cue: '检查错误后重试'
  }
}

export const diagnosisCopy = {
  sessionBannerTitle: '当前诊断来自监测会话承接',
  sessionBannerDescription(sessionId) {
    return `会话 ${sessionId} 已从实时监测页带入，可直接补做离线诊断。`
  },
  commandTitle: '任务发起 / 输入配置',
  commandDescription: '选择日志资产与诊断模型，启动可追溯的离线故障诊断。',
  stageLabel: '阶段',
  assetSummaryLabel: '资产',
  modelSummaryLabel: '模型',
  linkedSessionLabel: '承接会话',
  unselectedAsset: '尚未选择资产',
  unselectedModel: '尚未选择模型',
  formTitle: '离线诊断',
  formDescription: '选择日志资产与诊断模型，启动三分类故障诊断。',
  assetLabel: '日志资产',
  assetPlaceholder: '请选择日志资产',
  modelLabel: '诊断模型',
  modelPlaceholder: '请选择诊断模型',
  taskTypeLabel: '任务类型',
  submitButton: '启动离线诊断',
  rerunButton: '重新诊断',
  refreshButton: '刷新资产与模型',
  statusTitle: '任务流程',
  statusDescription: '展示数据加载、特征处理、模型推理、结果整理与报告生成状态。',
  resultTitle: '诊断结果',
  resultDescription: '查看本次诊断结论与类别概率，并生成诊断报告。',
  explanationTitle: '规则解释 / 诊断摘要',
  explanationDescription: '基于诊断结果摘要、概率分布、会话承接与规则模板生成可追溯解释。',
  evidenceTitle: '可追溯证据 TOP5',
  ruleTitle: '规则解释（判定依据）',
  historyTitle: '历史参考报告',
  recentRunsTitle: '最近诊断任务 / 报告出口',
  recentRunsDescription: '跟踪近期离线诊断任务和报告状态。',
  emptyRun: '尚未提交诊断任务',
  emptyProbability: '暂无概率分布',
  emptyReports: '暂无历史参考报告',
  emptyEvidence: '暂无可追溯证据；完成诊断后将根据结果摘要生成规则解释。',
  taskIdLabel: '任务 ID',
  statusLabel: '状态',
  createdAtLabel: '创建时间',
  updatedAtLabel: '更新时间',
  predictionLabel: '诊断结论',
  confidenceLabel: '诊断置信度',
  reportButton: '生成报告',
  reportLoadingButton: '正在生成报告',
  reportReadyButton: '报告已生成',
  reportCenterButton: '查看报告',
  aiActionButton: 'AI 解读',
  probabilityTitle: '类别概率',
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
  if (String(currentRun?.status || '').toLowerCase() === 'error') return STAGE_META.error
  if (currentRun) return STAGE_META.running
  return STAGE_META.idle
}

export function getDiagnosisLabelMeta(label) {
  return LABEL_META[String(label || '').trim()] || LABEL_META.unknown
}

export function formatDiagnosisModelName(value) {
  const raw = String(value || '').trim()
  if (!raw) return diagnosisCopy.unselectedModel
  const normalized = raw
    .replace(/^online_detector_/i, '')
    .replace(/^transfer\d*_/i, '')
    .replace(/_normalproto$/i, '')
    .replace(/_rfly$/i, '')
    .replace(/[_-]+/g, ' ')
    .trim()

  if (/kg\s*mstca/i.test(normalized)) return 'KG-MSTCA'
  if (/cnn\s*lstm/i.test(normalized)) return 'CNN-LSTM'
  if (/cnn\s*1d/i.test(normalized)) return 'CNN1D'
  if (/mstca/i.test(normalized)) return 'MSTCA'
  return raw
}

export function resolveRiskLevel({ label, confidence, severity } = {}) {
  const normalized = String(label || '').toLowerCase()
  const level = String(severity?.level || severity || '').toLowerCase()
  const value = Number(confidence || 0)

  if (normalized === 'normal') {
    return {
      label: '风险可控',
      tone: 'ok',
      description: '当前结果倾向正常，建议作为对照样本保留。'
    }
  }
  if (['heavy', 'high', 'critical', 'danger'].includes(level) || value >= 0.8) {
    return {
      label: '高风险',
      tone: 'danger',
      description: '故障类别置信度较高，建议优先复核并生成诊断报告。'
    }
  }
  if (['medium', 'warning', 'warn'].includes(level) || value >= 0.5) {
    return {
      label: '中风险',
      tone: 'warn',
      description: '结果存在明确异常倾向，建议结合原始日志继续复核。'
    }
  }
  return {
    label: '低风险',
    tone: 'ok',
    description: '结果置信度较低，建议补充上下文后再研判。'
  }
}

export function buildTraceableEvidenceRows({
  resultData,
  probabilityRows = [],
  activeSessionId = '',
  selectedAssetName = ''
} = {}) {
  const summary = resultData?.evidence_summary || resultData?.summary?.evidence_summary || {}
  const rawEvidence = summary.top_evidence || resultData?.evidence || resultData?.key_params || []
  const rows = []

  if (Array.isArray(rawEvidence)) {
    rawEvidence.slice(0, 5).forEach((item, index) => {
      const name = item.name || item.label || item.title || `证据 ${index + 1}`
      rows.push({
        rank: index + 1,
        name,
        source: item.source || item.kind || '结果摘要',
        window: item.window || item.time_window || (activeSessionId ? `会话 ${activeSessionId}` : '当前任务'),
        note: item.note || item.description || item.value || '来自诊断结果摘要',
        tone: index === 0 ? 'danger' : 'active'
      })
    })
  } else if (rawEvidence && typeof rawEvidence === 'object') {
    Object.entries(rawEvidence).slice(0, 5).forEach(([key, value], index) => {
      rows.push({
        rank: index + 1,
        name: key,
        source: '关键参数',
        window: activeSessionId ? `会话 ${activeSessionId}` : '当前任务',
        note: String(value),
        tone: index === 0 ? 'danger' : 'active'
      })
    })
  }

  if (rows.length) return rows

  probabilityRows.slice(0, 3).forEach((row, index) => {
    rows.push({
      rank: index + 1,
      name: `${row.label} 概率 ${row.probability}`,
      source: '类别概率',
      window: activeSessionId ? `会话 ${activeSessionId}` : '当前任务',
      note: '规则参考：用于说明模型输出排序，不代表真实特征贡献算法。',
      tone: index === 0 ? 'danger' : 'active'
    })
  })

  if (selectedAssetName && rows.length < 5) {
    rows.push({
      rank: rows.length + 1,
      name: '来源资产已锁定',
      source: '日志资产',
      window: selectedAssetName,
      note: '用于追溯本次离线诊断输入。',
      tone: 'ok'
    })
  }

  return rows
}

export function buildRuleExplanationRows({
  resultData,
  probabilityRows = [],
  activeSessionId = ''
} = {}) {
  const prediction = resultData?.prediction || {}
  const confidence = Number(prediction.confidence || 0)
  const topRow = probabilityRows[0]
  const rows = []

  if (topRow) {
    rows.push({
      title: '规则 1',
      detail: `最高概率类别为 ${topRow.label}，概率为 ${topRow.probability}，作为主要诊断结论参考。`,
      status: 'ok'
    })
  }
  rows.push({
    title: '规则 2',
    detail: confidence >= 0.8
      ? '结果置信度达到高风险提示阈值，建议生成报告并进行人工复核。'
      : '结果置信度未达到高风险阈值，建议结合原始日志继续确认。',
    status: confidence >= 0.8 ? 'warn' : 'ok'
  })
  rows.push({
    title: '规则 3',
    detail: activeSessionId
      ? `本次诊断承接实时监测会话 ${activeSessionId}，可回溯到对应采集上下文。`
      : '本次诊断未承接实时监测会话，仅基于所选日志资产进行解释。',
    status: activeSessionId ? 'active' : 'offline'
  })
  rows.push({
    title: '规则 4',
    detail: '以上解释来自诊断摘要与规则匹配，不代表真实特征贡献算法。',
    status: 'pending'
  })
  return rows
}
