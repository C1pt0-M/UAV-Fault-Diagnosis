export const aiQuickPrompts = [
  '解释当前报告结论',
  '列出风险和处置建议',
  '说明模型判断依据'
]

export const aiCopy = {
  heroTag: 'AI 助手',
  heroTitle: '报告解读与知识问答工作台',
  heroDescription: '围绕诊断报告、任务记录与知识库进行解释、追问和依据回溯，让分析过程更聚焦。',
  heroPrimaryCardTitle: '分析入口',
  heroPrimaryCardText: '优先承接报告分析，其次支持通用问答与知识检索。',
  heroSecondaryCardTitle: '知识库状态',
  heroSecondaryCardText: '可直接查看检索方式、Embedding 状态与重建结果。',

  conversationListTitle: '会话轨迹',
  conversationListDescription: '按最近更新时间查看历史分析对话，快速回到上一次分析现场。',
  newConversation: '新建会话',
  emptyConversationList: '暂无 AI 会话',
  conversationFallbackTitle(id) {
    return `会话 #${id}`
  },
  conversationMetaSeparator: ' / ',
  conversationScenePrefix: '场景',
  conversationReportPrefix: '报告',
  conversationRunPrefix: '任务',
  conversationRailCollapsed: '会话',
  conversationRailExpanded: '收起',

  chatTitle: '分析主区',
  chatDescription: '直接提问，或基于当前报告发起分析。重点关注结论、原因、风险和建议。',
  emptyChat: '当前没有消息，输入问题后即可开始分析。',
  quickPromptTitle: '快捷问题',
  composerPlaceholder: '输入你的问题，例如：请结合当前报告解释故障原因、风险等级和处置建议。',
  sendButton: '发送',
  seedAnalysisButton: '启动报告分析',
  userRole: '用户',

  contextTitle: '上下文与知识库',
  contextDescription: '查看当前绑定的报告、任务、检索方式和 Embedding 状态，快速判断当前回答依据。',
  scenePanelTitle: '分析场景',
  scenePanelDescription: '先确定任务类型和报告编号，聊天主区会按当前分析上下文回答。',
  contextBoxTitle: '当前分析上下文',
  sceneLabel: '分析场景',
  reportIdLabel: '报告 ID',
  reportIdPlaceholder: '请输入报告 ID',
  currentConversation: '当前会话',
  newConversationLabel: '新会话',
  currentScene: '当前场景',
  boundReport: '绑定报告',
  boundRun: '关联任务',
  knowledgeChunks: '知识片段',
  knowledgeStatusTitle: '知识库状态',
  knowledgeDocuments: '知识文档',
  retrievalMode: '检索方式',
  requestedRetrievalMode: '请求检索方式',
  currentRetrievalMode: '当前检索方式',
  provider: '回答来源',
  model: '使用模型',
  fallback: '降级模式',
  fallbackEnabled: '已启用',
  fallbackDisabled: '未启用',
  embeddingStatus: 'Embedding 状态',
  embeddingEnabled: '正常',
  embeddingDisabled: '未启用',
  embeddingErrorState: '异常',
  embeddingModel: 'Embedding 模型',
  embeddingError: 'Embedding 错误',

  referencesLabel: '参考资料',
  referenceSnippet: '摘要',
  reindexKnowledgeButton: '重建知识库',
  reindexKnowledgeSuccess: '知识库索引已重建',

  sourceTypeLabels: {
    report: '诊断报告',
    run: '任务记录',
    knowledge_base: '知识库',
    asset: '关联资产'
  },
  sceneLabels: {
    general: '通用问答',
    report_analysis: '报告分析'
  },
  defaultReportPrompt: '请结合当前报告给出故障原因分析、风险判断和处理建议。'
}

export function createConversationPreview(content, maxLength = 56) {
  const normalized = String(content || '').replace(/\s+/g, ' ').trim()
  if (!normalized) return ''
  if (normalized.length <= maxLength) return normalized
  return `${normalized.slice(0, maxLength)}...`
}

export function normalizeConversationSummary(conversation) {
  const meta = conversation?.meta || {}
  const contextSummary = meta.context_summary || {}

  return {
    id: conversation?.id ?? null,
    title: conversation?.title || '',
    scene: conversation?.scene || null,
    sceneLabel: contextSummary.scene_label || aiCopy.sceneLabels[conversation?.scene] || aiCopy.sceneLabels.general,
    reportId: conversation?.report_id ?? contextSummary.report_id ?? null,
    runId: conversation?.run_id ?? contextSummary.run_id ?? null,
    lastPreview: createConversationPreview(meta.last_message_preview || ''),
    messageCount: meta.message_count ?? 0
  }
}

export function normalizeAssistantMessage(message, fallback = {}) {
  return {
    ...message,
    citations: Array.isArray(message?.citations) ? message.citations : (fallback.citations || []),
    used_context: typeof message?.used_context === 'object' && message?.used_context !== null
      ? message.used_context
      : (fallback.used_context || {})
  }
}

export function createUserDraftMessage(content) {
  return {
    id: `draft-user-${Date.now()}`,
    role: 'user',
    content,
    citations: [],
    used_context: {}
  }
}

export function formatSourceType(type) {
  return aiCopy.sourceTypeLabels[type] || type || '未知来源'
}
