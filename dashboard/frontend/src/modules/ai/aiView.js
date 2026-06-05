export const aiCopy = {
  heroTag: 'AI 助手',
  heroTitle: '围绕报告、任务与知识库进行问答分析',
  heroDescription: '该模块不再承担系统配置职责，只专注于报告分析、知识片段问答和原因解释。',
  conversationListTitle: '会话列表',
  conversationListDescription: '按最近更新时间查看 AI 对话。',
  newConversation: '新建会话',
  emptyConversationList: '暂无 AI 会话',
  chatTitle: '对话面板',
  chatDescription: '直接提问，或以报告分析模式启动对话。',
  emptyChat: '当前没有消息，输入问题后即可发起分析。',
  composerPlaceholder: '输入你的问题，例如：请结合当前诊断报告分析最可能的故障原因。',
  sendButton: '发送',
  seedAnalysisButton: '启动报告分析',
  contextTitle: '上下文绑定',
  contextDescription: '限制 AI 的分析边界，保持问答围绕业务闭环。',
  sceneLabel: '分析场景',
  reportIdLabel: '报告 ID',
  reportIdPlaceholder: '请输入报告 ID',
  currentConversation: '当前会话',
  newConversationLabel: '新会话',
  currentScene: '当前场景',
  boundReport: '绑定报告',
  boundRun: '关联任务',
  knowledgeChunks: '知识片段',
  retrievalMode: '检索方式',
  provider: '回答来源',
  model: '使用模型',
  fallback: '降级模式',
  fallbackEnabled: '已启用',
  fallbackDisabled: '未启用',
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

export function normalizeConversationSummary(conversation) {
  const meta = conversation?.meta || {}
  const contextSummary = meta.context_summary || {}

  return {
    id: conversation?.id ?? null,
    scene: conversation?.scene || null,
    sceneLabel: contextSummary.scene_label || aiCopy.sceneLabels[conversation?.scene] || aiCopy.sceneLabels.general,
    reportId: conversation?.report_id ?? contextSummary.report_id ?? null,
    runId: conversation?.run_id ?? contextSummary.run_id ?? null
  }
}

export function normalizeAssistantMessage(message, fallback = {}) {
  const normalized = {
    ...message,
    citations: Array.isArray(message?.citations) ? message.citations : (fallback.citations || []),
    used_context: typeof message?.used_context === 'object' && message?.used_context !== null
      ? message.used_context
      : (fallback.used_context || {})
  }

  return normalized
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
