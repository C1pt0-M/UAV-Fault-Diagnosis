import { computed, onMounted, ref, watch } from 'vue'
import { useRoute } from 'vue-router'
import { ElMessage } from 'element-plus'
import { aiApi } from '@/shared/services'
import {
  aiCopy,
  createUserDraftMessage,
  normalizeAssistantMessage,
  normalizeConversationSummary
} from '@/modules/ai/aiView'

export function useAIAssistantWorkspace() {
  const route = useRoute()
  const scene = ref('general')
  const reportIdInput = ref('')
  const composer = ref('')
  const conversations = ref([])
  const currentConversationId = ref(null)
  const currentConversation = ref(null)
  const messages = ref([])
  const knowledge = ref({ total_chunks: 0, retrieval_mode: 'bm25' })
  const loadingConversations = ref(false)
  const loadingMessages = ref(false)
  const sending = ref(false)
  const reindexingKnowledge = ref(false)
  const autoAnalyzedKeys = ref(new Set())

  const canSeedAnalysis = computed(() => {
    return Boolean(scene.value === 'report_analysis' && String(reportIdInput.value || '').trim())
  })

  const contextSummary = computed(() => {
    const current = normalizeConversationSummary(currentConversation.value)
    const lastAssistant = [...messages.value].reverse().find((item) => item.role === 'assistant')
    const usedContext = lastAssistant?.used_context || {}
    const reportId = usedContext.report_id ?? current.reportId ?? reportIdInput.value ?? null
    const runId = usedContext.run_id ?? current.runId ?? null

    return {
      conversationId: currentConversationId.value,
      sceneLabel: currentConversation.value
        ? current.sceneLabel
        : (aiCopy.sceneLabels[scene.value] || aiCopy.sceneLabels.general),
      reportId: reportId || null,
      runId,
      provider: usedContext.provider || '',
      model: usedContext.model || '',
      knowledgeHits: usedContext.knowledge_hits ?? 0,
      retrievalMode: usedContext.knowledge_retrieval_mode || knowledge.value.retrieval_mode || '',
      fallbackUsed: Boolean(usedContext.fallback_used)
    }
  })

  function resetConversationState({ resetBindings = true } = {}) {
    currentConversationId.value = null
    currentConversation.value = null
    messages.value = []
    composer.value = ''

    if (resetBindings) {
      scene.value = 'general'
      reportIdInput.value = ''
    }
  }

  async function loadConversations() {
    loadingConversations.value = true
    try {
      const response = await aiApi.conversations()
      conversations.value = response.data || []
    } catch (error) {
      ElMessage.error(`加载会话失败：${error.message || '未知错误'}`)
    } finally {
      loadingConversations.value = false
    }
  }

  function applyConversationContext(conversation) {
    currentConversation.value = conversation || null
    if (!conversation) {
      return
    }

    scene.value = conversation.scene || scene.value
    reportIdInput.value = conversation.report_id ? String(conversation.report_id) : ''
  }

  async function openConversation(conversationId) {
    currentConversationId.value = conversationId
    loadingMessages.value = true

    try {
      const response = await aiApi.conversationDetail(conversationId)
      const payload = response.data || {}
      applyConversationContext(payload.conversation || null)
      messages.value = (payload.messages || []).map((item) => normalizeAssistantMessage(item))
    } catch (error) {
      ElMessage.error(`加载对话失败：${error.message || '未知错误'}`)
    } finally {
      loadingMessages.value = false
    }
  }

  function startFreshConversation() {
    resetConversationState({ resetBindings: true })
  }

  function applyRouteQuery({ resetConversation = false } = {}) {
    const routeScene = String(route.query.scene || '').trim()
    const routeReportId = String(route.query.report_id || '').trim()

    if (!routeScene && !routeReportId) {
      if (resetConversation) {
        resetConversationState({ resetBindings: true })
      }
      return
    }

    if (resetConversation) {
      resetConversationState({ resetBindings: true })
    }

    if (routeScene && aiCopy.sceneLabels[routeScene]) {
      scene.value = routeScene
    }

    if (routeReportId) {
      reportIdInput.value = routeReportId
    }
  }

  function shouldAutoAnalyze() {
    return (
      route.query.auto_analyze === '1' &&
      scene.value === 'report_analysis' &&
      String(reportIdInput.value || '').trim() &&
      !currentConversationId.value &&
      messages.value.length === 0 &&
      !sending.value
    )
  }

  async function maybeAutoAnalyzeRouteReport() {
    if (!shouldAutoAnalyze()) return

    const reportId = String(reportIdInput.value || '').trim()
    const autoKey = `${route.query.scene || ''}:${reportId}:${route.query.auto_analyze || ''}`
    if (autoAnalyzedKeys.value.has(autoKey)) return

    autoAnalyzedKeys.value.add(autoKey)
    await seedReportAnalysis()
  }

  function appendResponseMessages({ userMessage, payload }) {
    const conversation = payload?.conversation || null
    const assistantMessage = normalizeAssistantMessage(payload?.message, {
      citations: payload?.citations || [],
      used_context: payload?.used_context || {}
    })

    currentConversationId.value = conversation?.id ?? currentConversationId.value
    applyConversationContext(conversation)

    messages.value = [
      ...messages.value,
      createUserDraftMessage(userMessage),
      assistantMessage
    ]
  }

  async function sendMessage() {
    const userMessage = String(composer.value || '').trim()
    if (!userMessage) return

    sending.value = true
    try {
      const response = await aiApi.chat({
        conversation_id: currentConversationId.value || undefined,
        scene: scene.value,
        message: userMessage,
        report_id: reportIdInput.value || undefined
      })

      appendResponseMessages({ userMessage, payload: response.data })
      composer.value = ''
      await loadConversations()
    } catch (error) {
      ElMessage.error(`发送消息失败：${error.message || '未知错误'}`)
    } finally {
      sending.value = false
    }
  }

  async function seedReportAnalysis() {
    const reportId = String(reportIdInput.value || '').trim()
    if (!reportId) return

    sending.value = true
    try {
      const response = await aiApi.analyzeReport({
        report_id: reportId,
        conversation_id: currentConversationId.value || undefined
      })

      appendResponseMessages({
        userMessage: aiCopy.defaultReportPrompt,
        payload: response.data
      })
      await loadConversations()
    } catch (error) {
      ElMessage.error(`启动报告分析失败：${error.message || '未知错误'}`)
    } finally {
      sending.value = false
    }
  }

  async function loadKnowledge() {
    try {
      const response = await aiApi.knowledgeSources()
      knowledge.value = response.data || { total_chunks: 0, retrieval_mode: 'bm25' }
    } catch {
      knowledge.value = { total_chunks: 0, retrieval_mode: 'bm25' }
    }
  }

  async function reindexKnowledge() {
    reindexingKnowledge.value = true
    try {
      const response = await aiApi.reindexKnowledge()
      await loadKnowledge()
      ElMessage.success(response.message || aiCopy.reindexKnowledgeSuccess)
    } catch (error) {
      ElMessage.error(`重建知识库失败：${error.message || '未知错误'}`)
    } finally {
      reindexingKnowledge.value = false
    }
  }

  watch(scene, () => {
    if (scene.value === 'general') {
      reportIdInput.value = ''
    }
  })

  watch(
    () => [route.query.scene, route.query.report_id, route.query.auto_analyze],
    async () => {
      applyRouteQuery({ resetConversation: true })
      await maybeAutoAnalyzeRouteReport()
    }
  )

  onMounted(async () => {
    await loadConversations()
    await loadKnowledge()
    applyRouteQuery()
    await maybeAutoAnalyzeRouteReport()
  })

  return {
    canSeedAnalysis,
    composer,
    contextSummary,
    conversations,
    currentConversationId,
    knowledge,
    loadingConversations,
    loadingMessages,
    messages,
    openConversation,
    reportIdInput,
    reindexKnowledge,
    reindexingKnowledge,
    scene,
    seedReportAnalysis,
    sendMessage,
    sending,
    startFreshConversation
  }
}
