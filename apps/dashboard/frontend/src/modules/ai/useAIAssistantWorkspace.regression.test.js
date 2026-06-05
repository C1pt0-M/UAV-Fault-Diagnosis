import { defineComponent, nextTick } from 'vue'
import { flushPromises, mount } from '@vue/test-utils'
import { beforeEach, describe, expect, it, vi } from 'vitest'

const routeState = vi.hoisted(() => ({
  query: {
    scene: 'report_analysis',
    report_id: '42',
    auto_analyze: ''
  }
}))

const apiMocks = vi.hoisted(() => ({
  conversations: vi.fn(),
  conversationDetail: vi.fn(),
  chat: vi.fn(),
  analyzeReport: vi.fn(),
  knowledgeSources: vi.fn(),
  reindexKnowledge: vi.fn()
}))

const elementMessage = vi.hoisted(() => ({
  error: vi.fn(),
  success: vi.fn()
}))

vi.mock('element-plus', () => ({
  ElMessage: elementMessage
}))

vi.mock('vue-router', async () => {
  const { reactive } = await import('vue')
  const route = reactive(routeState)
  return {
    useRoute: () => route
  }
})

vi.mock('@/shared/services', () => ({
  aiApi: apiMocks
}))

import { useRoute } from 'vue-router'
import { useAIAssistantWorkspace } from './composables/useAIAssistantWorkspace'

const Harness = defineComponent({
  setup() {
    return useAIAssistantWorkspace()
  },
  template: '<div />'
})

describe('useAIAssistantWorkspace regression', () => {
  beforeEach(() => {
    routeState.query.scene = 'report_analysis'
    routeState.query.report_id = '42'
    routeState.query.auto_analyze = ''
    Object.values(apiMocks).forEach((mock) => mock.mockReset())
    elementMessage.error.mockReset()
    elementMessage.success.mockReset()

    apiMocks.conversations.mockResolvedValue({ data: [] })
    apiMocks.knowledgeSources.mockResolvedValue({
      data: { total_chunks: 6, retrieval_mode: 'hybrid' }
    })
  })

  it('clears scene and bound report when starting a fresh conversation', async () => {
    const wrapper = mount(Harness)
    await flushPromises()

    expect(wrapper.vm.scene).toBe('report_analysis')
    expect(wrapper.vm.reportIdInput).toBe('42')

    wrapper.vm.startFreshConversation()
    await nextTick()

    expect(wrapper.vm.currentConversationId).toBeNull()
    expect(wrapper.vm.messages).toEqual([])
    expect(wrapper.vm.scene).toBe('general')
    expect(wrapper.vm.reportIdInput).toBe('')
  })

  it('drops stale route-bound context when route query is cleared', async () => {
    const wrapper = mount(Harness)
    await flushPromises()

    const route = useRoute()
    route.query.scene = ''
    route.query.report_id = ''
    route.query.auto_analyze = ''
    await nextTick()

    expect(wrapper.vm.scene).toBe('general')
    expect(wrapper.vm.reportIdInput).toBe('')
  })

  it('auto-seeds report analysis once when routed from a report entry', async () => {
    routeState.query.auto_analyze = '1'
    apiMocks.analyzeReport.mockResolvedValue({
      data: {
        conversation: {
          id: 7,
          scene: 'report_analysis',
          report_id: 42,
          meta: {
            context_summary: {
              scene_label: '报告分析',
              report_id: 42
            }
          }
        },
        message: {
          id: 'm-ai',
          role: 'assistant',
          content: '直接结论：当前报告为高风险。',
          citations: [],
          used_context: { report_id: 42, fallback_used: true }
        },
        citations: [],
        used_context: { report_id: 42, fallback_used: true }
      }
    })

    const wrapper = mount(Harness)
    await flushPromises()

    expect(apiMocks.analyzeReport).toHaveBeenCalledTimes(1)
    expect(apiMocks.analyzeReport).toHaveBeenCalledWith({
      report_id: '42',
      conversation_id: undefined
    })
    expect(wrapper.vm.currentConversationId).toBe(7)
    expect(wrapper.vm.messages).toHaveLength(2)

    const route = useRoute()
    route.query.auto_analyze = '1'
    await nextTick()
    await flushPromises()

    expect(apiMocks.analyzeReport).toHaveBeenCalledTimes(1)
  })

  it('keeps report-analysis scene binding for AI interpretation entry', async () => {
    const wrapper = mount(Harness)
    await flushPromises()

    expect(wrapper.vm.scene).toBe('report_analysis')
    expect(wrapper.vm.reportIdInput).toBe('42')
    expect(routeState.query.scene).toBe('report_analysis')
    expect(routeState.query.report_id).toBe('42')
  })
})
