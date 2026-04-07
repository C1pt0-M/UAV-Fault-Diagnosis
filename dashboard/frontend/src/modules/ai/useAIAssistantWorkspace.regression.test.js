import { defineComponent, nextTick } from 'vue'
import { flushPromises, mount } from '@vue/test-utils'
import { beforeEach, describe, expect, it, vi } from 'vitest'

const routeState = vi.hoisted(() => ({
  query: {
    scene: 'report_analysis',
    report_id: '42'
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
    await nextTick()

    expect(wrapper.vm.scene).toBe('general')
    expect(wrapper.vm.reportIdInput).toBe('')
  })
})
