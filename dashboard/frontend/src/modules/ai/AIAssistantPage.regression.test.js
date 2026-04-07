import { mount } from '@vue/test-utils'
import { describe, expect, it, vi } from 'vitest'

const workspaceState = vi.hoisted(() => ({
  canSeedAnalysis: true,
  composer: '',
  contextSummary: {
    conversationId: 12,
    sceneLabel: '报告分析',
    reportId: 42,
    runId: 7,
    provider: 'mock-provider',
    model: 'mock-model',
    knowledgeHits: 3,
    retrievalMode: 'hybrid',
    fallbackUsed: false
  },
  conversations: [
    {
      id: 12,
      title: '报告 #42 风险复核',
      updated_at: '2026-04-07 14:08:00',
      scene: 'report_analysis',
      report_id: 42,
      meta: {
        last_message_preview: '请解释当前报告中的高风险结论。',
        message_count: 4,
        context_summary: {
          scene_label: '报告分析',
          report_id: 42,
          run_id: 7
        }
      }
    }
  ],
  currentConversationId: 12,
  knowledge: {
    total_chunks: 24,
    total_documents: 6,
    retrieval_mode: 'hybrid',
    requested_retrieval_mode: 'hybrid',
    embedding_enabled: true,
    embedding_model: 'text-embedding-3-large',
    embedding_error: null
  },
  loadingConversations: false,
  loadingMessages: false,
  messages: [
    {
      id: 'm1',
      role: 'assistant',
      content: '当前报告显示电机故障风险较高，建议优先检查电机与 ESC。',
      citations: [{ source_type: 'report', source_id: '42', title: '诊断报告 #42', snippet: '电机故障，置信度 92.4%' }],
      used_context: {}
    }
  ],
  openConversation: vi.fn(),
  reportIdInput: '42',
  reindexKnowledge: vi.fn(),
  reindexingKnowledge: false,
  scene: 'report_analysis',
  seedReportAnalysis: vi.fn(),
  sendMessage: vi.fn(),
  sending: false,
  startFreshConversation: vi.fn()
}))

vi.mock('@/modules/ai/composables/useAIAssistantWorkspace', () => ({
  useAIAssistantWorkspace: () => workspaceState
}))

import AIAssistantPage from './AIAssistantPage.vue'

describe('AIAssistantPage information architecture', () => {
  it('renders the AI workspace as an analysis assistant instead of a debug panel', () => {
    const wrapper = mount(AIAssistantPage)
    const text = wrapper.text()

    expect(text).toContain('分析入口')
    expect(text).toContain('会话轨迹')
    expect(text).toContain('分析主区')
    expect(text).toContain('上下文与知识库')
    expect(text).toContain('报告 #42 风险复核')
  })
})
