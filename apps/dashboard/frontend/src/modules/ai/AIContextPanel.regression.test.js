import { mount } from '@vue/test-utils'
import { describe, expect, it } from 'vitest'

import AIContextPanel from './components/AIContextPanel.vue'

describe('AIContextPanel regression', () => {
  it('shows knowledge-base embedding status and retrieval diagnostics', () => {
    const wrapper = mount(AIContextPanel, {
      props: {
        contextSummary: {
          conversationId: 7,
          sceneLabel: '报告分析',
          reportId: 42,
          runId: 15,
          model: 'mock-model',
          knowledgeHits: 3,
          retrievalMode: 'bm25',
          fallbackUsed: true,
          knowledgeEmbeddingEnabled: false
        },
        knowledge: {
          total_chunks: 24,
          total_documents: 5,
          retrieval_mode: 'bm25',
          requested_retrieval_mode: 'hybrid',
          embedding_enabled: false,
          embedding_model: 'text-embedding-3-large',
          embedding_error: 'embedding provider timeout'
        },
        reindexing: false,
        scene: 'report_analysis',
        reportIdInput: '42'
      }
    })

    const text = wrapper.text()
    expect(text).toContain('知识库状态')
    expect(text).toContain('Embedding 状态')
    expect(text).toContain('异常')
    expect(text).toContain('embedding provider timeout')
    expect(text).toContain('请求检索方式')
    expect(text).not.toContain('回答来源')
  })
})
