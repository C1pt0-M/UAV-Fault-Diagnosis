import request from '@/utils/request'

export const aiApi = {
  conversations(params = {}) {
    return request.get('/ai/conversations', { params })
  },
  conversationDetail(conversationId) {
    return request.get(`/ai/conversations/${conversationId}`)
  },
  chat(payload) {
    return request.post('/ai/chat', payload)
  },
  analyzeReport(payload) {
    return request.post('/ai/analyze-report', payload)
  },
  knowledgeSources() {
    return request.get('/ai/knowledge/sources')
  },
  reindexKnowledge() {
    return request.post('/ai/knowledge/reindex', {})
  }
}
