import request from '@/utils/request'

export const settingsApi = {
  getAiSettings() {
    return request.get('/settings/ai')
  },
  updateAiSettings(payload) {
    return request.put('/settings/ai', payload)
  },
  testAiSettings(payload) {
    return request.post('/settings/ai/test', payload)
  },
  testEmbeddingSettings(payload) {
    return request.post('/settings/ai/test-embedding', payload)
  }
}
