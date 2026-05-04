import request from '@/utils/request'

export const runApi = {
  createOfflineDetection(payload) {
    return request.post('/runs', {
      task: 'offline_multiclass',
      asset_id: payload.assetId,
      session_id: payload.sessionId,
      model: payload.model
    })
  },
  createOffline5Class(payload) {
    return this.createOfflineDetection(payload)
  },
  list(params = {}) {
    return request.get('/runs', { params })
  },
  detail(runId) {
    return request.get(`/runs/${runId}`)
  },
  result(runId) {
    return request.get(`/runs/${runId}/result`)
  }
}
