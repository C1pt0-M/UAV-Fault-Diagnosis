import request from '@/utils/request'

export const modelApi = {
  getModels: (params = {}) => request.get('/models', { params })
}
