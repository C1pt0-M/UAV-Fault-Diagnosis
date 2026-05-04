import request from '@/utils/request'
import { downloadBlob } from '@/shared/services/download'

export const assetApi = {
  upload(file) {
    const formData = new FormData()
    formData.append('file', file)
    return request.postForm('/assets/upload', formData)
  },
  list(kind = 'all') {
    return request.get('/assets', { params: { kind } })
  },
  detail(assetId) {
    return request.get(`/assets/${assetId}`)
  },
  remove(assetId) {
    return request.delete(`/assets/${assetId}`)
  },
  prepare(assetId) {
    return request.post(`/assets/${assetId}/prepare`, {})
  },
  async download(assetId, filename) {
    const blob = await request.getBlob(`/assets/${assetId}/download`)
    await downloadBlob(blob, filename || `asset_${assetId}`)
  }
}
