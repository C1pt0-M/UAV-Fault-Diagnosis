import request from '@/utils/request'

export const analysisApi = {
  fields(assetId) {
    return request.get('/analysis/fields', { params: { asset_id: assetId } })
  },
  timeseries(params) {
    return request.get('/analysis/timeseries', {
      params: {
        asset_id: params.assetId,
        fields: params.fields.join(','),
        start_t: params.startT,
        end_t: params.endT,
        max_points: params.maxPoints
      }
    })
  },
  faultProbCurve(assetId) {
    return request.get('/analysis/fault_prob_curve', {
      params: { asset_id: assetId }
    })
  }
}
