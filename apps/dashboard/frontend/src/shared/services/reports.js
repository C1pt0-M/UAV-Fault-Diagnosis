import request from '@/utils/request'

export const diagnosisReportApi = {
  create(runId) {
    return request.post('/reports/diagnosis', { run_id: runId })
  },
  list(params = {}) {
    return request.get('/reports/diagnosis', { params })
  },
  detail(reportId) {
    return request.get(`/reports/diagnosis/${reportId}`)
  },
  remove(reportId) {
    return request.delete(`/reports/diagnosis/${reportId}`)
  }
}
