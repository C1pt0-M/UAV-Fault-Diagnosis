import request from '@/utils/request'
import { downloadBlob } from '@/shared/services/download'

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
  },
  async download(reportId, filename) {
    const blob = await request.getBlob(`/reports/diagnosis/${reportId}/pdf`)
    await downloadBlob(blob, filename || `diagnosis_report_${reportId}.pdf`)
  }
}
