import request from '@/utils/request'

export const telemetryApi = {
  startCollection(payload) {
    return request.post('/telemetry/start-collection', payload)
  },
  stopCollection() {
    return request.post('/telemetry/stop-collection')
  },
  startDetection(payload = {}) {
    return request.post('/telemetry/start-detection', payload)
  },
  stopDetection() {
    return request.post('/telemetry/stop-detection')
  },
  status() {
    return request.get('/telemetry/status')
  },
  rflyDiagnostics() {
    return request.get('/telemetry/rfly/diagnostics')
  },
  testRflyDiagnostics(payload = {}) {
    return request.post('/telemetry/rfly/diagnostics', payload)
  },
  realtimeData() {
    return request.get('/telemetry/realtime-data')
  },
  networkComparison() {
    return request.get('/telemetry/network-comparison')
  },
  sessions(params = {}) {
    return request.get('/telemetry/sessions', { params })
  },
  sessionDetail(sessionId) {
    return request.get(`/telemetry/sessions/${sessionId}`)
  },
  startBlindTest(payload = {}) {
    return request.post('/telemetry/blind-test/start', payload)
  },
  revealBlindTest() {
    return request.get('/telemetry/blind-test/reveal')
  }
}
