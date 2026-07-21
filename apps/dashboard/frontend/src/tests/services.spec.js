import { beforeEach, describe, expect, it, vi } from 'vitest'

const mocks = vi.hoisted(() => ({
  get: vi.fn(),
  post: vi.fn(),
  put: vi.fn()
}))

vi.mock('@/utils/request', () => ({
  default: {
    get: mocks.get,
    post: mocks.post,
    put: mocks.put
  }
}))

import { runApi, settingsApi, telemetryApi } from '@/shared/services'

describe('shared services', () => {
  beforeEach(() => {
    mocks.get.mockReset()
    mocks.post.mockReset()
    mocks.put.mockReset()
  })

  it('通过共享 service 暴露离线诊断任务创建', () => {
    runApi.createOfflineDetection({ sessionId: 'TS-002', model: 'offline_multiclass_base' })

    expect(mocks.post).toHaveBeenCalledWith('/runs', {
      task: 'offline_multiclass',
      asset_id: undefined,
      session_id: 'TS-002',
      model: 'offline_multiclass_base'
    })
  })

  it('通过共享 service 暴露监测会话详情查询', () => {
    telemetryApi.sessionDetail('TS-002')

    expect(mocks.get).toHaveBeenCalledWith('/telemetry/sessions/TS-002')
  })
  it('exposes backend persisted AI settings endpoints', () => {
    settingsApi.getAiSettings()
    settingsApi.updateAiSettings({ base_url: 'https://api.example/v1' })
    settingsApi.testAiSettings({ api_key: 'secret' })

    expect(mocks.get).toHaveBeenCalledWith('/settings/ai')
    expect(mocks.put).toHaveBeenCalledWith('/settings/ai', {
      base_url: 'https://api.example/v1'
    })
    expect(mocks.post).toHaveBeenCalledWith('/settings/ai/test', {
      api_key: 'secret'
    })
  })

  it('exposes RflySim diagnostics endpoints', () => {
    telemetryApi.rflyDiagnostics()
    telemetryApi.testRflyDiagnostics({ rfly_px4mavctrl_path: 'D:/PX4MavCtrlV4.py' })

    expect(mocks.get).toHaveBeenCalledWith('/telemetry/rfly/diagnostics')
    expect(mocks.post).toHaveBeenCalledWith('/telemetry/rfly/diagnostics', {
      rfly_px4mavctrl_path: 'D:/PX4MavCtrlV4.py'
    })
  })

  it('exposes the 5G realtime session comparison endpoint', () => {
    telemetryApi.networkComparison()

    expect(mocks.get).toHaveBeenCalledWith('/telemetry/network-comparison')
  })
})
