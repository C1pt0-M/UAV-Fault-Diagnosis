import { mount } from '@vue/test-utils'
import { defineComponent, nextTick } from 'vue'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

const routeQuery = vi.hoisted(() => ({ session_id: 'TS-262735', auto: '1' }))
const routerReplace = vi.hoisted(() => vi.fn())
const routerPush = vi.hoisted(() => vi.fn())
const assetApiMock = vi.hoisted(() => ({ list: vi.fn() }))
const modelApiMock = vi.hoisted(() => ({ getModels: vi.fn() }))
const runApiMock = vi.hoisted(() => ({
  createOfflineDetection: vi.fn(),
  detail: vi.fn(),
  list: vi.fn(),
  result: vi.fn()
}))
const reportApiMock = vi.hoisted(() => ({
  create: vi.fn(),
  list: vi.fn()
}))
const telemetryApiMock = vi.hoisted(() => ({ sessionDetail: vi.fn() }))

vi.mock('vue-router', () => ({
  useRoute: () => ({ query: routeQuery }),
  useRouter: () => ({ push: routerPush, replace: routerReplace })
}))

vi.mock('element-plus', () => ({
  ElMessage: {
    error: vi.fn(),
    success: vi.fn(),
    warning: vi.fn()
  }
}))

vi.mock('@/shared/services', () => ({
  assetApi: assetApiMock,
  diagnosisReportApi: reportApiMock,
  modelApi: modelApiMock,
  runApi: runApiMock,
  telemetryApi: telemetryApiMock
}))

import { useDiagnosisWorkspace } from './composables/useDiagnosisWorkspace'
import { applyRuntimeCapabilities } from '@/shared/state/runtimeCapabilities'

const Harness = defineComponent({
  setup() {
    return useDiagnosisWorkspace()
  },
  template: '<div />'
})

async function flushMountedWork() {
  await nextTick()
  await Promise.resolve()
  await Promise.resolve()
  await Promise.resolve()
}

describe('useDiagnosisWorkspace monitoring session handoff', () => {
  beforeEach(() => {
    routerReplace.mockClear()
    routerPush.mockClear()
    applyRuntimeCapabilities({ ai: true, telemetry: true })
    assetApiMock.list.mockResolvedValue({ data: [] })
    modelApiMock.getModels.mockResolvedValue({
      data: [{ name: 'online_detector_kg_mstca_rfly' }]
    })
    runApiMock.list.mockResolvedValue({ data: [] })
    runApiMock.createOfflineDetection.mockResolvedValue({ data: { id: 42, status: 'running' } })
    reportApiMock.list.mockResolvedValue({ data: [] })
    telemetryApiMock.sessionDetail.mockResolvedValue({ data: { id: 'TS-262735' } })
  })

  afterEach(() => {
    vi.clearAllMocks()
  })

  it('does not request a disabled monitoring handoff and preserves asset diagnosis', async () => {
    applyRuntimeCapabilities({ telemetry: false, ai: false })
    const wrapper = mount(Harness)
    await flushMountedWork()
    expect(telemetryApiMock.sessionDetail).not.toHaveBeenCalled()
    expect(runApiMock.createOfflineDetection).not.toHaveBeenCalled()
    expect(wrapper.vm.activeSessionId).toBeFalsy()
    wrapper.vm.selectedAssetId = 7
    await wrapper.vm.startDiagnosis()
    expect(runApiMock.createOfflineDetection).toHaveBeenCalledWith(expect.objectContaining({ assetId: 7, sessionId: undefined }))
    wrapper.unmount()
  })

  it.each([false, true])('guards diagnosis report assistant handoff when ai=%s', async (enabled) => {
    applyRuntimeCapabilities({ telemetry: false, ai: enabled })
    const wrapper = mount(Harness)
    await flushMountedWork()
    wrapper.vm.openReportAssistant({ report_id: 88 })
    if (enabled) expect(routerPush).toHaveBeenCalledWith({ path: '/ai', query: { scene: 'report_analysis', report_id: '88', auto_analyze: '1' } })
    else expect(routerPush).not.toHaveBeenCalled()
    wrapper.unmount()
  })

  it('validates the session, consumes auto once, and starts offline diagnosis', async () => {
    const wrapper = mount(Harness)
    await flushMountedWork()

    expect(telemetryApiMock.sessionDetail).toHaveBeenCalledWith('TS-262735')
    expect(routerReplace).toHaveBeenCalledWith({
      path: '/diagnosis',
      query: { session_id: 'TS-262735' }
    })
    expect(runApiMock.createOfflineDetection).toHaveBeenCalledWith({
      assetId: null,
      sessionId: 'TS-262735',
      model: 'online_detector_kg_mstca_rfly'
    })
    expect(wrapper.vm.activeSessionId).toBe('TS-262735')

    wrapper.unmount()
  })
})
