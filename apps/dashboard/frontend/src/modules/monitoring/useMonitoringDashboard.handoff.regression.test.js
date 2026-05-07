import { mount } from '@vue/test-utils'
import { defineComponent, nextTick } from 'vue'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { readFileSync } from 'node:fs'

const routerPush = vi.hoisted(() => vi.fn())
const telemetryApiMock = vi.hoisted(() => ({
  realtimeData: vi.fn(),
  rflyDiagnostics: vi.fn(),
  sessions: vi.fn(),
  status: vi.fn(),
  stopCollection: vi.fn()
}))
const assetApiMock = vi.hoisted(() => ({
  list: vi.fn()
}))
const modelApiMock = vi.hoisted(() => ({
  getModels: vi.fn()
}))

vi.mock('vue-router', () => ({
  useRouter: () => ({ push: routerPush })
}))

vi.mock('element-plus', () => ({
  ElMessage: {
    error: vi.fn(),
    success: vi.fn(),
    warning: vi.fn()
  },
  ElMessageBox: {
    confirm: vi.fn()
  }
}))

vi.mock('@/shared/services', () => ({
  assetApi: assetApiMock,
  modelApi: modelApiMock,
  telemetryApi: telemetryApiMock
}))

import { useMonitoringDashboard } from './composables/useMonitoringDashboard'

const Harness = defineComponent({
  setup() {
    return useMonitoringDashboard()
  },
  template: '<div />'
})

async function flushMountedWork() {
  await nextTick()
  await Promise.resolve()
  await Promise.resolve()
}

describe('useMonitoringDashboard fault handoff dialog', () => {
  beforeEach(() => {
    routerPush.mockClear()
    telemetryApiMock.status.mockResolvedValue({ data: {} })
    telemetryApiMock.sessions.mockResolvedValue({ data: { items: [] } })
    telemetryApiMock.realtimeData.mockResolvedValue({ data: { times: [] } })
    telemetryApiMock.rflyDiagnostics.mockResolvedValue({ data: null })
    telemetryApiMock.stopCollection.mockResolvedValue({
      message: '采集已停止',
      data: {
        session_id: 'TS-262735',
        fault_events_count: 2
      }
    })
    assetApiMock.list.mockResolvedValue({ data: [] })
    modelApiMock.getModels.mockResolvedValue({ data: [] })
  })

  afterEach(() => {
    vi.clearAllMocks()
  })

  it('opens a custom handoff dialog when stopping collection returns fault events', async () => {
    const wrapper = mount(Harness)
    await flushMountedWork()

    await wrapper.vm.stopCollection()
    await flushMountedWork()

    expect(wrapper.vm.faultHandoffDialogVisible).toBe(true)
    expect(wrapper.vm.faultHandoffPayload).toMatchObject({
      session_id: 'TS-262735',
      fault_events_count: 2
    })
  })

  it('routes to diagnosis with session_id and auto flag when confirmed', async () => {
    const wrapper = mount(Harness)
    await flushMountedWork()

    await wrapper.vm.stopCollection()
    wrapper.vm.confirmFaultHandoff()

    expect(routerPush).toHaveBeenCalledWith({
      path: '/diagnosis',
      query: {
        session_id: 'TS-262735',
        auto: '1'
      }
    })
    expect(wrapper.vm.faultHandoffDialogVisible).toBe(false)
  })

  it('does not open the handoff dialog when there are no fault events', async () => {
    telemetryApiMock.stopCollection.mockResolvedValueOnce({
      message: '采集已停止',
      data: {
        session_id: 'TS-262735',
        fault_events_count: 0
      }
    })

    const wrapper = mount(Harness)
    await flushMountedWork()

    await wrapper.vm.stopCollection()
    await flushMountedWork()

    expect(wrapper.vm.faultHandoffDialogVisible).toBe(false)
  })

  it('keeps visible monitoring copy free of common mojibake fragments', () => {
    const source = readFileSync(
      'src/modules/monitoring/composables/useMonitoringDashboard.js',
      'utf8'
    )

    expect(source).not.toMatch(/鍚|璇|锛|鐩|瀛|妫/)
  })
})
