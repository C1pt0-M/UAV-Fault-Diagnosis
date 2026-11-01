import { flushPromises, mount } from '@vue/test-utils'
import { defineComponent, nextTick } from 'vue'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

const routerPush = vi.hoisted(() => vi.fn())
const telemetryApiMock = vi.hoisted(() => ({
  realtimeData: vi.fn(),
  rflyDiagnostics: vi.fn(),
  sessions: vi.fn(),
  startCollection: vi.fn(),
  startBlindTest: vi.fn(),
  testRflyDiagnostics: vi.fn(),
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
  await flushPromises()
}

describe('useMonitoringDashboard fault handoff dialog', () => {
  beforeEach(() => {
    routerPush.mockClear()
    telemetryApiMock.status.mockResolvedValue({ data: {
      telemetry_capabilities: { enabled: true, allowed_backends: ['replay', 'rfly'] }
    } })
    telemetryApiMock.sessions.mockResolvedValue({ data: { items: [] } })
    telemetryApiMock.realtimeData.mockResolvedValue({ data: { times: [] } })
    telemetryApiMock.rflyDiagnostics.mockResolvedValue({ data: null })
    telemetryApiMock.startCollection.mockResolvedValue({
      data: {
        session_id: 'TS-REPLAY-001',
        scenario_backend: 'replay'
      }
    })
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

  it.each([{}, { enabled: true, allowed_backends: [] }, { enabled: true, allowed_backends: ['replay'] }])('does not probe or start forbidden RflySim for %j', async (capabilities) => {
    telemetryApiMock.status.mockResolvedValue({ data: { telemetry_capabilities: capabilities } })
    const wrapper = mount(Harness)
    await flushMountedWork()
    wrapper.vm.selectedBackend = 'rfly'
    await flushMountedWork()
    await wrapper.vm.testRflyDiagnostics()
    await wrapper.vm.startCollection()
    expect(telemetryApiMock.rflyDiagnostics).not.toHaveBeenCalled()
    expect(telemetryApiMock.testRflyDiagnostics).not.toHaveBeenCalled()
    expect(telemetryApiMock.startCollection).not.toHaveBeenCalled()
    wrapper.unmount()
  })

  it('clears old backend declarations when a later status omits capabilities', async () => {
    const wrapper = mount(Harness)
    await flushMountedWork()
    expect(wrapper.vm.backendOptions).toHaveLength(2)
    telemetryApiMock.status.mockResolvedValue({ data: {} })
    await wrapper.vm.stopCollection()
    expect(wrapper.vm.backendOptions).toEqual([])
    expect(wrapper.vm.selectedBackend).toBe('')
    wrapper.unmount()
  })

  it('refuses blind-test requests from RflySim even if PX4 blind tests are enabled', async () => {
    telemetryApiMock.status.mockResolvedValue({ data: { telemetry_capabilities: {
      enabled: true, allowed_backends: ['rfly', 'replay', 'px4'], blind_test_enabled: true
    } } })
    const wrapper = mount(Harness)
    await flushMountedWork()
    wrapper.vm.selectedBackend = 'rfly'
    await wrapper.vm.startBlindTest()
    expect(telemetryApiMock.startBlindTest).not.toHaveBeenCalled()
    wrapper.unmount()
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

  it('passes replay asset id when starting replay collection', async () => {
    assetApiMock.list.mockResolvedValueOnce({
      data: [
        {
          id: 265,
          filename: 'flight.csv',
          kind: 'csv',
          source_category: 'csv',
          meta: { prepared_csv_path: 'runtime://uploads/csv/flight.csv' }
        }
      ]
    })

    const wrapper = mount(Harness)
    await flushMountedWork()

    wrapper.vm.selectedReplayAssetId = 265
    await nextTick()
    await wrapper.vm.startCollection()
    await flushMountedWork()

    expect(telemetryApiMock.startCollection).toHaveBeenCalledWith(
      expect.objectContaining({
        backend: 'replay',
        replay_path: 'runtime://uploads/csv/flight.csv',
        replay_asset_id: 265
      })
    )
  })

  it('passes the default center inference location when starting RflySim collection', async () => {
    telemetryApiMock.rflyDiagnostics.mockResolvedValueOnce({
      data: {
        ready: true,
        summary: { status: 'ready', message: 'RflySim 链路配置可用' },
        checks: {},
        issues: []
      }
    })

    const wrapper = mount(Harness)
    await flushMountedWork()

    wrapper.vm.selectedBackend = 'rfly'
    await nextTick()
    await wrapper.vm.startCollection()
    await flushMountedWork()

    expect(telemetryApiMock.startCollection).toHaveBeenCalledWith(
      expect.objectContaining({
        backend: 'rfly',
        rfly_inference_location: 'center'
      })
    )
  })

  it('hides inference location metric when MEC is disabled in frontend', async () => {
    telemetryApiMock.status.mockResolvedValueOnce({
      data: {
        inference: {
          location: 'mec',
          status: 'active',
          avg_inference_ms: 8.4,
          p95_inference_ms: 12.8
        }
      }
    })

    const wrapper = mount(Harness)
    await flushMountedWork()

    wrapper.vm.selectedBackend = 'rfly'
    await nextTick()

    expect(wrapper.vm.metrics).not.toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          label: '推理位置'
        })
      ])
    )
  })

  it('surfaces blocked input contract as paused detection with zero-signal context', async () => {
    telemetryApiMock.status.mockResolvedValueOnce({
      data: {
        input_contract_status: {
          status: 'degraded',
          reason: 'untrusted_zero_signals',
          detection_blocked: true,
          actuator_source: 'px4',
          zero_signal_groups: ['actuator_outputs', 'imu_accel_z']
        }
      }
    })

    const wrapper = mount(Harness)
    await flushMountedWork()

    expect(wrapper.vm.inputContractLabel).toBe('检测暂停')
    expect(wrapper.vm.inputContractMeta).toContain('检测已暂停')
    expect(wrapper.vm.inputContractMeta).toContain('零信号：动力输出、Z 轴加速度')
    expect(wrapper.vm.metrics).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          label: '字段合同',
          value: '检测暂停',
          tone: 'danger'
        })
      ])
    )
  })

  it('shows Rfly NED position z as positive altitude in the height chart', async () => {
    telemetryApiMock.status.mockResolvedValueOnce({
      data: { scenario_backend: 'rfly' }
    })
    telemetryApiMock.realtimeData.mockResolvedValueOnce({
      data: { times: [0, 1, 2], pos_z: [0, -5, -10] }
    })

    const wrapper = mount(Harness)
    await flushMountedWork()

    expect(wrapper.vm.realtimePositionSeries).toEqual([
      { name: 'Altitude', data: [0, 5, 10] }
    ])
  })

})
