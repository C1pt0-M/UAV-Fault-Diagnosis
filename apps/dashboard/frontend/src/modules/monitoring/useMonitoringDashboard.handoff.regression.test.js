import { mount } from '@vue/test-utils'
import { defineComponent, nextTick } from 'vue'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { readFileSync } from 'node:fs'

const routerPush = vi.hoisted(() => vi.fn())
const telemetryApiMock = vi.hoisted(() => ({
  realtimeData: vi.fn(),
  networkComparison: vi.fn(),
  rflyDiagnostics: vi.fn(),
  sessions: vi.fn(),
  startCollection: vi.fn(),
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
    telemetryApiMock.networkComparison.mockResolvedValue({
      data: { profiles: [], complete: false }
    })
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

  it('passes default 5G profile when starting RflySim collection', async () => {
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
        rfly_network_profile: 'enhanced_5g'
      })
    )
  })

  it('surfaces 5G link metrics from telemetry status', async () => {
    telemetryApiMock.status.mockResolvedValueOnce({
      data: {
        network_link: {
          enabled: true,
          profile: 'degraded',
          avg_latency_ms: 82.4,
          p95_latency_ms: 123.8,
          loss_rate: 0.03,
          data_integrity: 0.97
        }
      }
    })

    const wrapper = mount(Harness)
    await flushMountedWork()

    expect(wrapper.vm.metrics).not.toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          label: '5G链路场景'
        })
      ])
    )

    wrapper.vm.selectedBackend = 'rfly'
    await nextTick()

    expect(wrapper.vm.metrics).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          label: '5G链路场景',
          value: '弱网链路 degraded',
          meta: '均值 82.4 ms / P95 124 ms',
          tone: 'danger'
        }),
        expect.objectContaining({
          label: '链路完整率',
          value: '97.0%',
          meta: '丢包 3.0%',
          tone: 'danger'
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

  it('keeps visible monitoring copy free of common mojibake fragments', () => {
    const source = readFileSync(
      'src/modules/monitoring/composables/useMonitoringDashboard.js',
      'utf8'
    )

    expect(source).not.toMatch(/鍚|璇|锛|鐩|瀛|妫/)
  })
})
