import { mount } from '@vue/test-utils'
import { describe, expect, it, vi } from 'vitest'

const dashboardState = vi.hoisted(() => ({
  actionLoading: {},
  activeModelLabel: 'KG-MSTCA 在线检测模型',
  alerts: [
    {
      id: 1,
      start_time: '10:22:14',
      end_time: '10:23:12',
      severity: 'high',
      diagnosis: { label: '电机电流异常', severity: 'high' }
    }
  ],
  backendOptions: [{ label: 'CSV 回放', value: 'replay' }],
  blindTestEnabled: false,
  confirmFaultHandoff: vi.fn(),
  continueDiagnosis: vi.fn(),
  currentSessionId: 'TS-262735',
  currentState: 'DETECTING',
  detailVisible: false,
  dismissFaultHandoff: vi.fn(),
  estimatedSampleRate: '25.0 Hz',
  faultHandoffDialogVisible: true,
  faultHandoffPayload: {
    session_id: 'TS-262735',
    fault_events_count: 2
  },
  faultPeakText: '0.83',
  inputContractLabel: '字段正常',
  inputContractMeta: '动力字段：motor / healthy',
  loadSessions: vi.fn(),
  metrics: [
    { label: '当前状态', value: '检测中', meta: '会话 TS-262735', tone: 'active', icon: 'Monitor' },
    { label: '故障概率峰值', value: '0.83', meta: '风险评分 0.14', tone: 'danger', icon: 'Warning' }
  ],
  onlineModels: [],
  openSessionDetail: vi.fn(),
  pipelineItems: [
    { label: '遥测采集', value: '采集中', status: 'active' },
    { label: '在线检测模型', value: 'KG-MSTCA 在线检测模型', status: 'ok' }
  ],
  replayAssets: [],
  realtimeChartTimes: [],
  realtimeFaultProb: [],
  realtimeAttitudeSeries: [],
  realtimeMotorSeries: [],
  realtimePositionSeries: [],
  rflyDiagnostics: null,
  rflyDiagnosticsLoading: false,
  rflyPx4mavctrlPath: '',
  sampleSummaryItems: [
    { label: '采样点数量', value: 0 },
    { label: '估算采样率', value: '25.0 Hz' }
  ],
  selectedBackend: 'replay',
  selectedBackendLabel: 'CSV 回放',
  selectedDuration: 60,
  selectedModel: 'online_detector_kg_mstca_rfly',
  selectedReplayAssetId: null,
  sessionBridgeText: '支持承接到离线诊断',
  sessionDetail: null,
  sessions: [
    {
      id: 'TS-262735',
      drone_id: 'DRONE-001',
      start_time: '2026-04-28 09:38:21',
      duration_seconds: 2775,
      fault_events_count: 1,
      status: 'running'
    }
  ],
  startBlindTest: vi.fn(),
  startCollection: vi.fn(),
  startDetection: vi.fn(),
  stateLabel: '检测中',
  statusOverviewItems: [
    { label: '连接状态', value: '已连接', status: 'ok' },
    { label: '估算采样率', value: '25.0 Hz', status: 'ok' }
  ],
  stopCollection: vi.fn(),
  stopDetection: vi.fn(),
  testRflyDiagnostics: vi.fn()
}))

vi.mock('@/modules/monitoring/composables/useMonitoringDashboard', () => ({
  useMonitoringDashboard: () => dashboardState
}))

import MonitoringPage from './MonitoringPage.vue'

function mountPage() {
  return mount(MonitoringPage, {
    global: {
      stubs: {
        ElButton: {
          emits: ['click'],
          template: '<button type="button" @click="$emit(\'click\')"><slot /></button>'
        },
        ElDialog: {
          props: ['modelValue'],
          template: '<section v-if="modelValue" class="el-dialog-stub"><slot name="header" /><slot /><footer><slot name="footer" /></footer></section>'
        },
        MonitoringControlPanel: { template: '<section>监测控制台</section>' },
        MonitoringMetrics: { template: '<section>故障概率峰值</section>' },
        MonitoringRealtimeChartPanel: { template: '<section>故障概率趋势</section>' },
        MonitoringInsightsPanel: { template: '<section>实时状态概览 检测链路状态 最近告警事件</section>' },
        MonitoringSessionsPanel: { template: '<section>监测会话记录</section>' },
        MonitoringSessionDialog: { template: '<div />' }
      }
    }
  })
}

describe('MonitoringPage telemetry cockpit', () => {
  it('renders the compact cockpit structure without unsupported telemetry labels', () => {
    const wrapper = mountPage()

    const text = wrapper.text()
    expect(text).toContain('实时监测')
    expect(text).toContain('统一管理遥测采集、在线检测、实时告警与会话承接')
    expect(text).toContain('当前状态')
    expect(text).toContain('当前会话')
    expect(text).toContain('在线模型')
    expect(text).toContain('字段合同')
    expect(text).toContain('KG-MSTCA 在线检测模型')
    expect(text).toContain('实时状态概览')
    expect(text).toContain('检测链路状态')
    expect(text).toContain('最近告警事件')
    expect(text).toContain('监测会话记录')
    expect(text).not.toContain('电池电压')
    expect(text).not.toContain('电机温度')
    expect(text).not.toContain('GPS')
  })

  it('keeps long model names traceable through a title attribute', () => {
    dashboardState.activeModelLabel = 'online_detector_kg_mstca_rfly_with_extra_long_runtime_name'

    const wrapper = mountPage()

    const modelValue = wrapper.find('[data-test="monitoring-status-model"]')
    expect(modelValue.attributes('title')).toBe('online_detector_kg_mstca_rfly_with_extra_long_runtime_name')
  })

  it('renders the fault handoff dialog and wires the actions', async () => {
    dashboardState.activeModelLabel = 'KG-MSTCA 在线检测模型'
    dashboardState.faultHandoffDialogVisible = true
    dashboardState.faultHandoffPayload = {
      session_id: 'TS-262735',
      fault_events_count: 2
    }
    dashboardState.confirmFaultHandoff.mockClear()
    dashboardState.dismissFaultHandoff.mockClear()

    const wrapper = mountPage()

    const text = wrapper.text()
    expect(text).toContain('检测到故障事件')
    expect(text).toContain('当前会话已确认 2 个故障事件')
    expect(text).toContain('TS-262735')
    expect(text).toContain('进入诊断中心')
    expect(text).toContain('稍后处理')

    const buttons = wrapper.findAll('button')
    await buttons.find((button) => button.text() === '稍后处理').trigger('click')
    await buttons.find((button) => button.text() === '进入诊断中心').trigger('click')

    expect(dashboardState.dismissFaultHandoff).toHaveBeenCalledTimes(1)
    expect(dashboardState.confirmFaultHandoff).toHaveBeenCalledTimes(1)
  })
})
