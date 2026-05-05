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
  currentState: 'DETECTING',
  detailVisible: false,
  estimatedSampleRate: '25.0 Hz',
  loadSessions: vi.fn(),
  metrics: [
    { label: '当前状态', value: '检测中', meta: '会话 TS-262735', tone: 'active', icon: 'Monitor' },
    { label: '故障概率峰值', value: '0.83', meta: '累计告警 1', tone: 'danger', icon: 'Warning' }
  ],
  onlineModels: [],
  openSessionDetail: vi.fn(),
  pipelineItems: [
    { label: '遥测采集', value: '检测中', status: 'active' },
    { label: '在线检测模型', value: 'KG-MSTCA 在线检测模型', status: 'ok' }
  ],
  replayAssets: [],
  realtimeChartTimes: [],
  realtimeFaultProb: [],
  realtimeAttitudeSeries: [],
  realtimeMotorSeries: [],
  realtimePositionSeries: [],
  sampleSummaryItems: [
    { label: '采样点数量', value: 0 },
    { label: '估算采样率', value: '25.0 Hz' }
  ],
  selectedBackend: 'replay',
  selectedDuration: 60,
  selectedModel: 'online_detector_kg_mstca_rfly',
  selectedReplayAssetId: null,
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
  stopCollection: vi.fn(),
  stopDetection: vi.fn(),
  continueDiagnosis: vi.fn(),
  stateLabel: '检测中',
  statusOverviewItems: [
    { label: '连接状态', value: '已连接', status: 'ok' },
    { label: '估算采样率', value: '25.0 Hz', status: 'ok' }
  ],
  faultPeakText: '0.83',
  inputContractLabel: '字段正常',
  inputContractMeta: '动力字段：motor / healthy',
  selectedBackendLabel: 'CSV 回放',
  sessionBridgeText: '支持承接到离线诊断',
  rflyDiagnostics: null,
  rflyDiagnosticsLoading: false,
  rflyPx4mavctrlPath: ''
}))

vi.mock('@/modules/monitoring/composables/useMonitoringDashboard', () => ({
  useMonitoringDashboard: () => dashboardState
}))

import MonitoringPage from './MonitoringPage.vue'

describe('MonitoringPage telemetry cockpit', () => {
  it('renders the compact cockpit structure without unsupported telemetry labels', () => {
    const wrapper = mount(MonitoringPage, {
      global: {
        stubs: {
          MonitoringControlPanel: { template: '<section>监测控制台</section>' },
          MonitoringMetrics: { template: '<section>故障概率峰值</section>' },
          MonitoringRealtimeChartPanel: { template: '<section>故障概率趋势</section>' },
          MonitoringInsightsPanel: { template: '<section>实时状态概览检测链路状态最近告警事件</section>' },
          MonitoringSessionsPanel: { template: '<section>监测会话记录</section>' },
          MonitoringSessionDialog: { template: '<div />' }
        }
      }
    })

    const text = wrapper.text()
    expect(text).toContain('实时监测')
    expect(text).toContain('遥测采集、在线检测、实时告警与会话承接')
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

    const wrapper = mount(MonitoringPage, {
      global: {
        stubs: {
          MonitoringControlPanel: true,
          MonitoringMetrics: true,
          MonitoringRealtimeChartPanel: true,
          MonitoringInsightsPanel: true,
          MonitoringSessionsPanel: true,
          MonitoringSessionDialog: true
        }
      }
    })

    const modelValue = wrapper.find('[data-test="monitoring-status-model"]')
    expect(modelValue.attributes('title')).toBe('online_detector_kg_mstca_rfly_with_extra_long_runtime_name')
  })
})
