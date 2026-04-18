import { mount } from '@vue/test-utils'
import { describe, expect, it, vi } from 'vitest'

const dashboardState = vi.hoisted(() => ({
  actionLoading: {},
  alerts: [],
  backendOptions: [],
  blindTestEnabled: false,
  currentState: 'DETECTING',
  detailVisible: false,
  loadSessions: vi.fn(),
  metrics: [],
  onlineModels: [],
  openSessionDetail: vi.fn(),
  replayAssets: [],
  realtimeChartTimes: [],
  realtimeFaultProb: [],
  realtimeAttitudeSeries: [],
  realtimeMotorSeries: [],
  realtimePositionSeries: [],
  sampleSummaryItems: [],
  selectedBackend: 'replay',
  selectedDuration: 60,
  selectedModel: 'online_detector_cnn1d_rfly',
  selectedReplayAssetId: null,
  sessionDetail: null,
  sessions: [],
  startBlindTest: vi.fn(),
  startCollection: vi.fn(),
  startDetection: vi.fn(),
  stopCollection: vi.fn(),
  stopDetection: vi.fn(),
  continueDiagnosis: vi.fn(),
  stateLabel: '检测中',
  statusMeta: 'CSV 回放 / online_detector_cnn1d_rfly',
  faultPeakText: '0.83',
  sessionBridgeText: '支持承接到离线诊断'
}))

vi.mock('@/modules/monitoring/composables/useMonitoringDashboard', () => ({
  useMonitoringDashboard: () => dashboardState
}))

import MonitoringPage from './MonitoringPage.vue'

describe('MonitoringPage command deck redesign', () => {
  it('renders the monitoring command hero and continuation narrative', () => {
    const wrapper = mount(MonitoringPage, {
      global: {
        stubs: {
          MonitoringControlPanel: { template: '<section>监测控制台</section>' },
          MonitoringMetrics: { template: '<section>状态指标</section>' },
          MonitoringRealtimeChartPanel: { template: '<section>实时曲线主区</section>' },
          MonitoringInsightsPanel: { template: '<section>告警与样本摘要</section>' },
          MonitoringSessionsPanel: { template: '<section>监测会话承接区</section>' },
          MonitoringSessionDialog: { template: '<div />' }
        }
      }
    })

    const text = wrapper.text()
    expect(text).toContain('在线监测')
    expect(text).toContain('实时监测与在线预警工作台')
    expect(text).toContain('监测控制台')
    expect(text).toContain('实时曲线主区')
    expect(text).toContain('监测会话承接区')
  })
})
