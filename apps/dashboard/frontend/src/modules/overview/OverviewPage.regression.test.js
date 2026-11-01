import { flushPromises, mount } from '@vue/test-utils'
import { beforeEach, describe, expect, it, vi } from 'vitest'

const getOverviewMock = vi.hoisted(() => vi.fn())
const messageErrorMock = vi.hoisted(() => vi.fn())

vi.mock('@/shared/services', () => ({
  overviewApi: {
    getOverview: getOverviewMock
  }
}))

vi.mock('element-plus', () => ({
  ElMessage: {
    error: messageErrorMock
  },
  ElTag: { name: 'ElTag', template: '<span><slot /></span>' },
  ElTable: { name: 'ElTable', props: ['data'], template: '<div><slot /></div>' },
  ElTableColumn: { name: 'ElTableColumn', template: '<div><slot /></div>' },
  ElProgress: { name: 'ElProgress', template: '<div />' }
}))

vi.mock('@element-plus/icons-vue', () => ({
  Bell: { name: 'Bell', template: '<i data-icon="bell" />' },
  Box: { name: 'Box', template: '<i data-icon="box" />' },
  CircleCheck: { name: 'CircleCheck', template: '<i data-icon="circle-check" />' },
  Document: { name: 'Document', template: '<i data-icon="document" />' },
  Finished: { name: 'Finished', template: '<i data-icon="finished" />' },
  RefreshRight: { name: 'RefreshRight', template: '<i data-icon="refresh-right" />' },
  TrendCharts: { name: 'TrendCharts', template: '<i data-icon="trend-charts" />' },
  Warning: { name: 'Warning', template: '<i data-icon="warning" />' }
}))

vi.mock('echarts', () => ({
  init: vi.fn(() => ({
    setOption: vi.fn(),
    resize: vi.fn(),
    dispose: vi.fn()
  }))
}))

import OverviewPage from './OverviewPage.vue'
import { applyRuntimeCapabilities } from '@/shared/state/runtimeCapabilities'

describe('OverviewPage industrial research dashboard', () => {
  beforeEach(() => {
    applyRuntimeCapabilities({ telemetry: true, ai: true })
    getOverviewMock.mockReset()
    messageErrorMock.mockReset()
    getOverviewMock.mockResolvedValue({
      data: {
        assets_total: 128,
        asset_summary: {
          total: 128,
          csv_total: 76,
          ulg_total: 52,
          diagnosable_total: 102
        },
        runs_total: 42,
        runs_by_status: {
          queued: 2,
          running: 1,
          success: 36,
          error: 3
        },
        task_summary: {
          completed_total: 36,
          today_diagnosis_total: 6,
          success_total: 36,
          failed_total: 3,
          recent: [
            {
              id: 42,
              task: 'offline_multiclass',
              status: 'success',
              asset_id: 7,
              result: {
                prediction: { label: 'motor', confidence: 0.92 }
              },
              created_at: '2026-05-05 10:22:10',
              updated_at: '2026-05-05 10:24:16'
            }
          ]
        },
        fault_distribution: { normal: 21, motor: 13, sensor: 8 },
        recent_runs: [],
        warning_summary: {
          confirmed_total: 8,
          severity_counts: { high: 2, medium: 4, low: 2 },
          recent: [
            {
              id: 'FE-1',
              time: '10:22:14',
              asset: 'UAV-1023',
              message: 'fault_probability',
              severity: 'high'
            }
          ]
        },
        fault_summary: {
          dominant_label: 'motor',
          dominant_label_text: '电机故障',
          dominant_ratio: 44,
          avg_confidence: 0.86,
          report_count: 20
        },
        risk_trend: {
          points: [],
          threshold: 0.7
        },
        model_summary: {
          name: 'online_detector_kg_mstca_rfly',
          architecture: 'KG-MSTCA',
          task: 'online_binary',
          metrics: { macro_f1: 0.87, weighted_f1: 0.89, accuracy: 0.88 },
          capabilities: {
            offline_detection: false,
            realtime_detection: true,
            fault_prob_curve: true
          },
          feature_count: 48,
          window_size: 256,
          updated_at_text: '2026-05-04'
        },
        field_contract: {
          status: 'passed',
          schema_version: 'v1.4',
          input_integrity: 96,
          missing_fields: 3,
          feature_coverage: 89,
          source: '最近一次输入'
        },
        recent_evidence: [
          { name: 'fault_probability', value: '+0.32', status: 'observed' },
          { name: 'warning_stage', value: 'critical', status: 'observed' },
          { name: 'lead_time_s', value: '12.8', status: 'observed' }
        ],
        similar_reports: [
          {
            id: 3,
            asset: 'Asset #178',
            fault_label: '电机故障',
            confidence: 0.89,
            created_at: '2026-05-04 12:00:00'
          }
        ],
        telemetry_summary: {
          current_session_id: '',
          latest_session_id: 'TS-UNIT',
          business_status: 'idle',
          backend_status: '正常',
          last_sync: '10:24:36'
        }
      }
    })
  })

  it('renders the industrial research dashboard with traceable aggregates', async () => {
    const wrapper = mount(OverviewPage)
    await flushPromises()

    const text = wrapper.text()
    expect(text).toContain('系统在线')
    expect(text).toContain('最近会话')
    expect(text).toContain('日志资产总数')
    expect(text).toContain('可诊断资产数')
    expect(text).toContain('已完成任务数')
    expect(text).toContain('确认告警数')
    expect(text).toContain('故障分类概览')
    expect(text).toContain('平均诊断置信度')
    expect(text).toContain('近 24h 诊断风险趋势')
    expect(text).not.toContain('字段合同状态')
    expect(text).not.toContain('模型注册状态')
    expect(text).not.toContain('schema version')
    expect(text).not.toContain('诊断流程与任务状态')
    expect(text).not.toContain('离线分析流程')
    expect(text).toContain('当前检测模型')
    expect(text).not.toContain('默认在线模型')
    expect(text).not.toContain('诊断可解释性')
    expect(text).toContain('相似历史报告')
    expect(text).toContain('近 24h 无诊断报告或告警事件')
    expect(text).toContain('生成新诊断报告或启动实时检测后自动形成趋势')
    expect(text).toContain('统计基于日志资产、诊断任务、诊断报告与 telemetry fault events 聚合。')
    expect(text).toContain('128')
    expect(text).toContain('102')
    expect(text).toContain('36')
    expect(text).toContain('电机故障')
    expect(text).toContain('0.86')
    expect(text).toContain('KG-MSTCA 在线检测模型')
    expect(text).toContain('online_detector_kg_mstca_rfly')
    expect(text).toContain('故障概率异常')
    expect(text).not.toContain('告警阶段')
    expect(text).not.toContain('预测提前量')
    expect(text).not.toContain('异常信号')
    expect(text).not.toContain('风险/告警')
    expect(text).not.toContain('时间窗口')
    expect(text).toContain('实时检测')
    expect(text).toContain('故障概率曲线')
    expect(text).toContain('资产 #178')
    expect(text).not.toContain('fault_probability')
    expect(text).not.toContain('warning_stage')
    expect(text).not.toContain('realtime detection')
    expect(text).not.toContain('fault probability curve')
    expect(text).not.toContain('Asset #178')
    expect(wrapper.find('a.kpi-card[href="/assets"]').exists()).toBe(true)
    expect(wrapper.find('a.kpi-card[href="/diagnosis"]').exists()).toBe(true)
    expect(wrapper.find('a.kpi-card[href="/monitoring"]').exists()).toBe(true)
    expect(wrapper.find('a.kpi-card[href="/reports"]').exists()).toBe(true)
    expect(wrapper.find('a.panel-action[href="/monitoring"]').text()).toBe('查看监测')
    expect(wrapper.find('a.panel-action[href="/diagnosis"]').text()).toBe('进入诊断')
    expect(wrapper.find('a.panel-action[href="/reports"]').text()).toBe('查看报告')
    expect(wrapper.find('[data-icon="document"]').exists()).toBe(true)
  })

  it('retains historical warnings without linking to disabled monitoring', async () => {
    applyRuntimeCapabilities({ telemetry: false })
    const wrapper = mount(OverviewPage)
    await flushPromises()
    expect(wrapper.find('a[href="/monitoring"]').exists()).toBe(false)
    expect(wrapper.text()).toContain('确认告警数')
    expect(wrapper.text()).toContain('故障概率异常')
    expect(wrapper.get('[role="status"]').text()).toContain('历史会话和告警统计仍保留')
    expect(wrapper.find('a.kpi-card[href="/reports"]').exists()).toBe(true)
    expect(wrapper.text()).not.toContain('或启动实时检测后')
  })
})
