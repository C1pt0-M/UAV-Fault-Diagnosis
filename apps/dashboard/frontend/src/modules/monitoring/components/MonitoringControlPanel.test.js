import { shallowMount } from '@vue/test-utils'
import { describe, expect, it } from 'vitest'

import MonitoringControlPanel from './MonitoringControlPanel.vue'

function mountControlPanel(overrides = {}) {
  return shallowMount(MonitoringControlPanel, {
    props: {
      actionLoading: {},
      backendOptions: [
        { label: 'RflySim', value: 'rfly' },
        { label: 'CSV 回放', value: 'replay' }
      ],
      blindTestEnabled: false,
      currentState: 'IDLE',
      onlineModels: [
        {
          name: 'online_detector_kg_mstca_rfly',
          display_name: 'KG-MSTCA 在线检测模型'
        }
      ],
      replayAssets: [],
      rflyDiagnostics: null,
      rflyDiagnosticsLoading: false,
      selectedBackend: 'replay',
      selectedDuration: 60,
      selectedModel: 'online_detector_kg_mstca_rfly',
      selectedReplayAssetId: null,
      rflyPx4mavctrlPath: '',
      ...overrides
    }
  })
}

describe('MonitoringControlPanel telemetry console', () => {
  it('renders three compact status cards with readable Chinese labels', () => {
    const wrapper = mountControlPanel()

    expect(wrapper.findAll('.monitoring-param-card')).toHaveLength(3)
    expect(wrapper.text()).toContain('当前会话状态')
    expect(wrapper.text()).toContain('空闲')
    expect(wrapper.text()).toContain('数据源')
    expect(wrapper.text()).toContain('CSV 回放')
    expect(wrapper.text()).toContain('在线检测模型')
    expect(wrapper.text()).toContain('KG-MSTCA 在线检测模型')
    expect(wrapper.text()).toContain('online_detector_kg_mstca_rfly')
  })

  it('does not render raw model id when no model is selected', () => {
    const wrapper = mountControlPanel({
      selectedModel: ''
    })

    expect(wrapper.text()).toContain('未选择')
    expect(wrapper.text()).not.toContain('online_detector_kg_mstca_rfly')
  })

  it('shows RflySim diagnostics and disables collection when not ready', () => {
    const wrapper = mountControlPanel({
      selectedBackend: 'rfly',
      rflyDiagnostics: {
        ready: false,
        summary: {
          status: 'not_ready',
          message: 'PX4MavCtrlV4.py 未配置'
        },
        checks: {
          px4mavctrl: { ok: false, label: 'PX4MavCtrlV4.py 缺失' }
        },
        issues: [{ code: 'px4mavctrl_missing', message: '缺少 PX4MavCtrlV4.py' }]
      }
    })

    expect(wrapper.find('[data-test="rfly-diagnostics-panel"]').exists()).toBe(true)
    expect(wrapper.text()).toContain('RflySim 配置诊断')
    expect(wrapper.text()).toContain('PX4MavCtrlV4.py 未配置')
    expect(wrapper.find('[data-test="start-collection"]').attributes('disabled')).toBeDefined()
  })

  it('emits diagnostics request when the RflySim diagnostics button is clicked', async () => {
    const wrapper = mountControlPanel({
      selectedBackend: 'rfly',
      rflyDiagnostics: {
        ready: true,
        summary: { status: 'ready', message: 'RflySim 链路配置可用' },
        checks: {},
        issues: []
      }
    })

    await wrapper.find('[data-test="rfly-diagnostics-button"]').trigger('click')

    expect(wrapper.emitted('test-rfly-diagnostics')).toHaveLength(1)
  })
})
