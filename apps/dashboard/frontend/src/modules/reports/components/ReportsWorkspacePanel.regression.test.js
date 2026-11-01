import { flushPromises, mount } from '@vue/test-utils'
import { defineComponent } from 'vue'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { applyRuntimeCapabilities } from '@/shared/state/runtimeCapabilities'

const routerPush = vi.hoisted(() => vi.fn())
vi.mock('vue-router', () => ({ useRoute: () => ({ query: {} }), useRouter: () => ({ push: routerPush }) }))
vi.mock('@/shared/services', () => ({
  assetApi: { list: vi.fn().mockResolvedValue({ data: [] }) },
  diagnosisReportApi: { list: vi.fn().mockResolvedValue({ data: [] }) }
}))
vi.mock('element-plus', () => ({ ElMessage: { warning: vi.fn(), error: vi.fn() }, ElMessageBox: {} }))

import ReportsWorkspacePanel from './ReportsWorkspacePanel.vue'
import { createReportActionItems } from '@/modules/reports/reportsView'
import { useReportsWorkspace } from '@/modules/reports/composables/useReportsWorkspace'

describe('ReportsWorkspacePanel regression', () => {
  beforeEach(() => {
    applyRuntimeCapabilities({ ai: true })
    routerPush.mockClear()
  })
  const sampleReports = [
    {
      id: 3,
      run_id: 11,
      asset_id: 178,
      model_name: 'transfer3_cnnlstm_normalproto',
      prediction_label: 'motor',
      confidence: 0.9959,
      created_at: '2026-04-10 17:29:01'
    },
    {
      id: 2,
      run_id: 2,
      asset_id: 12,
      model_name: 'online_detector_kg_mstca_rfly',
      prediction_label: 'sensor',
      confidence: 0.9,
      created_at: '2026-04-07 14:28:01'
    }
  ]

  function mountPanel(props = {}) {
    return mount(ReportsWorkspacePanel, {
      props: {
        assetOptions: [{ id: 178, filename: 'TS-148898.csv' }],
        assetNameMap: new Map([[178, 'TS-148898.csv']]),
        filters: { assetId: undefined, label: '' },
        loading: false,
        pagination: { page: 1, perPage: 20, total: 42, pages: 3 },
        reports: sampleReports,
        ...props
      },
      global: {
        directives: {
          loading: () => {}
        }
      }
    })
  }

  it('renders current-list report summary and toolbar actions', () => {
    const wrapper = mountPanel()
    const text = wrapper.text()

    expect(text).toContain('报告总数')
    expect(text).toContain('当前页报告')
    expect(text).toContain('异常报告')
    expect(text).toContain('可 AI 解读')
    expect(text).toContain('共 42 份报告')
    expect(text).toContain('重置')
    expect(text).toContain('刷新')
  })

  it('keeps report management without advertising disabled AI actions or counts', async () => {
    applyRuntimeCapabilities({ ai: false })
    const wrapper = mountPanel()
    await flushPromises()
    expect(wrapper.text()).not.toContain('可 AI 解读')
    expect(wrapper.vm.reportActions.map(item => item.key)).toEqual(['detail', 'remove'])
    expect(wrapper.get('[role="status"]').text()).toContain('报告详情与管理仍可使用')
  })

  it.each([false, true])('guards report assistant navigation when ai=%s', async (enabled) => {
    applyRuntimeCapabilities({ ai: enabled })
    const harness = defineComponent({ setup: useReportsWorkspace, template: '<div />' })
    const wrapper = mount(harness)
    await flushPromises()
    wrapper.vm.showDetail(sampleReports[0])
    expect(wrapper.vm.detailVisible).toBe(true)
    wrapper.vm.openInAssistant(sampleReports[0])
    if (enabled) expect(routerPush).toHaveBeenCalledWith({ path: '/ai', query: { scene: 'report_analysis', report_id: '3', auto_analyze: '1' } })
    else expect(routerPush).not.toHaveBeenCalled()
    wrapper.unmount()
  })

  it('renders compact report operation entries with danger delete semantics', () => {
    const actionItems = createReportActionItems()

    expect(actionItems.map((item) => item.label)).toEqual(['详情', 'AI 解读', '删除'])
    expect(actionItems.find((item) => item.key === 'remove')).toMatchObject({ type: 'danger' })
  })
})
