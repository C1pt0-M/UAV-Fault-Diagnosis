import { mount } from '@vue/test-utils'
import { describe, expect, it } from 'vitest'

import ReportsWorkspacePanel from './ReportsWorkspacePanel.vue'
import { createReportActionItems } from '@/modules/reports/reportsView'

describe('ReportsWorkspacePanel regression', () => {
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

    expect(text).toContain('当前列表报告')
    expect(text).toContain('平均置信度')
    expect(text).toContain('可下载报告')
    expect(text).toContain('可 AI 解读')
    expect(text).toContain('重置')
    expect(text).toContain('刷新')
  })

  it('renders compact report operation entries with danger delete semantics', () => {
    const actionItems = createReportActionItems()

    expect(actionItems.map((item) => item.label)).toEqual(['详情', 'AI 解读', '下载', '删除'])
    expect(actionItems.find((item) => item.key === 'remove')).toMatchObject({ type: 'danger' })
  })
})
