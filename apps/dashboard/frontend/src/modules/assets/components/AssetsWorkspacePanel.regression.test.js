import { mount } from '@vue/test-utils'
import { describe, expect, it } from 'vitest'

import AssetsWorkspacePanel from './AssetsWorkspacePanel.vue'
import { createAssetActionItems } from '@/modules/assets/assetsView'

describe('AssetsWorkspacePanel regression', () => {
  const sampleAssets = [
    {
      id: 1,
      filename: 'Case_1007000005.csv',
      kind: 'csv',
      size_bytes: 1024 * 1024,
      created_at: '2026-04-06 20:54:57',
      meta: { prepared_csv_path: 'prepared.csv' }
    },
    {
      id: 2,
      filename: 'flight_001.ulg',
      kind: 'ulg',
      size_bytes: 1024 * 1024 * 2,
      created_at: '2026-04-06 21:02:29',
      meta: {}
    }
  ]

  it('renders the assets workspace actions without crashing', () => {
    const wrapper = mount(AssetsWorkspacePanel, {
      props: {
        activeKind: 'all',
        assets: [],
        currentPage: 1,
        pageSize: 10,
        total: 0,
        loading: false,
        uploading: false,
        preparingMap: {},
        deletingMap: {}
      },
      global: {
        directives: {
          loading: () => {}
        }
      }
    })

    expect(wrapper.text()).toContain('上传新资产')
    expect(wrapper.text()).toContain('前往报告中心')
  })

  it('renders compact current-page asset stats', () => {
    const wrapper = mount(AssetsWorkspacePanel, {
      props: {
        activeKind: 'all',
        assets: sampleAssets,
        currentPage: 1,
        pageSize: 10,
        total: 18,
        loading: false,
        uploading: false,
        preparingMap: {},
        deletingMap: {}
      },
      global: {
        directives: {
          loading: () => {}
        }
      }
    })

    const text = wrapper.text()
    expect(text).toContain('当前筛选总数')
    expect(text).toContain('18')
    expect(text).toContain('当前页 CSV')
    expect(text).toContain('当前页 ULG')
    expect(text).toContain('当前页已就绪')
    expect(text).toContain('当前页待处理')
  })

  it('keeps prepare action only for ULG assets', () => {
    const csvActions = createAssetActionItems(sampleAssets[0]).map((item) => item.label)
    const ulgActions = createAssetActionItems(sampleAssets[1]).map((item) => item.label)

    expect(csvActions).not.toContain('处理')
    expect(ulgActions).toContain('处理')
  })
})
