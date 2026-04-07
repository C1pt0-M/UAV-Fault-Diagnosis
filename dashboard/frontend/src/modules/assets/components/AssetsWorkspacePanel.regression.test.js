import { mount } from '@vue/test-utils'
import { describe, expect, it } from 'vitest'

import AssetsWorkspacePanel from './AssetsWorkspacePanel.vue'

describe('AssetsWorkspacePanel regression', () => {
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
})
