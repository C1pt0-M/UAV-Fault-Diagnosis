import { mount } from '@vue/test-utils'
import { describe, expect, it } from 'vitest'

import MonitoringSessionDialog from './MonitoringSessionDialog.vue'

const globalStubs = {
  ElDialog: {
    props: ['modelValue', 'title'],
    template: '<section v-if="modelValue"><h2>{{ title }}</h2><slot /></section>'
  },
  ElDescriptions: {
    template: '<dl><slot /></dl>'
  },
  ElDescriptionsItem: {
    props: ['label'],
    template: '<div><dt>{{ label }}</dt><dd><slot /></dd></div>'
  },
  ElEmpty: {
    props: ['description'],
    template: '<div>{{ description }}</div>'
  }
}

describe('MonitoringSessionDialog', () => {
  it('renders the selected monitoring session fields and confirmed fault count', () => {
    const wrapper = mount(MonitoringSessionDialog, {
      props: {
        visible: true,
        sessionDetail: {
          id: 'TS-262735',
          drone_id: 'DRONE-001',
          status: 'completed',
          fault_events_count: 2
        }
      },
      global: { stubs: globalStubs }
    })

    const text = wrapper.text()
    expect(text).toContain('会话详情')
    expect(text).toContain('TS-262735')
    expect(text).toContain('DRONE-001')
    expect(text).toContain('completed')
    expect(text).toContain('故障事件')
    expect(text).toContain('2')
  })

  it('renders the empty state when no session has been selected', () => {
    const wrapper = mount(MonitoringSessionDialog, {
      props: { visible: true, sessionDetail: null },
      global: { stubs: globalStubs }
    })

    expect(wrapper.text()).toContain('暂无会话详情')
  })
})
