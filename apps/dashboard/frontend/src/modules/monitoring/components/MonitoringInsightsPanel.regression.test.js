import { mount } from '@vue/test-utils'
import { describe, expect, it } from 'vitest'

import MonitoringInsightsPanel from './MonitoringInsightsPanel.vue'

describe('MonitoringInsightsPanel alert polish', () => {
  it('uses a compact empty state for alerts and readable Chinese labels', () => {
    const wrapper = mount(MonitoringInsightsPanel, {
      props: {
        alerts: [],
        sampleSummaryItems: [{ label: '采样点数量', value: 203 }]
      },
      global: {
        stubs: {
          ElEmpty: {
            props: ['description'],
            template: '<div class="el-empty">{{ description }}</div>'
          }
        }
      }
    })

    expect(wrapper.find('.monitoring-alert-empty').exists()).toBe(true)
    expect(wrapper.text()).toContain('当前没有故障告警')
    expect(wrapper.text()).not.toContain('楂樺嵄')
    expect(wrapper.text()).not.toContain('涓嵄')
  })

  it('renders alert table headers and severity labels in Chinese', () => {
    const wrapper = mount(MonitoringInsightsPanel, {
      props: {
        alerts: [
          {
            id: 1,
            start_time: '10:22:14',
            diagnosis: { label: '电机电流异常', severity: 'high' }
          }
        ],
        sampleSummaryItems: []
      }
    })

    expect(wrapper.text()).toContain('时间')
    expect(wrapper.text()).toContain('事件 / 信号')
    expect(wrapper.text()).toContain('级别')
    expect(wrapper.text()).toContain('高危')
  })
})
