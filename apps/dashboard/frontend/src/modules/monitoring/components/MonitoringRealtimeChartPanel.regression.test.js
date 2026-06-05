import { mount } from '@vue/test-utils'
import { describe, expect, it } from 'vitest'

import MonitoringRealtimeChartPanel from './MonitoringRealtimeChartPanel.vue'

describe('MonitoringRealtimeChartPanel chart polish', () => {
  it('keeps the main risk chart annotated while small charts stay compact', () => {
    const wrapper = mount(MonitoringRealtimeChartPanel, {
      props: {
        chartTimes: [0, 1, 2],
        faultProbData: [0.1, 0.4, 0.2],
        motorSeries: [{ name: 'Motor 1', data: [1200, 1210, 1208] }],
        attitudeSeries: [{ name: 'Roll', data: [0, 0.1, 0.2] }],
        positionSeries: [{ name: 'Position Z', data: [1, 2, 3] }]
      },
      global: {
        stubs: {
          TimeSeriesChart: {
            props: [
              'showSlider',
              'showLegend',
              'showThresholdLine',
              'thresholdValue',
              'thresholdLabel'
            ],
            template: `
              <div
                class="chart-stub"
                :data-show-slider="String(showSlider)"
                :data-show-legend="String(showLegend)"
                :data-show-threshold-line="String(showThresholdLine)"
                :data-threshold-value="thresholdValue"
                :data-threshold-label="thresholdLabel"
              />
            `
          }
        }
      }
    })

    const charts = wrapper.findAll('.chart-stub')
    expect(charts).toHaveLength(4)
    expect(charts[0].attributes('data-show-slider')).toBe('true')
    expect(charts[0].attributes('data-show-legend')).toBe('true')
    expect(charts[0].attributes('data-show-threshold-line')).toBe('true')
    expect(charts[0].attributes('data-threshold-value')).toBe('0.7')
    expect(charts[0].attributes('data-threshold-label')).toBe('风险阈值 0.70')

    for (const chart of charts.slice(1)) {
      expect(chart.attributes('data-show-slider')).toBe('false')
      expect(chart.attributes('data-show-legend')).toBe('false')
      expect(chart.attributes('data-show-threshold-line')).toBe('false')
    }
  })
})
