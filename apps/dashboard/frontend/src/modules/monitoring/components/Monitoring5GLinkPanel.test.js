import { shallowMount } from '@vue/test-utils'
import { describe, expect, it } from 'vitest'

import Monitoring5GLinkPanel from './Monitoring5GLinkPanel.vue'

describe('Monitoring5GLinkPanel', () => {
  it('renders live 5G KPI, SLA state and latency trend contract', () => {
    const wrapper = shallowMount(Monitoring5GLinkPanel, {
      props: {
        networkLink: {
          enabled: true,
          profile: 'enhanced_5g',
          status: 'active',
          avg_latency_ms: 29.5,
          p95_latency_ms: 32,
          loss_rate: 0.0057,
          data_integrity: 0.9943,
          rx_packets: 1918,
          lost_packets: 11,
          metrics_csv: 'runtime://telemetry/5g_link/TS-5G.csv'
        },
        mec: { enabled: true, location: 'mec', status: 'active' },
        inference: {
          location: 'mec',
          avg_inference_ms: 4.8,
          p95_inference_ms: 6.2,
          avg_decision_latency_ms: 34.7,
          p95_decision_latency_ms: 41.3,
          inference_count: 128
        },
        processes: { network_link: 'running', mec: 'running' },
        times: [1, 2],
        latencySeries: [28.1, 29.7],
        lossSeries: [0, 0.0057],
        integritySeries: [1, 0.9943]
      }
    })

    expect(wrapper.text()).toContain('5G链路运行监测')
    expect(wrapper.text()).toContain('enhanced_5g')
    expect(wrapper.text()).toContain('29.5 ms')
    expect(wrapper.text()).toContain('0.57%')
    expect(wrapper.text()).toContain('99.43%')
    expect(wrapper.text()).toContain('MEC 边缘节点')
    expect(wrapper.text()).toContain('4.8 ms')
    expect(wrapper.text()).toContain('6.2 ms')
    expect(wrapper.text()).toContain('34.7 ms')
    expect(wrapper.text()).toContain('41.3 ms')
    expect(wrapper.text()).toContain('128')
    expect(wrapper.text()).toContain('链路稳定')
    expect(wrapper.text()).toContain('达标')
    expect(wrapper.findComponent({ name: 'TimeSeriesChart' }).props('series')).toEqual([
      { name: 'E2E latency', data: [28.1, 29.7] }
    ])
  })

  it('shows waiting state when no live metrics are available', () => {
    const wrapper = shallowMount(Monitoring5GLinkPanel, {
      props: {
        networkLink: { enabled: true, profile: 'enhanced_5g', status: 'waiting' },
        processes: { network_link: 'running' }
      }
    })

    expect(wrapper.text()).toContain('正在建立')
    expect(wrapper.text()).toContain('等待数据')
  })
})
