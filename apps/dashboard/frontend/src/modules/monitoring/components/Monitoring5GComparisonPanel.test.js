import { shallowMount } from '@vue/test-utils'
import { describe, expect, it } from 'vitest'

import Monitoring5GComparisonPanel from './Monitoring5GComparisonPanel.vue'

describe('Monitoring5GComparisonPanel', () => {
  it('renders three latest realtime sessions and enhanced 5G improvements', () => {
    const wrapper = shallowMount(Monitoring5GComparisonPanel, {
      props: {
        comparison: {
          complete: true,
          profiles: [
            { profile: 'baseline', available: true, session_id: 'TS-BASE', avg_latency_ms: 12, p95_latency_ms: 16, loss_rate: 0.001, data_integrity: 0.999, frame_count: 1500 },
            { profile: 'enhanced_5g', available: true, session_id: 'TS-5G', avg_latency_ms: 29.5, p95_latency_ms: 32, loss_rate: 0.0057, data_integrity: 0.9943, frame_count: 1918 },
            { profile: 'degraded', available: true, session_id: 'TS-WEAK', avg_latency_ms: 85, p95_latency_ms: 112, loss_rate: 0.031, data_integrity: 0.969, frame_count: 1450 }
          ],
          enhanced_vs_degraded: {
            latency_reduction_ms: 55.5,
            p95_reduction_ms: 80,
            loss_reduction: 0.0253,
            integrity_improvement: 0.0253
          }
        }
      }
    })

    expect(wrapper.findAll('.profile-result')).toHaveLength(3)
    expect(wrapper.text()).toContain('TS-BASE')
    expect(wrapper.text()).toContain('TS-5G')
    expect(wrapper.text()).toContain('TS-WEAK')
    expect(wrapper.text()).toContain('平均时延降低 55.5 ms')
    expect(wrapper.text()).toContain('丢包率降低 2.53 pct')
    expect(wrapper.text()).toContain('完整率提升 2.53 pct')
  })

  it('keeps all three profile slots and marks missing realtime evidence', () => {
    const wrapper = shallowMount(Monitoring5GComparisonPanel, {
      props: {
        comparison: {
          profiles: [
            { profile: 'enhanced_5g', available: true, session_id: 'TS-ONLY-5G' }
          ],
          enhanced_vs_degraded: {}
        }
      }
    })

    expect(wrapper.findAll('.profile-result')).toHaveLength(3)
    expect(wrapper.text()).toContain('TS-ONLY-5G')
    expect(wrapper.text().match(/待采集/g)).toHaveLength(2)
    expect(wrapper.text()).toContain('完成 enhanced_5g 与 degraded 实时会话后生成改善量')
  })
})
