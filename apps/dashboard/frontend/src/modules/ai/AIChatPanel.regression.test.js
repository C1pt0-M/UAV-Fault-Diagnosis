import { mount } from '@vue/test-utils'
import { nextTick } from 'vue'
import { describe, expect, it, vi } from 'vitest'

import AIChatPanel from './components/AIChatPanel.vue'

describe('AIChatPanel regression', () => {
  it('renders assistant answers with a highlighted conclusion block and collapsible references', () => {
    const wrapper = mount(AIChatPanel, {
      props: {
        loadingMessages: false,
        sending: false,
        canSeedAnalysis: true,
        composer: '',
        messages: [
          {
            id: 'm1',
            role: 'assistant',
            content: [
              '直接结论：当前报告更接近电机故障。',
              '主要原因：动力输出一致性异常，并伴随风险评分升高。',
              '风险与建议：建议优先检查电机与 ESC，再回看原始日志。'
            ].join('\n\n'),
            citations: [{ source_type: 'report', source_id: '42', title: '诊断报告 #42', snippet: '电机故障，置信度 92.4%' }],
            used_context: {}
          }
        ]
      }
    })

    const text = wrapper.text()
    expect(text).toContain('直接结论')
    expect(text).toContain('主要原因')
    expect(text).toContain('风险与建议')
    expect(wrapper.find('.message-card__summary').exists()).toBe(true)
    expect(wrapper.find('details.message-card__references').exists()).toBe(true)
  })

  it('scrolls to the newest message when chat content updates', async () => {
    const wrapper = mount(AIChatPanel, {
      props: {
        loadingMessages: false,
        sending: false,
        canSeedAnalysis: true,
        composer: '',
        messages: [
          {
            id: 'm1',
            role: 'assistant',
            content: 'Direct conclusion: initial message.',
            citations: [],
            used_context: {}
          }
        ]
      }
    })

    const stream = wrapper.get('.message-stream').element
    const scrollTo = vi.fn()
    Object.defineProperty(stream, 'scrollHeight', {
      configurable: true,
      value: 640
    })
    Object.defineProperty(stream, 'scrollTo', {
      configurable: true,
      value: scrollTo
    })

    await wrapper.setProps({
      messages: [
        {
          id: 'm1',
          role: 'assistant',
          content: 'Direct conclusion: initial message.',
          citations: [],
          used_context: {}
        },
        {
          id: 'm2',
          role: 'assistant',
          content: 'Direct conclusion: latest reply.',
          citations: [],
          used_context: {}
        }
      ]
    })
    await nextTick()

    expect(scrollTo).toHaveBeenCalledWith({
      top: 640,
      behavior: 'smooth'
    })
  })
})
