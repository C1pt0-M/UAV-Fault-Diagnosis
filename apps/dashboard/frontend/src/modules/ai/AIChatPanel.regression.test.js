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
    expect(text).toContain('参考资料（1）')
    expect(wrapper.find('.message-card__summary').exists()).toBe(true)
    expect(wrapper.find('details.message-card__references').exists()).toBe(true)
  })

  it('renders quick prompts in the empty state and fills composer without sending', async () => {
    const wrapper = mount(AIChatPanel, {
      props: {
        loadingMessages: false,
        sending: false,
        canSeedAnalysis: false,
        composer: '',
        messages: []
      }
    })

    expect(wrapper.text()).toContain('解释当前报告结论')
    expect(wrapper.text()).toContain('列出风险和处置建议')
    expect(wrapper.text()).toContain('说明模型判断依据')

    await wrapper.get('.quick-prompt').trigger('click')

    expect(wrapper.emitted('send')).toBeUndefined()
    expect(wrapper.emitted('update:composer')?.[0]).toEqual(['解释当前报告结论'])
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
