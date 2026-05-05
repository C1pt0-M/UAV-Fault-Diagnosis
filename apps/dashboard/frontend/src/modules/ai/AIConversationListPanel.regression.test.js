import { mount } from '@vue/test-utils'
import { describe, expect, it } from 'vitest'

import AIConversationListPanel from './components/AIConversationListPanel.vue'
import { createConversationPreview } from './aiView'

const longPreview = '直接结论：当前报告显示电机故障风险较高，建议优先检查电机绕组、相电流、ESC 供电和温升记录，并结合原始日志复核异常出现的时间窗口。'

const conversations = [
  {
    id: 12,
    title: '报告 #12 AI 解读',
    updated_at: '2026-04-10 19:51:59',
    scene: 'report_analysis',
    report_id: 12,
    meta: {
      last_message_preview: longPreview,
      context_summary: {
        scene_label: '报告分析',
        report_id: 12
      }
    }
  }
]

describe('AIConversationListPanel regression', () => {
  it('supports a collapsible rail mode for medium-width AI workspaces', async () => {
    const wrapper = mount(AIConversationListPanel, {
      props: {
        conversations,
        currentConversationId: 12,
        loading: false,
        railExpanded: false
      }
    })

    expect(wrapper.classes()).not.toContain('ai-list--rail-expanded')
    expect(wrapper.get('.ai-list__rail-toggle').text()).toContain('会话')

    await wrapper.get('.ai-list__rail-toggle').trigger('click')
    expect(wrapper.emitted('toggle-rail')).toHaveLength(1)

    await wrapper.setProps({ railExpanded: true })
    expect(wrapper.classes()).toContain('ai-list--rail-expanded')
    expect(wrapper.get('.ai-list__rail-toggle').text()).toContain('收起')
  })

  it('keeps conversation history internally scrollable and truncates previews', () => {
    const wrapper = mount(AIConversationListPanel, {
      props: {
        conversations,
        currentConversationId: 12,
        loading: false,
        railExpanded: true
      }
    })

    expect(wrapper.find('.conversation-list--scrollable').exists()).toBe(true)
    expect(wrapper.find('.conversation-item__preview').text().length).toBeLessThan(longPreview.length)
    expect(createConversationPreview(longPreview, 36)).toMatch(/\.\.\.$/)
    expect(createConversationPreview(longPreview, 36).length).toBeLessThanOrEqual(39)
  })
})
