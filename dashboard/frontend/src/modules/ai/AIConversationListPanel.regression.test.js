import { mount } from '@vue/test-utils'
import { describe, expect, it } from 'vitest'

import AIConversationListPanel from './components/AIConversationListPanel.vue'

const conversations = [
  {
    id: 12,
    title: '报告 #12 AI 解读',
    updated_at: '2026-04-10 19:51:59',
    scene: 'report_analysis',
    report_id: 12,
    meta: {
      last_message_preview: '当前报告需要复核。',
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
})
