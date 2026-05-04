import { mount } from '@vue/test-utils'
import { describe, expect, it } from 'vitest'

import AIHeroSection from './components/AIHeroSection.vue'

describe('AIHeroSection regression', () => {
  it('wraps the main copy in a dedicated content container for consistent insets', () => {
    const wrapper = mount(AIHeroSection)

    expect(wrapper.find('.ai-hero__content').exists()).toBe(true)
  })
})
