import { describe, expect, it, vi } from 'vitest'
import { shallowMount } from '@vue/test-utils'

import RegisterPage from '@/modules/auth/RegisterPage.vue'

describe('RegisterPage', () => {
  it('renders only username and password fields plus the login entry', () => {
    const wrapper = shallowMount(RegisterPage, {
      global: {
        stubs: ['el-form', 'el-form-item', 'el-input', 'el-button'],
        mocks: {
          $router: { push: vi.fn() }
        },
        provide: {
          router: { push: vi.fn() }
        }
      }
    })

    expect(wrapper.text()).toContain('注册')
    expect(wrapper.text()).toContain('已有账号')
    expect(wrapper.text()).toContain('创建新账号')
    expect(wrapper.text()).not.toContain('姓名')
    expect(wrapper.text()).not.toContain('邮箱')
    expect(wrapper.text()).not.toContain('使用最小信息完成注册')
  })
})
