import { defineComponent } from 'vue'
import { flushPromises, mount } from '@vue/test-utils'
import { beforeEach, describe, expect, it, vi } from 'vitest'

const routerPush = vi.hoisted(() => vi.fn())
const authApiMock = vi.hoisted(() => ({
  login: vi.fn(),
  register: vi.fn()
}))
const messageMock = vi.hoisted(() => ({
  success: vi.fn(),
  error: vi.fn()
}))

vi.mock('vue-router', () => ({
  useRouter: () => ({ push: routerPush })
}))

vi.mock('element-plus', () => ({
  ElMessage: messageMock
}))

vi.mock('@/shared/services', () => ({
  authApi: authApiMock
}))

const ElFormStub = defineComponent({
  name: 'ElForm',
  setup(_, { expose, slots }) {
    expose({
      validate(callback) {
        callback(true)
      }
    })
    return () => slots.default?.()
  }
})

const ElFormItemStub = defineComponent({
  name: 'ElFormItem',
  setup(_, { slots }) {
    return () => slots.default?.()
  }
})

const ElInputStub = defineComponent({
  name: 'ElInput',
  props: ['modelValue'],
  emits: ['update:modelValue'],
  template: '<input :value="modelValue" @input="$emit(\'update:modelValue\', $event.target.value)" />'
})

const ElButtonStub = defineComponent({
  name: 'ElButton',
  emits: ['click'],
  template: '<button type="button" @click="$emit(\'click\')"><slot /></button>'
})

import LoginPage from '@/modules/auth/LoginPage.vue'

function mountLoginPage() {
  return mount(LoginPage, {
    global: {
      stubs: {
        ElForm: ElFormStub,
        ElFormItem: ElFormItemStub,
        ElInput: ElInputStub,
        ElButton: ElButtonStub
      }
    }
  })
}

describe('LoginPage inline register', () => {
  beforeEach(() => {
    routerPush.mockReset()
    authApiMock.login.mockReset()
    authApiMock.register.mockReset()
    messageMock.success.mockReset()
    messageMock.error.mockReset()
    localStorage.clear()
  })

  it('switches to registration inside the current login page without routing away', async () => {
    const wrapper = mountLoginPage()

    expect(wrapper.text()).toContain('登录系统')
    expect(wrapper.text()).toContain('还没有账号？')

    await wrapper.findAll('button').find((button) => button.text().includes('注册')).trigger('click')

    expect(routerPush).not.toHaveBeenCalledWith('/register')
    expect(wrapper.text()).toContain('创建新账号')
    expect(wrapper.text()).toContain('已有账号？')
  })

  it('registers from the inline form and returns to login mode without page navigation', async () => {
    authApiMock.register.mockResolvedValue({ success: true })
    const wrapper = mountLoginPage()

    await wrapper.findAll('button').find((button) => button.text().includes('注册')).trigger('click')
    wrapper.vm.form.username = 'new_user'
    wrapper.vm.form.password = 'new_password'
    await wrapper.findAll('button').find((button) => button.text().includes('创建账号')).trigger('click')
    await flushPromises()

    expect(authApiMock.register).toHaveBeenCalledWith({
      username: 'new_user',
      password: 'new_password'
    })
    expect(routerPush).not.toHaveBeenCalled()
    expect(wrapper.text()).toContain('登录系统')
    expect(messageMock.success).toHaveBeenCalledWith('注册成功，请使用新账号登录')
  })

  it('still logs in from the default login mode', async () => {
    authApiMock.login.mockResolvedValue({
      data: {
        token: 'token-1',
        user: { username: 'admin' }
      }
    })
    const wrapper = mountLoginPage()

    wrapper.vm.form.username = 'admin'
    wrapper.vm.form.password = 'admin123'
    await wrapper.findAll('button').find((button) => button.text().includes('登录')).trigger('click')
    await flushPromises()

    expect(authApiMock.login).toHaveBeenCalledWith({
      username: 'admin',
      password: 'admin123'
    })
    expect(localStorage.getItem('token')).toBe('token-1')
    expect(routerPush).toHaveBeenCalledWith('/')
  })
})
