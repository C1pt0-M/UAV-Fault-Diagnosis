import { defineComponent, h } from 'vue'
import { flushPromises, mount } from '@vue/test-utils'
import { beforeEach, describe, expect, it, vi } from 'vitest'

const routerPush = vi.hoisted(() => vi.fn())
const authApiMock = vi.hoisted(() => ({
  login: vi.fn()
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
  props: ['label', 'prop'],
  setup(props, { slots }) {
    return () => h('label', { 'data-prop': props.prop }, [h('span', props.label), slots.default?.()])
  }
})

const ElInputStub = defineComponent({
  name: 'ElInput',
  props: ['modelValue', 'placeholder', 'type'],
  emits: ['update:modelValue'],
  template:
    '<input :type="type || \'text\'" :placeholder="placeholder" :value="modelValue" @input="$emit(\'update:modelValue\', $event.target.value)" />'
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

describe('LoginPage', () => {
  beforeEach(() => {
    routerPush.mockReset()
    authApiMock.login.mockReset()
    messageMock.success.mockReset()
    messageMock.error.mockReset()
    localStorage.clear()
  })

  it('renders only the administrator login entry and setup link', async () => {
    const wrapper = mountLoginPage()

    expect(wrapper.text()).toContain('智诊天眸')
    expect(wrapper.text()).toContain('无人机故障诊断研究平台')
    expect(wrapper.text()).toContain('登录系统')
    expect(wrapper.text()).toContain('首次使用？')
    expect(wrapper.text()).toContain('进入系统初始化')
    expect(wrapper.text()).not.toContain('立即注册')
    expect(wrapper.text()).not.toContain('创建研究账号')
    expect(wrapper.find('[data-prop="email"]').exists()).toBe(false)

    await wrapper.findAll('button').find((button) => button.text().includes('系统初始化')).trigger('click')

    expect(routerPush).toHaveBeenCalledWith('/setup')
  })

  it('logs in and stores the authenticated session', async () => {
    authApiMock.login.mockResolvedValue({
      data: {
        token: 'token-1',
        user: { username: 'admin', role: 'admin' }
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
    expect(JSON.parse(localStorage.getItem('user'))).toEqual({ username: 'admin', role: 'admin' })
    expect(routerPush).toHaveBeenCalledWith('/')
    expect(messageMock.success).toHaveBeenCalledWith('登录成功')
  })
})
