import { defineComponent, h } from 'vue'
import { flushPromises, mount } from '@vue/test-utils'
import { beforeEach, describe, expect, it, vi } from 'vitest'

const routerPush = vi.hoisted(() => vi.fn())
const routeMock = vi.hoisted(() => ({ path: '/login' }))
const authApiMock = vi.hoisted(() => ({
  login: vi.fn(),
  register: vi.fn()
}))
const messageMock = vi.hoisted(() => ({
  success: vi.fn(),
  error: vi.fn()
}))

vi.mock('vue-router', () => ({
  useRouter: () => ({ push: routerPush }),
  useRoute: () => routeMock
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
    routeMock.path = '/login'
    routerPush.mockReset()
    authApiMock.login.mockReset()
    authApiMock.register.mockReset()
    messageMock.success.mockReset()
    messageMock.error.mockReset()
    localStorage.clear()
  })

  it('renders the login form and register entry', async () => {
    const wrapper = mountLoginPage()

    expect(wrapper.text()).toContain('基于数据驱动的无人机故障智能诊断系统')
    expect(wrapper.text()).toContain('无人机飞行日志、在线监测与故障诊断一体化平台')
    expect(wrapper.text()).toContain('登录系统')
    expect(wrapper.text()).toContain('还没有账号？')
    expect(wrapper.text()).toContain('立即注册')
    expect(wrapper.find('[data-prop="email"]').exists()).toBe(false)
    expect(wrapper.find('[data-prop="confirmPassword"]').exists()).toBe(false)

    await wrapper.findAll('button').find((button) => button.text().includes('立即注册')).trigger('click')

    expect(routerPush).toHaveBeenCalledWith('/register')
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

  it('registers with username and password then returns to login', async () => {
    routeMock.path = '/register'
    authApiMock.register.mockResolvedValue({
      data: { username: 'newuser', role: 'user' }
    })
    const wrapper = mountLoginPage()

    expect(wrapper.text()).toContain('注册账号')
    expect(wrapper.text()).toContain('已有账号？')
    expect(wrapper.text()).toContain('返回登录')
    expect(wrapper.find('[data-prop="confirmPassword"]').exists()).toBe(true)

    wrapper.vm.form.username = 'newuser'
    wrapper.vm.form.password = 'newuser123'
    wrapper.vm.form.confirmPassword = 'newuser123'
    await wrapper.findAll('button').find((button) => button.text().includes('注册账号')).trigger('click')
    await flushPromises()

    expect(authApiMock.register).toHaveBeenCalledWith({
      username: 'newuser',
      password: 'newuser123'
    })
    expect(messageMock.success).toHaveBeenCalledWith('注册成功，请登录')
    expect(routerPush).toHaveBeenCalledWith('/login')
  })
})
