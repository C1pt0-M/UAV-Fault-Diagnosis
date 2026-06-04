import { defineComponent, h } from 'vue'
import { flushPromises, mount } from '@vue/test-utils'
import { beforeEach, describe, expect, it, vi } from 'vitest'

const routerPush = vi.hoisted(() => vi.fn())
const setupApiMock = vi.hoisted(() => ({
  status: vi.fn(),
  initialize: vi.fn()
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
  setupApi: setupApiMock
}))

const ElFormStub = defineComponent({
  name: 'ElForm',
  setup(_, { expose, slots }) {
    expose({
      validate(callback) {
        callback(true)
      }
    })
    return () => h('form', slots.default?.())
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

import SetupPage from '@/modules/setup/SetupPage.vue'

function mountSetupPage() {
  return mount(SetupPage, {
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

describe('SetupPage', () => {
  beforeEach(() => {
    routerPush.mockReset()
    setupApiMock.status.mockReset()
    setupApiMock.initialize.mockReset()
    messageMock.success.mockReset()
    messageMock.error.mockReset()
  })

  it('renders the administrator initialization form when setup is needed', async () => {
    setupApiMock.status.mockResolvedValue({
      data: { is_installed: false, needs_setup: true }
    })

    const wrapper = mountSetupPage()
    await flushPromises()

    expect(setupApiMock.status).toHaveBeenCalledTimes(1)
    expect(wrapper.text()).toContain('系统初始化')
    expect(wrapper.text()).toContain('管理员账号')
    expect(wrapper.text()).toContain('管理员姓名')
    expect(wrapper.find('[data-prop="email"]').exists()).toBe(true)
    expect(wrapper.text()).toContain('完成初始化')
  })

  it('redirects users to login when the system is already initialized', async () => {
    setupApiMock.status.mockResolvedValue({
      data: { is_installed: true, needs_setup: false }
    })

    const wrapper = mountSetupPage()
    await flushPromises()

    expect(wrapper.text()).toContain('系统已完成初始化')
    await wrapper.findAll('button').find((button) => button.text().includes('进入登录页')).trigger('click')

    expect(routerPush).toHaveBeenCalledWith('/login')
  })

  it('initializes the administrator account and returns to login', async () => {
    setupApiMock.status.mockResolvedValue({
      data: { is_installed: false, needs_setup: true }
    })
    setupApiMock.initialize.mockResolvedValue({ success: true })
    const wrapper = mountSetupPage()
    await flushPromises()

    wrapper.vm.form.username = 'admin_user'
    wrapper.vm.form.real_name = 'Admin User'
    wrapper.vm.form.email = 'admin@example.com'
    wrapper.vm.form.password = 'admin123'
    wrapper.vm.form.confirmPassword = 'admin123'
    await wrapper.findAll('button').find((button) => button.text().includes('完成初始化')).trigger('click')
    await flushPromises()

    expect(setupApiMock.initialize).toHaveBeenCalledWith({
      username: 'admin_user',
      real_name: 'Admin User',
      email: 'admin@example.com',
      password: 'admin123'
    })
    expect(messageMock.success).toHaveBeenCalledWith('系统初始化成功，请登录管理员账号')
    expect(routerPush).toHaveBeenCalledWith('/login')
  })

  it('shows an error when initialization fails', async () => {
    setupApiMock.status.mockResolvedValue({
      data: { is_installed: false, needs_setup: true }
    })
    setupApiMock.initialize.mockRejectedValue(new Error('初始化失败'))
    const wrapper = mountSetupPage()
    await flushPromises()

    wrapper.vm.form.username = 'admin_user'
    wrapper.vm.form.real_name = 'Admin User'
    wrapper.vm.form.email = 'admin@example.com'
    wrapper.vm.form.password = 'admin123'
    wrapper.vm.form.confirmPassword = 'admin123'
    await wrapper.findAll('button').find((button) => button.text().includes('完成初始化')).trigger('click')
    await flushPromises()

    expect(messageMock.error).toHaveBeenCalledWith('初始化失败')
    expect(routerPush).not.toHaveBeenCalledWith('/login')
  })
})
