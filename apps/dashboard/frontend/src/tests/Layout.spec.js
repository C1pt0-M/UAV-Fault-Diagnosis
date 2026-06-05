import { mount } from '@vue/test-utils'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import { navigationItems, resolvePageMeta } from '@/shared/constants/navigation'

const routeState = vi.hoisted(() => ({ path: '/reports' }))
const routerPush = vi.hoisted(() => vi.fn())
const logoutMock = vi.hoisted(() => vi.fn().mockResolvedValue(undefined))

vi.mock('vue-router', () => ({
  useRoute: () => routeState,
  useRouter: () => ({
    push: routerPush
  })
}))

vi.mock('@/shared/services', () => ({
  authApi: {
    logout: logoutMock
  }
}))

import Layout from '@/components/Layout.vue'

const SlotStub = { template: '<div><slot /></div>' }

function mountLayout() {
  return mount(Layout, {
    global: {
      stubs: {
        RouterLink: SlotStub,
        RouterView: true,
        ElIcon: SlotStub
      }
    }
  })
}

describe('Layout shell', () => {
  beforeEach(() => {
    localStorage.clear()
    routeState.path = '/reports'
    routerPush.mockReset()
    logoutMock.mockClear()
  })

  it('falls back to the demo user and renders the current primary navigation', () => {
    localStorage.setItem('user', '{bad-json')

    const wrapper = mountLayout()
    const text = wrapper.text()

    navigationItems.forEach((item) => {
      expect(text).toContain(item.label)
    })
    expect(wrapper.find('img.sidebar-account__logo').exists()).toBe(true)
    expect(wrapper.find('.sidebar-account').exists()).toBe(true)
    expect(text).toContain('基于数据驱动的无人机故障智能诊断系统')
    expect(text).toContain('演示账号 · 演示用户')
    expect(text).toContain(resolvePageMeta('/reports').title)
  })

  it('clears session state and returns to login after logout', async () => {
    localStorage.setItem('token', 'demo-token')
    localStorage.setItem('user', JSON.stringify({ username: 'demo', role: 'admin' }))

    const wrapper = mountLayout()
    await wrapper.get('button.sidebar-footer__logout').trigger('click')

    expect(logoutMock).toHaveBeenCalledTimes(1)
    expect(localStorage.getItem('token')).toBeNull()
    expect(localStorage.getItem('user')).toBeNull()
    expect(routerPush).toHaveBeenCalledWith('/login')
  })
})
