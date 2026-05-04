import { beforeEach, describe, expect, it, vi } from 'vitest'

const routerState = vi.hoisted(() => ({ guard: null, routes: [] }))
const elementMessage = vi.hoisted(() => ({ warning: vi.fn(), error: vi.fn(), info: vi.fn() }))

vi.mock('../components/Layout.vue', () => ({
  default: { name: 'LayoutStub', template: '<div />' }
}))

vi.mock('element-plus', () => ({ ElMessage: elementMessage }))
vi.mock('vue-router', () => ({
  createWebHistory: vi.fn(),
  createRouter: vi.fn((options) => {
    routerState.routes = options.routes
    return {
      beforeEach: (guard) => {
        routerState.guard = guard
      }
    }
  })
}))

import '@/router/index.js'

describe('router guard', () => {
  beforeEach(() => {
    localStorage.clear()
    elementMessage.warning.mockReset()
    elementMessage.error.mockReset()
    elementMessage.info.mockReset()
  })

  it('defines normal system routes without competition model comparison', () => {
    const protectedRoot = routerState.routes.find((route) => route.path === '/')
    const childPaths = protectedRoot.children.map((route) => route.path)

    expect(childPaths).toEqual(['', 'monitoring', 'diagnosis', 'assets', 'reports', 'ai', 'settings'])
    expect(childPaths).not.toContain('model-compare')
  })

  it('redirects unauthenticated users to login for protected pages', async () => {
    const next = vi.fn()

    routerState.guard(
      { path: '/monitoring', meta: { requiresAuth: true } },
      { path: '/' },
      next
    )

    expect(elementMessage.warning).toHaveBeenCalledWith('请先登录')
    expect(next).toHaveBeenCalledWith('/login')
  })

  it('redirects authenticated users away from login', async () => {
    localStorage.setItem('token', 'token')
    localStorage.setItem('user', JSON.stringify({ username: 'demo', role: 'admin' }))
    const next = vi.fn()

    routerState.guard(
      { path: '/login', meta: { requiresAuth: false } },
      { path: '/' },
      next
    )

    expect(next).toHaveBeenCalledWith('/')
  })
})
