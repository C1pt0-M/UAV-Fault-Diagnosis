import { beforeEach, describe, expect, it, vi } from 'vitest'

const routerState = vi.hoisted(() => ({ guard: null, routes: [] }))
const elementMessage = vi.hoisted(() => ({ warning: vi.fn(), error: vi.fn(), info: vi.fn() }))
const authApiMock = vi.hoisted(() => ({
  profile: vi.fn()
}))
const setupApiMock = vi.hoisted(() => ({
  status: vi.fn()
}))

vi.mock('../components/Layout.vue', () => ({
  default: { name: 'LayoutStub', template: '<div />' }
}))

vi.mock('element-plus', () => ({ ElMessage: elementMessage }))
vi.mock('@/shared/services', () => ({
  authApi: authApiMock,
  setupApi: setupApiMock
}))
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

async function runGuard(to, from = { path: '/' }) {
  const next = vi.fn()
  await routerState.guard(to, from, next)
  return next
}

describe('router guard', () => {
  beforeEach(() => {
    localStorage.clear()
    elementMessage.warning.mockReset()
    elementMessage.error.mockReset()
    elementMessage.info.mockReset()
    authApiMock.profile.mockReset()
    setupApiMock.status.mockReset()
  })

  it('defines normal system routes without competition model comparison', () => {
    const protectedRoot = routerState.routes.find((route) => route.path === '/')
    const childPaths = protectedRoot.children.map((route) => route.path)

    expect(childPaths).toEqual(['', 'monitoring', 'diagnosis', 'assets', 'reports', 'ai', 'settings'])
    expect(childPaths).not.toContain('model-compare')
  })

  it('redirects unauthenticated users to login for protected pages', async () => {
    const next = await runGuard(
      { path: '/monitoring', meta: { requiresAuth: true } },
      { path: '/' }
    )

    expect(elementMessage.warning).toHaveBeenCalledWith('请先登录')
    expect(next).toHaveBeenCalledWith('/login')
    expect(authApiMock.profile).not.toHaveBeenCalled()
  })

  it('validates stored sessions before entering protected pages', async () => {
    authApiMock.profile.mockResolvedValue({
      data: { username: 'admin', role: 'admin', real_name: '管理员' }
    })
    localStorage.setItem('token', 'token')
    localStorage.setItem('user', JSON.stringify({ username: 'admin', role: 'admin' }))

    const next = await runGuard(
      { path: '/monitoring', meta: { requiresAuth: true } },
      { path: '/' }
    )

    expect(authApiMock.profile).toHaveBeenCalledTimes(1)
    expect(JSON.parse(localStorage.getItem('user'))).toEqual({
      username: 'admin',
      role: 'admin',
      real_name: '管理员'
    })
    expect(next).toHaveBeenCalledWith()
  })

  it('clears invalid stored sessions and returns protected navigation to login', async () => {
    authApiMock.profile.mockRejectedValue(new Error('invalid token'))
    localStorage.setItem('token', 'token')
    localStorage.setItem('user', JSON.stringify({ username: 'admin', role: 'admin' }))

    const next = await runGuard(
      { path: '/reports', meta: { requiresAuth: true } },
      { path: '/' }
    )

    expect(localStorage.getItem('token')).toBeNull()
    expect(localStorage.getItem('user')).toBeNull()
    expect(elementMessage.warning).toHaveBeenCalledWith('登录已过期，请重新登录')
    expect(next).toHaveBeenCalledWith('/login')
  })

  it('redirects authenticated users away from public auth pages after validation', async () => {
    authApiMock.profile.mockResolvedValue({
      data: { username: 'admin', role: 'admin' }
    })
    localStorage.setItem('token', 'token')
    localStorage.setItem('user', JSON.stringify({ username: 'admin', role: 'admin' }))

    const next = await runGuard(
      { path: '/login', meta: { requiresAuth: false } },
      { path: '/' }
    )

    expect(next).toHaveBeenCalledWith('/')
  })
})
