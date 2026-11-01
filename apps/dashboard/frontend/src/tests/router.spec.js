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
import { applyRuntimeCapabilities, resetRuntimeCapabilities, runtimeCapabilities } from '@/shared/state/runtimeCapabilities'
import { clearStoredSession } from '@/shared/utils/user'

async function runGuard(to, from = { path: '/' }) {
  const next = vi.fn()
  await routerState.guard(to, from, next)
  return next
}

describe('router guard', () => {
  beforeEach(() => {
    localStorage.clear()
    resetRuntimeCapabilities()
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
      data: { username: 'admin', role: 'admin', real_name: '管理员' },
      capabilities: { ai: true, telemetry: true }
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

  it.each(['/ai', '/monitoring'])('blocks disabled optional deep link %s after authentication', async (path) => {
    authApiMock.profile.mockResolvedValue({
      data: { username: 'admin', role: 'admin' }, capabilities: { ai: false, telemetry: false }
    })
    localStorage.setItem('token', 'token')
    localStorage.setItem('user', JSON.stringify({ username: 'admin', role: 'admin' }))
    const next = await runGuard({ path, meta: { requiresAuth: true } })
    expect(next).toHaveBeenCalledWith('/')
    expect(elementMessage.info).toHaveBeenCalledWith(expect.stringContaining('未启用或能力信息不可用'))
    expect(localStorage.getItem('token')).toBe('token')
  })

  it.each([undefined, {}, { ai: 'true', telemetry: 1 }])('fails closed for missing or malformed declarations %j', async (capabilities) => {
    applyRuntimeCapabilities({ ai: true, telemetry: true })
    authApiMock.profile.mockResolvedValue({ data: { username: 'admin', role: 'admin' }, capabilities })
    localStorage.setItem('token', 'token')
    localStorage.setItem('user', JSON.stringify({ username: 'admin', role: 'admin' }))
    const next = await runGuard({ path: '/ai', meta: { requiresAuth: true } })
    expect(next).toHaveBeenCalledWith('/')
    expect(runtimeCapabilities.ai).toBe(false)
    expect(runtimeCapabilities.telemetry).toBe(false)
  })

  it('does not restore session or capabilities from a profile returned after logout', async () => {
    let complete
    authApiMock.profile.mockReturnValue(new Promise(resolve => { complete = resolve }))
    localStorage.setItem('token', 'token')
    localStorage.setItem('user', JSON.stringify({ username: 'admin', role: 'admin' }))
    const navigation = runGuard({ path: '/ai', meta: { requiresAuth: true } })
    clearStoredSession()
    complete({ data: { username: 'admin', role: 'admin' }, capabilities: { ai: true, telemetry: true } })
    await navigation
    expect(localStorage.getItem('token')).toBeNull()
    expect(runtimeCapabilities.ai).toBe(false)
  })

  it('validates a new token separately and ignores an old pending profile', async () => {
    let completeOld, completeNew
    authApiMock.profile
      .mockReturnValueOnce(new Promise(resolve => { completeOld = resolve }))
      .mockReturnValueOnce(new Promise(resolve => { completeNew = resolve }))
    localStorage.setItem('token', 'old-token')
    localStorage.setItem('user', JSON.stringify({ username: 'old', role: 'admin' }))
    const oldNavigation = runGuard({ path: '/reports', meta: { requiresAuth: true } })
    localStorage.setItem('token', 'new-token')
    localStorage.setItem('user', JSON.stringify({ username: 'new', role: 'user' }))
    const newNavigation = runGuard({ path: '/reports', meta: { requiresAuth: true } })
    completeOld({ data: { username: 'old', role: 'admin' }, capabilities: { ai: true, telemetry: true } })
    await oldNavigation
    expect(localStorage.getItem('token')).toBe('new-token')
    expect(runtimeCapabilities.ai).toBe(false)
    completeNew({ data: { username: 'new', role: 'user' }, capabilities: { ai: false, telemetry: true } })
    const next = await newNavigation
    expect(authApiMock.profile).toHaveBeenCalledTimes(2)
    expect(next).toHaveBeenCalledWith()
    expect(JSON.parse(localStorage.getItem('user')).username).toBe('new')
    expect(runtimeCapabilities.ai).toBe(false)
    expect(runtimeCapabilities.telemetry).toBe(true)
  })

  it('redirects authenticated users away from public auth pages after validation', async () => {
    authApiMock.profile.mockResolvedValue({
      data: { username: 'admin', role: 'admin' },
      capabilities: { ai: true, telemetry: true }
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
