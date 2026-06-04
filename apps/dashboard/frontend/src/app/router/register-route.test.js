import { beforeEach, describe, expect, it, vi } from 'vitest'

const setupApiMock = vi.hoisted(() => ({
  status: vi.fn()
}))

vi.mock('@/shared/services', () => ({
  authApi: { profile: vi.fn() },
  setupApi: setupApiMock
}))

import router from '@/app/router'

describe('auth router', () => {
  beforeEach(() => {
    setupApiMock.status.mockReset()
  })

  it('keeps register as a compatibility route without exposing a register page', () => {
    const route = router.getRoutes().find((item) => item.path === '/register')

    expect(route).toBeTruthy()
    expect(route.meta?.requiresAuth).toBe(false)
    expect(route.components?.default?.name).toBe('RegisterRedirect')
  })

  it('redirects register to setup when initialization is still needed', async () => {
    setupApiMock.status.mockResolvedValue({ data: { needs_setup: true } })
    const route = router.getRoutes().find((item) => item.path === '/register')
    const next = vi.fn()

    await route.beforeEnter({}, {}, next)

    expect(next).toHaveBeenCalledWith('/setup')
  })

  it('redirects register to login after initialization', async () => {
    setupApiMock.status.mockResolvedValue({ data: { needs_setup: false } })
    const route = router.getRoutes().find((item) => item.path === '/register')
    const next = vi.fn()

    await route.beforeEnter({}, {}, next)

    expect(next).toHaveBeenCalledWith('/login')
  })
})
