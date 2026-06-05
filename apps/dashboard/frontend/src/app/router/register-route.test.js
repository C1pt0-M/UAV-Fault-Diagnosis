import { describe, expect, it, vi } from 'vitest'

vi.mock('@/shared/services', () => ({
  authApi: { profile: vi.fn() }
}))

import router from '@/app/router'

describe('auth router', () => {
  it('exposes register as a normal public auth route', () => {
    const route = router.getRoutes().find((item) => item.path === '/register')

    expect(route).toBeTruthy()
    expect(route.meta?.requiresAuth).toBe(false)
    expect(route.beforeEnter).toBeUndefined()
    expect(typeof route.components?.default).toBe('function')
  })
})
