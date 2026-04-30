import { describe, expect, it } from 'vitest'

import router from '@/app/router'

describe('auth router', () => {
  it('register route is available without authentication', () => {
    const route = router.getRoutes().find((item) => item.path === '/register')

    expect(route).toBeTruthy()
    expect(route.meta?.requiresAuth).toBe(false)
  })
})
