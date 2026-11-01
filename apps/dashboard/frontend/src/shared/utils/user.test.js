import { describe, expect, it, vi } from 'vitest'

import { clearStoredSession, storeSession } from '@/shared/utils/user'

describe('clearStoredSession', () => {
  it('removes token and user from storage', () => {
    const storage = {
      removeItem: vi.fn()
    }

    clearStoredSession(storage)

    expect(storage.removeItem).toHaveBeenCalledWith('token')
    expect(storage.removeItem).toHaveBeenCalledWith('user')
  })
})

describe('storeSession', () => {
  it('stores token and user JSON in storage', () => {
    const storage = {
      getItem: vi.fn().mockReturnValue(null),
      setItem: vi.fn()
    }

    storeSession('token-1', { username: 'admin' }, storage)

    expect(storage.setItem).toHaveBeenCalledWith('token', 'token-1')
    expect(storage.setItem).toHaveBeenCalledWith('user', '{"username":"admin"}')
  })
})
