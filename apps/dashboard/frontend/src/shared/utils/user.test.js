import { describe, expect, it, vi } from 'vitest'

import { clearStoredSession } from '@/shared/utils/user'

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
