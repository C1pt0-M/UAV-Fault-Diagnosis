import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import request from '@/utils/request'
import { clearStoredSession, storeSession } from '@/shared/utils/user'
import { applyRuntimeCapabilities, runtimeCapabilities } from '@/shared/state/runtimeCapabilities'

beforeEach(() => {
  localStorage.clear()
  clearStoredSession()
  window.history.replaceState({}, '', '/login')
})

afterEach(() => {
  vi.unstubAllGlobals()
  window.history.replaceState({}, '', '/')
})

describe('runtime capabilities follow the authenticated session', () => {
  it('clears capabilities on a current-session 401', async () => {
    storeSession('token', { username: 'admin' })
    applyRuntimeCapabilities({ ai: true, telemetry: true })
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ status: 401 }))
    await expect(request.get('/auth/profile')).rejects.toThrow('登录已过期')
    expect(localStorage.getItem('token')).toBeNull()
    expect(runtimeCapabilities.ai).toBe(false)
    expect(runtimeCapabilities.telemetry).toBe(false)
  })

  it('does not clear a new session for an old in-flight request 401', async () => {
    let complete
    storeSession('old', { username: 'old' })
    vi.stubGlobal('fetch', vi.fn().mockReturnValue(new Promise(resolve => { complete = resolve })))
    const pending = request.get('/auth/profile')
    storeSession('new', { username: 'new' })
    applyRuntimeCapabilities({ ai: true, telemetry: false })
    complete({ status: 401 })
    await expect(pending).rejects.toThrow('登录已过期')
    expect(localStorage.getItem('token')).toBe('new')
    expect(runtimeCapabilities.ai).toBe(true)
    expect(runtimeCapabilities.telemetry).toBe(false)
  })

  it('resets capabilities on token replacement instead of persisting them', () => {
    storeSession('old', { username: 'old' })
    applyRuntimeCapabilities({ ai: true, telemetry: true })
    storeSession('new', { username: 'new' })
    expect(runtimeCapabilities.ai).toBe(false)
    expect(runtimeCapabilities.telemetry).toBe(false)
    expect(Object.keys(localStorage).sort()).toEqual(['token', 'user'])
  })
})
