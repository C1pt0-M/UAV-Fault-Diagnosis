import { resetRuntimeCapabilities } from '@/shared/state/runtimeCapabilities'

export function parseStoredUser(rawValue) {
  if (!rawValue) return null

  try {
    const parsed = JSON.parse(rawValue)
    return parsed && typeof parsed === 'object' ? parsed : null
  } catch {
    return null
  }
}

export function getDisplayName(user) {
  return user?.real_name || user?.username || '演示用户'
}

export function getRoleLabel(user) {
  if (user?.role === 'admin') return '管理员'
  return user?.role || '演示账号'
}

export function clearStoredSession(storage = localStorage) {
  resetRuntimeCapabilities()
  storage.removeItem('token')
  storage.removeItem('user')
}

export function storeSession(token, user, storage = localStorage) {
  if (storage.getItem('token') !== token) resetRuntimeCapabilities()
  storage.setItem('token', token)
  storage.setItem('user', JSON.stringify(user))
}
