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
