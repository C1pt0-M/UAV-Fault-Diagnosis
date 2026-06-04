import request from '@/utils/request'

export const authApi = {
  login: (payload) => request.post('/auth/login', payload),
  register: (payload) => request.post('/auth/register', payload),
  logout: () => request.post('/auth/logout', {}),
  profile: () => request.get('/auth/profile')
}

export const setupApi = {
  status: () => request.get('/setup/status'),
  initialize: (payload) => request.post('/setup/initialize', payload)
}
