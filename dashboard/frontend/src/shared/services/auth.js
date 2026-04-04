import request from '@/utils/request'

export const authApi = {
  login: (payload) => request.post('/auth/login', payload),
  register: (payload) => request.post('/auth/register', payload),
  logout: () => request.post('/auth/logout', {}),
  profile: () => request.get('/auth/profile')
}
