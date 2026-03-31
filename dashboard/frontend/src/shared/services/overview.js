import request from '@/utils/request'

export const overviewApi = {
  getOverview: () => request.get('/overview')
}
