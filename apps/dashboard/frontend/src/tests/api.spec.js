import { beforeEach, describe, expect, it, vi } from 'vitest'

const mocks = vi.hoisted(() => ({
  post: vi.fn()
}))

vi.mock('@/utils/request', () => ({
  default: {
    post: mocks.post
  }
}))

import { runApi } from '@/utils/api'

describe('runApi.createOfflineDetection', () => {
  beforeEach(() => {
    mocks.post.mockReset()
  })

  it('透传 session_id 创建离线分类任务', () => {
    runApi.createOfflineDetection({ sessionId: 'TS-001', model: 'lstm' })

    expect(mocks.post).toHaveBeenCalledWith('/runs', {
      task: 'offline_multiclass',
      session_id: 'TS-001',
      model: 'lstm'
    })
  })
})
