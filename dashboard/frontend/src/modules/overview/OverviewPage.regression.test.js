import { flushPromises, mount } from '@vue/test-utils'
import { beforeEach, describe, expect, it, vi } from 'vitest'

const getOverviewMock = vi.hoisted(() => vi.fn())
const messageErrorMock = vi.hoisted(() => vi.fn())

vi.mock('@/shared/services', () => ({
  overviewApi: {
    getOverview: getOverviewMock
  }
}))

vi.mock('element-plus', () => ({
  ElMessage: {
    error: messageErrorMock
  },
  ElTag: { name: 'ElTag', template: '<span><slot /></span>' },
  ElTable: { name: 'ElTable', props: ['data'], template: '<div><slot /></div>' },
  ElTableColumn: { name: 'ElTableColumn', template: '<div><slot /></div>' },
  ElProgress: { name: 'ElProgress', template: '<div />' }
}))

import OverviewPage from './OverviewPage.vue'

describe('OverviewPage competition redesign', () => {
  beforeEach(() => {
    getOverviewMock.mockReset()
    messageErrorMock.mockReset()
    getOverviewMock.mockResolvedValue({
      data: {
        assets_total: 128,
        runs_total: 42,
        runs_by_status: {
          queued: 2,
          running: 1,
          success: 36,
          error: 3
        },
        fault_distribution: { normal: 21, motor: 13, sensor: 8 },
        recent_runs: []
      }
    })
  })

  it('renders the competition-grade command deck narrative', async () => {
    const wrapper = mount(OverviewPage)
    await flushPromises()

    const text = wrapper.text()
    expect(text).toContain('系统在线')
    expect(text).toContain('诊断链路')
    expect(text).toContain('关键任务流')
    expect(text).toContain('累计资产数')
    expect(text).toContain('累计任务数')
    expect(text).toContain('成功任务数')
    expect(text).toContain('当前主故障')
    expect(text).toContain('128')
    expect(text).toContain('42')
    expect(text).toContain('36')
    expect(text).toContain('电机故障')
    expect(text).toContain('AI 解读')
    expect(text).not.toContain('故障识别能力')
    expect(text).not.toContain('诊断闭环效率')
    expect(text).not.toContain('当前异常检出')
    expect(text).not.toContain('把算法工作流讲清楚')
    expect(text).not.toContain('以时间顺序呈现最近的系统动作')
    expect(text).not.toContain('聚焦系统在链路联动')
  })
})
