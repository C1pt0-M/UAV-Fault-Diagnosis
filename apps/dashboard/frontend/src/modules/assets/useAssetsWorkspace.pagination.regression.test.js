import { defineComponent, nextTick } from 'vue'
import { flushPromises, mount } from '@vue/test-utils'
import { beforeEach, describe, expect, it, vi } from 'vitest'

const routerPush = vi.hoisted(() => vi.fn())

const assetApiMocks = vi.hoisted(() => ({
  list: vi.fn(),
  upload: vi.fn(),
  prepare: vi.fn(),
  download: vi.fn(),
  remove: vi.fn()
}))

const analysisApiMocks = vi.hoisted(() => ({
  fields: vi.fn(),
  timeseries: vi.fn(),
  faultProbCurve: vi.fn()
}))

const runApiMocks = vi.hoisted(() => ({
  detail: vi.fn()
}))

vi.mock('vue-router', () => ({
  useRouter: () => ({
    push: routerPush
  })
}))

vi.mock('element-plus', () => ({
  ElMessage: {
    error: vi.fn(),
    success: vi.fn()
  },
  ElMessageBox: {
    confirm: vi.fn()
  }
}))

const { ElMessageBox } = await import('element-plus')

vi.mock('@/shared/services', () => ({
  assetApi: assetApiMocks,
  analysisApi: analysisApiMocks,
  runApi: runApiMocks
}))

import { useAssetsWorkspace } from './composables/useAssetsWorkspace'

const Harness = defineComponent({
  setup() {
    return useAssetsWorkspace()
  },
  template: '<div />'
})

function makeAssets(count) {
  return Array.from({ length: count }, (_, index) => ({
    id: index + 1,
    filename: `asset-${index + 1}.csv`,
    kind: 'csv',
    size_bytes: 1024,
    created_at: '2026-04-07 18:00:00',
    meta: {}
  }))
}

describe('useAssetsWorkspace pagination regression', () => {
  beforeEach(() => {
    routerPush.mockReset()
    assetApiMocks.list.mockReset()
    assetApiMocks.upload.mockReset()
    assetApiMocks.prepare.mockReset()
    assetApiMocks.download.mockReset()
    assetApiMocks.remove.mockReset()
    analysisApiMocks.fields.mockReset()
    analysisApiMocks.timeseries.mockReset()
    analysisApiMocks.faultProbCurve.mockReset()
    runApiMocks.detail.mockReset()

    assetApiMocks.list.mockResolvedValue({
      data: makeAssets(12)
    })
    analysisApiMocks.fields.mockResolvedValue({
      data: {
        field_groups: [],
        annotations: [],
        time: {
          min: 0,
          max: 10
        }
      }
    })
    analysisApiMocks.timeseries.mockResolvedValue({
      data: {
        x: [],
        series: []
      }
    })
    analysisApiMocks.faultProbCurve.mockResolvedValue({
      data: []
    })
  })

  it('paginates assets locally with ten items per page and resets to page one on kind change', async () => {
    assetApiMocks.list.mockResolvedValue({
      data: makeAssets(12).reverse()
    })

    const wrapper = mount(Harness)
    await flushPromises()

    expect(wrapper.vm.pageSize).toBe(10)
    expect(wrapper.vm.currentPage).toBe(1)
    expect(wrapper.vm.totalAssets).toBe(12)
    expect(wrapper.vm.paginatedAssets).toHaveLength(10)
    expect(wrapper.vm.paginatedAssets[0].filename).toBe('asset-1.csv')
    expect(wrapper.vm.paginatedAssets[9].filename).toBe('asset-10.csv')

    wrapper.vm.setCurrentPage(2)
    await nextTick()

    expect(wrapper.vm.currentPage).toBe(2)
    expect(wrapper.vm.paginatedAssets).toHaveLength(2)
    expect(wrapper.vm.paginatedAssets[0].filename).toBe('asset-11.csv')
    expect(wrapper.vm.paginatedAssets[1].filename).toBe('asset-12.csv')

    wrapper.vm.activeKind = 'csv'
    await nextTick()

    expect(wrapper.vm.currentPage).toBe(1)
  })

  it('keeps the current page after deleting an item when the page still has data', async () => {
    const beforeDelete = makeAssets(21)
    const afterDelete = beforeDelete.filter((item) => item.id !== 11)
    assetApiMocks.list
      .mockResolvedValueOnce({ data: beforeDelete })
      .mockResolvedValueOnce({ data: afterDelete })
    assetApiMocks.remove.mockResolvedValue({})
    ElMessageBox.confirm.mockResolvedValue()

    const wrapper = mount(Harness)
    await flushPromises()

    wrapper.vm.setCurrentPage(2)
    await nextTick()
    expect(wrapper.vm.currentPage).toBe(2)

    await wrapper.vm.removeAsset(beforeDelete[10])
    await flushPromises()

    expect(assetApiMocks.remove).toHaveBeenCalledWith(11)
    expect(wrapper.vm.currentPage).toBe(2)
    expect(wrapper.vm.paginatedAssets[0].id).toBe(12)
  })

  it('moves back one page after deleting the only item on the last page', async () => {
    const beforeDelete = makeAssets(21)
    const afterDelete = beforeDelete.filter((item) => item.id !== 21)
    assetApiMocks.list
      .mockResolvedValueOnce({ data: beforeDelete })
      .mockResolvedValueOnce({ data: afterDelete })
    assetApiMocks.remove.mockResolvedValue({})
    ElMessageBox.confirm.mockResolvedValue()

    const wrapper = mount(Harness)
    await flushPromises()

    wrapper.vm.setCurrentPage(3)
    await nextTick()
    expect(wrapper.vm.currentPage).toBe(3)

    await wrapper.vm.removeAsset(beforeDelete[20])
    await flushPromises()

    expect(assetApiMocks.remove).toHaveBeenCalledWith(21)
    expect(wrapper.vm.currentPage).toBe(2)
    expect(wrapper.vm.paginatedAssets).toHaveLength(10)
  })
})
