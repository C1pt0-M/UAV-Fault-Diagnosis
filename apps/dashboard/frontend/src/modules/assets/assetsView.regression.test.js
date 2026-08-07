import { describe, expect, it } from 'vitest'

import { assetKindTabs, assetsCopy, getAssetCategoryMeta } from './assetsView'

describe('assets copy regression', () => {
  it('keeps the prepare-already-done message complete and readable', () => {
    expect(assetsCopy.prepareAlreadyDone).toBe('该资产已完成预处理，无需重复执行')
  })
})

describe('asset category tabs', () => {
  it('exposes ts/csv/ulg classification tabs', () => {
    expect(assetKindTabs.map((item) => item.value)).toEqual(['all', 'ts', 'csv', 'ulg'])
  })

  it('maps source_category to readable labels', () => {
    expect(getAssetCategoryMeta({ source_category: 'ts' }).label).toBe('TS 会话')
    expect(getAssetCategoryMeta({ source_category: 'ulg' }).label).toBe('ULG')
    expect(getAssetCategoryMeta({ source_category: 'csv' }).label).toBe('CSV')
    expect(getAssetCategoryMeta({}).label).toBe('')
  })
})
