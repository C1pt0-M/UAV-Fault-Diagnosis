import { describe, expect, it } from 'vitest'

import { assetsCopy } from './assetsView'

describe('assets copy regression', () => {
  it('keeps the prepare-already-done message complete and readable', () => {
    expect(assetsCopy.prepareAlreadyDone).toBe('该资产已完成预处理，无需重复执行')
  })
})
