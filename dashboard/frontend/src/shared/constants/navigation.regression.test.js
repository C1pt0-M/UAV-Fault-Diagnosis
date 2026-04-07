import { describe, expect, it } from 'vitest'

import { navigationItems, pageMeta, resolvePageMeta } from './navigation'

describe('workspace navigation regression', () => {
  it('exposes separate entries for assets center and diagnosis reports', () => {
    const assetsItem = navigationItems.find((item) => item.key === 'assets')
    const reportsItem = navigationItems.find((item) => item.key === 'reports')

    expect(assetsItem).toMatchObject({
      path: '/assets',
      label: '资产中心'
    })
    expect(reportsItem).toMatchObject({
      path: '/reports',
      label: '诊断报告'
    })
  })

  it('returns readable page meta for the assets and reports pages', () => {
    expect(pageMeta['/assets']).toMatchObject({
      title: '资产中心'
    })
    expect(pageMeta['/reports']).toMatchObject({
      title: '诊断报告'
    })
    expect(resolvePageMeta('/missing').title).toBe('工作台')
  })
})
