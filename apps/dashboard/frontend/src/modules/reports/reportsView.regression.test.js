import { describe, expect, it } from 'vitest'

import { reportsCopy } from './reportsView'

describe('reports workspace copy regression', () => {
  it('uses the refined report-center language', () => {
    expect(reportsCopy.title).toBe('诊断报告中心')
    expect(reportsCopy.description).toContain('诊断结论')
    expect(reportsCopy.description).toContain('处置建议')
  })
})
