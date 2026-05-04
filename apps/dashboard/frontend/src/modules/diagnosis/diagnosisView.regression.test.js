import { describe, expect, it } from 'vitest'

import { diagnosisCopy } from './diagnosisView'

describe('diagnosis command deck copy regression', () => {
  it('exposes the command-center labels used by the redesigned diagnosis page', () => {
    expect(diagnosisCopy.commandTitle).toBe('离线诊断任务指挥台')
    expect(diagnosisCopy.resultTitle).toBe('诊断结果主舞台')
  })
})
