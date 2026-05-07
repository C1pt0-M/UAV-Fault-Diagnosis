import { describe, expect, it } from 'vitest'

import {
  buildRuleExplanationRows,
  buildTraceableEvidenceRows,
  diagnosisCopy,
  formatDiagnosisModelName,
  resolveDiagnosisStage,
  resolveRiskLevel
} from './diagnosisView'

describe('diagnosis command deck copy regression', () => {
  it('exposes the command-center labels used by the redesigned diagnosis page', () => {
    expect(diagnosisCopy.commandTitle).toBe('任务发起 / 输入配置')
    expect(diagnosisCopy.resultTitle).toBe('诊断结果')
    expect(diagnosisCopy.explanationTitle).toBe('规则解释 / 诊断摘要')
    expect(diagnosisCopy.recentRunsTitle).toBe('最近诊断任务 / 报告出口')
  })

  it('formats model names and resolves risk levels for compact diagnosis display', () => {
    expect(formatDiagnosisModelName('online_detector_kg_mstca_rfly')).toBe('KG-MSTCA')
    expect(formatDiagnosisModelName('transfer3_cnnlstm_normalproto')).toBe('CNN-LSTM')
    expect(resolveRiskLevel({ label: 'motor', confidence: 0.89 }).label).toBe('高风险')
    expect(resolveRiskLevel({ label: 'normal', confidence: 0.95 }).label).toBe('风险可控')
  })

  it('maps failed runs to an explicit error stage instead of keeping them running', () => {
    expect(resolveDiagnosisStage({
      currentRun: { id: 16, status: 'error', error: '离线检测失败' },
      resultData: null
    })).toMatchObject({
      key: 'error',
      label: '诊断失败'
    })
  })

  it('builds rule explanations and traceable evidence without overclaiming SHAP', () => {
    const resultData = {
      prediction: { label: 'motor', confidence: 0.89 },
      probs: { motor: 0.89, sensor: 0.11 },
      recommendations: [{ title: '检查电机绕组与供电链路' }]
    }
    const evidenceRows = buildTraceableEvidenceRows({
      resultData,
      probabilityRows: [{ label: 'motor', probability: '89.00%', value: 0.89 }],
      activeSessionId: 'TS-1'
    })
    const ruleRows = buildRuleExplanationRows({
      resultData,
      probabilityRows: [{ label: 'motor', probability: '89.00%', value: 0.89 }],
      activeSessionId: 'TS-1'
    })

    expect(evidenceRows.length).toBeGreaterThan(0)
    expect(ruleRows.map((item) => `${item.title} ${item.detail}`).join(' ')).toContain('规则')
    expect(JSON.stringify({ evidenceRows, ruleRows })).not.toContain('SHAP')
    expect(JSON.stringify({ evidenceRows, ruleRows })).not.toContain('feature importance')
    expect(diagnosisCopy.emptyReports).toBe('暂无历史参考报告')
  })
})
