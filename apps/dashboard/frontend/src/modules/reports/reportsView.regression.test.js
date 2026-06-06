import { describe, expect, it } from 'vitest'

import {
  createReportActionItems,
  createReportSummaryItems,
  getReportLabelText,
  getReportModelDisplayName,
  reportsCopy
} from './reportsView'

describe('reports workspace copy regression', () => {
  it('uses the refined report-center language', () => {
    expect(reportsCopy.title).toBe('诊断报告中心')
    expect(reportsCopy.description).toContain('诊断结论')
    expect(reportsCopy.description).toContain('处置建议')
    expect(reportsCopy.emptyText).toBe('暂无诊断报告')
    expect(reportsCopy.dataQualitySection).toBe('数据质量')
    expect(reportsCopy.traceabilitySection).toBe('模型与规则追溯')
    expect(reportsCopy.reportConfidenceSection).toBe('报告可信度')
    expect(reportsCopy.reviewSection).toBe('人工复核')
    expect(reportsCopy.riskLabel).toBe('风险等级')
  })

  it('maps report model and diagnosis labels into display text', () => {
    expect(getReportModelDisplayName('transfer3_cnnlstm_normalproto')).toBe('CNN-LSTM')
    expect(getReportModelDisplayName('online_detector_kg_mstca_rfly')).toBe('KG-MSTCA')
    expect(getReportModelDisplayName('custom_model_alpha')).toBe('custom model alpha')

    expect(getReportLabelText('motor')).toBe('电机故障')
    expect(getReportLabelText('sensor')).toBe('传感器故障')
    expect(getReportLabelText('unknown_fault')).toBe('unknown_fault')
  })

  it('creates current-list summary items from existing reports', () => {
    const items = createReportSummaryItems([
      { id: 1, prediction_label: 'motor', confidence: 0.9 },
      { id: 2, prediction_label: 'sensor', confidence: 0.8 }
    ], { total: 35 })

    expect(items).toEqual([
      { label: '报告总数', value: 35, tone: 'primary' },
      { label: '当前页报告', value: 2, tone: 'primary' },
      { label: '异常报告', value: 2, tone: 'warning' },
      { label: '可 AI 解读', value: 2, tone: 'ai' }
    ])
  })

  it('keeps the compact report action group stable', () => {
    expect(createReportActionItems().map((item) => item.label)).toEqual([
      '详情',
      'AI 解读',
      '删除'
    ])
  })
})
