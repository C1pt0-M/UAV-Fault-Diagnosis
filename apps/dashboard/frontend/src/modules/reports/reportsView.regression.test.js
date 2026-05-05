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
      { id: 1, confidence: 0.9 },
      { id: 2, confidence: 0.8 }
    ])

    expect(items).toEqual([
      { label: '当前列表报告', value: 2, tone: 'primary' },
      { label: '平均置信度', value: '85.00%', tone: 'ok' },
      { label: '可下载报告', value: 2, tone: 'ok' },
      { label: '可 AI 解读', value: 2, tone: 'ai' }
    ])
  })

  it('keeps the compact report action group stable', () => {
    expect(createReportActionItems().map((item) => item.label)).toEqual([
      '详情',
      'AI 解读',
      '下载',
      '删除'
    ])
  })
})
