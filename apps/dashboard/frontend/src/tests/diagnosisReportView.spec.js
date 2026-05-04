import { describe, expect, it } from 'vitest'

import { buildDiagnosisReportView } from '@/utils/diagnosisReportView'

describe('buildDiagnosisReportView', () => {
  it('把 motor 故障转换成用户可读的中文摘要', () => {
    const view = buildDiagnosisReportView({
      id: 12,
      prediction_label: 'motor',
      confidence: 0.93,
      probs: {
        motor: 0.93,
        propeller: 0.04,
        normal: 0.02,
        sensor: 0.01
      },
      summary: {
        severity: {
          level: 'heavy',
          score: 0.88,
          method: 'rule_v1',
          evidence: [
            {
              metric: 'motor_imbalance_ratio',
              status: 'used',
              value: 0.41,
              score: 1,
              required: true
            },
            {
              metric: 'esc_rpm_imbalance_ratio',
              status: 'used',
              value: 0.33,
              score: 0.92,
              required: true
            }
          ]
        },
        key_params: {
          motor_imbalance_ratio: 0.41,
          esc_rpm_imbalance_ratio: 0.33,
          battery_remaining_min: 0.76
        },
        recommendations: [
          {
            title: '检查电机温度、振动和相电流是否异常。',
            priority: 'P1',
            reason_codes: ['motor_imbalance_ratio']
          },
          {
            title: '必要时停飞并安排动力系统复检。',
            priority: 'P0',
            reason_codes: ['esc_rpm_imbalance_ratio']
          }
        ],
        topk: [
          { label: 'motor', prob: 0.93 },
          { label: 'propeller', prob: 0.04 },
          { label: 'normal', prob: 0.02 }
        ]
      }
    })

    expect(view.labelText).toBe('电机故障')
    expect(view.headline).toContain('电机故障')
    expect(view.headline).toContain('高风险')
    expect(view.riskText).toContain('建议立即复核')
    expect(view.actionSummary).toContain('停飞')
    expect(view.whyItems[0].title).toContain('电机输出均衡性')
    expect(view.actionItems[0].priorityText).toBe('优先处理')
  })

  it('把 normal 结果转换成清晰易懂的正常结论', () => {
    const view = buildDiagnosisReportView({
      id: 18,
      prediction_label: 'normal',
      confidence: 0.82,
      probs: {
        normal: 0.82,
        motor: 0.1,
        propeller: 0.05,
        sensor: 0.03
      },
      summary: {
        severity: {
          level: 'none',
          score: 0,
          method: 'rule_v1',
          evidence: []
        },
        key_params: {},
        recommendations: [
          '继续执行例行检查并关注飞行前后巡检记录。'
        ],
        topk: [
          { label: 'normal', prob: 0.82 },
          { label: 'motor', prob: 0.1 },
          { label: 'propeller', prob: 0.05 }
        ]
      }
    })

    expect(view.labelText).toBe('状态正常')
    expect(view.headline).toContain('状态正常')
    expect(view.riskText).toContain('风险可控')
    expect(view.actionItems[0].title).toContain('例行检查')
    expect(view.whyItems[0].title).toContain('模型判断')
  })
})
