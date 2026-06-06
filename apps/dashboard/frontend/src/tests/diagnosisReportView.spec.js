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

  it('展示新增的数据质量、追溯、可信度与复核字段', () => {
    const view = buildDiagnosisReportView({
      id: 21,
      prediction_label: 'sensor',
      confidence: 0.88,
      model_name: 'cnn_lstm',
      summary: {
        severity: { level: 'medium', evidence: [] },
        data_quality: {
          level: 'warning',
          sample_count: 0,
          window_count: 0,
          feature_count: 0,
          missing_value_ratio: null,
          invalid_value_ratio: null,
          warnings: ['未记录样本数量，无法完整评估输入覆盖度。']
        },
        traceability: {
          run_id: 9,
          asset_id: 4,
          asset_filename: 'sensor_case.csv',
          input_csv: '/tmp/sensor_case.csv',
          model_name: 'cnn_lstm',
          rules_version: 'rule_v1'
        },
        model_card: {
          task: 'offline_multiclass',
          class_set: ['normal', 'motor', 'sensor'],
          intended_use: '飞行日志 / CSV 回放离线故障诊断辅助分析',
          limitations: []
        },
        report_confidence: {
          level: 'medium',
          score: 0.72,
          reasons: ['输入数据质量信息不完整，建议结合原始日志复核。']
        },
        review: {
          status: 'pending',
          reviewer: '',
          reviewed_at: '',
          conclusion: '',
          maintenance_notes: ''
        }
      }
    })

    expect(view.dataQuality.levelText).toBe('需关注')
    expect(view.traceability.inputCsv).toBe('/tmp/sensor_case.csv')
    expect(view.traceability.intendedUse).toBe('飞行日志 / CSV 回放离线故障诊断辅助分析')
    expect(view.traceability.intendedUse).not.toContain('RflySim')
    expect(view.reportConfidence.levelText).toBe('中')
    expect(view.review.statusText).toBe('待复核')
  })

  it('旧报告缺少新增字段时仍能稳定展示默认值', () => {
    const view = buildDiagnosisReportView({
      id: 8,
      prediction_label: 'motor',
      confidence: 0.7,
      summary: {
        severity: { level: 'light', evidence: [] },
        recommendations: []
      }
    })

    expect(view.dataQuality.levelText).toBe('未记录')
    expect(view.dataQuality.warnings[0]).toContain('历史报告未记录数据质量字段')
    expect(view.traceability.rulesVersion).toBe('rule_v1')
    expect(view.reportConfidence.levelText).toBe('未知')
    expect(view.review.statusText).toBe('待复核')
    expect(view.evidenceChain).toEqual([])
  })

  it('优先使用新报告 readable 字段并区分缺失规则指标', () => {
    const view = buildDiagnosisReportView({
      prediction_label: 'sensor',
      confidence: 0.92,
      summary: {
        readable: {
          conclusion: '本次离线诊断识别为传感器故障，主分类置信度为 92.0%。',
          risk: '风险等级为高风险，建议优先复核相关传感器。',
          evidence: [
            '模型当前主要判断为传感器故障（92.0%）。',
            '规则指标缺失：陀螺振动偏高，未从当前 CSV 中提取到对应规则指标。'
          ],
          recommendations: ['[优先处理] 检查 IMU 安装与校准']
        },
        severity: {
          level: 'heavy',
          method: 'model_confidence_fallback',
          evidence: [
            {
              metric: 'gyro_vibration_p95',
              status: 'missing',
              source: 'rule',
              reason_text: '未从当前 CSV 中提取到对应规则指标。'
            },
            {
              metric: 'prediction_confidence',
              status: 'used',
              source: 'model',
              value: 0.92,
              score: 1
            }
          ]
        },
        topk: [{ label: 'sensor', prob: 0.92 }],
        recommendations: [{ title: '检查 IMU 安装与校准', priority: 'P1' }]
      }
    })

    expect(view.headline).toContain('主分类置信度为 92.0%')
    expect(view.riskText).toContain('高风险')
    expect(view.whyItems[1].description).toContain('未从当前 CSV 中提取到对应规则指标')
    expect(view.evidenceChain[0]).toMatchObject({
      sourceText: '规则证据',
      statusText: '缺失'
    })
    expect(view.evidenceChain[1].sourceText).toBe('模型证据')
  })

  it('序列化后的 summary 不出现 Unicode 转义', () => {
    const summary = buildDiagnosisReportView({
      prediction_label: 'sensor',
      confidence: 0.88,
      summary: {
        severity: { level: 'medium', evidence: [] },
        data_quality: {
          level: 'warning',
          warnings: ['未记录样本数量，无法完整评估输入覆盖度。']
        },
        review: { status: 'pending' }
      }
    })

    const serialized = JSON.stringify({
      dataQuality: summary.dataQuality,
      review: summary.review
    })

    expect(serialized).toContain('未记录样本数量')
    expect(serialized).not.toMatch(/\\u[0-9a-f]{4}/i)
  })
})
