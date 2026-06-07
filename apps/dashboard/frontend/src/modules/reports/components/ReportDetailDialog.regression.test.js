import { mount } from '@vue/test-utils'
import { describe, expect, it } from 'vitest'

import ReportDetailDialog from './ReportDetailDialog.vue'

const enhancedReport = {
  id: 21,
  run_id: 9,
  asset_id: 4,
  model_name: 'transfer3_cnnlstm_normalproto',
  prediction_label: 'sensor',
  confidence: 0.88,
  created_at: '2026-06-05 19:30:10',
  summary: {
    severity: {
      level: 'medium',
      evidence: [
        {
          metric: 'gyro_vibration_p95',
          status: 'used',
          source: 'rule',
          value: 0.42,
          score: 0.8
        },
        {
          metric: 'accel_vibration_p95',
          status: 'missing',
          source: 'rule',
          reason_text: '未从当前 CSV 中提取到对应规则指标。'
        }
      ]
    },
    readable: {
      conclusion: '本次离线诊断识别为传感器故障，主分类置信度为 88.000%。',
      risk: '风险等级为中度风险，建议尽快结合原始日志和维护记录复核。',
      evidence: [
        '模型当前主要判断为传感器故障（88.000%），次高可能为状态正常（9.000%）。',
        '规则指标：陀螺振动偏高为 0.4200，规则得分 0.8000。',
        '规则指标缺失：加速度振动偏高，未从当前 CSV 中提取到对应规则指标。'
      ],
      recommendations: ['[优先处理] 检查 IMU 安装与校准'],
      cautions: ['规则指标不足时需要结合原始日志复核。']
    },
    topk: [
      { label: 'sensor', prob: 0.88 },
      { label: 'normal', prob: 0.09 },
      { label: 'motor', prob: 0.03 }
    ],
    recommendations: [{ title: '检查 IMU 安装与校准', priority: 'P1' }],
    data_quality: {
      level: 'warning',
      sample_count: 800,
      window_count: 0,
      feature_count: 31,
      missing_value_ratio: null,
      invalid_value_ratio: null,
      warnings: ['未记录时间范围，建议结合原始日志复核。']
    },
    traceability: {
      run_id: 9,
      asset_filename: 'sensor_case.csv',
      input_csv: '/tmp/sensor_case.csv',
      model_name: 'transfer3_cnnlstm_normalproto',
      rules_version: 'rule_v1'
    },
    model_card: {
      task: 'offline_multiclass',
      class_set: ['normal', 'motor', 'sensor'],
      intended_use: '飞行日志 / CSV 回放离线故障诊断辅助分析'
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
}

const legacyReport = {
  id: 8,
  prediction_label: 'motor',
  confidence: 0.7,
  model_name: 'cnn_lstm',
  created_at: '2026-03-01 10:00:00',
  summary: {
    severity: { level: 'light', evidence: [] },
    recommendations: []
  }
}

function mountDialog(props = {}) {
  return mount(ReportDetailDialog, {
    props: {
      visible: true,
      report: enhancedReport,
      detailJson: JSON.stringify(enhancedReport.summary, null, 2),
      ...props
    },
    global: {
      stubs: {
        'el-dialog': {
          template: '<div class="el-dialog-stub"><slot /></div>'
        }
      }
    }
  })
}

describe('ReportDetailDialog regression', () => {
  it('renders enhanced report first screen as readable conclusion, evidence and actions', () => {
    const wrapper = mountDialog()
    const technicalText = wrapper.find('.technical-details').text()
    const firstScreenText = wrapper.find('.report-stack').text().replace(technicalText, '')

    expect(firstScreenText).toContain('本次离线诊断识别为传感器故障')
    expect(firstScreenText).toContain('中度风险')
    expect(firstScreenText).toContain('概率分布')
    expect(firstScreenText).toContain('处置建议')
    expect(firstScreenText).toContain('关键依据')
    expect(firstScreenText).toContain('数据质量')
    expect(firstScreenText).toContain('未从当前 CSV 中提取到对应规则指标')
    expect(firstScreenText).not.toContain('模型与规则追溯')
    expect(firstScreenText).not.toContain('报告可信度')
    expect(firstScreenText).not.toContain('人工复核')

    expect(technicalText).toContain('模型与规则追溯')
    expect(technicalText).toContain('报告可信度')
    expect(technicalText).toContain('人工复核')
    expect(technicalText).toContain('证据链')
    expect(technicalText).toContain('/tmp/sensor_case.csv')
    expect(technicalText).toContain('normal / motor / sensor')
  })

  it('renders legacy report with safe defaults and keeps json in technical details only', () => {
    const detailJson = JSON.stringify(legacyReport.summary, null, 2)
    const wrapper = mountDialog({
      report: legacyReport,
      detailJson
    })
    const technicalText = wrapper.find('.technical-details').text()
    const visibleText = wrapper.find('.report-stack').text().replace(technicalText, '')

    expect(visibleText).toContain('数据质量')
    expect(visibleText).toContain('未记录')
    expect(visibleText).toContain('历史报告未记录数据质量字段')
    expect(visibleText).not.toContain('待补充')
    expect(visibleText).not.toContain('待复核')
    expect(visibleText).not.toContain('"severity"')
    expect(wrapper.find('pre.report-json').text()).toContain('"severity"')
    expect(detailJson).not.toMatch(/\\u[0-9a-f]{4}/i)
  })
})
