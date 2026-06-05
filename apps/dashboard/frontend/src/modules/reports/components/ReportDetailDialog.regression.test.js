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
          value: 0.42,
          score: 0.8
        }
      ]
    },
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
  it('renders enhanced report sections for data quality, traceability, confidence and review', () => {
    const wrapper = mountDialog()
    const text = wrapper.text()

    expect(text).toContain('数据质量')
    expect(text).toContain('模型与规则追溯')
    expect(text).toContain('报告可信度')
    expect(text).toContain('人工复核')
    expect(text).toContain('证据链')
    expect(text).toContain('需关注')
    expect(text).toContain('待复核')
    expect(text).toContain('/tmp/sensor_case.csv')
    expect(text).toContain('normal / motor / sensor')
    expect(text).toContain('技术详情')
  })

  it('renders legacy report with safe defaults and keeps json in technical details only', () => {
    const detailJson = JSON.stringify(legacyReport.summary, null, 2)
    const wrapper = mountDialog({
      report: legacyReport,
      detailJson
    })
    const visibleText = wrapper.find('.report-stack').text().split('技术详情')[0]

    expect(visibleText).toContain('数据质量')
    expect(visibleText).toContain('未知')
    expect(visibleText).toContain('待复核')
    expect(visibleText).not.toContain('"severity"')
    expect(wrapper.find('pre.report-json').text()).toContain('"severity"')
    expect(detailJson).not.toMatch(/\\u[0-9a-f]{4}/i)
  })
})
