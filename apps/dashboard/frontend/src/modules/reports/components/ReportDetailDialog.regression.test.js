import { mount } from '@vue/test-utils'
import { describe, expect, it } from 'vitest'

import ReportDetailDialog from './ReportDetailDialog.vue'

const ElDialogStub = {
  name: 'ElDialog',
  props: ['modelValue', 'title', 'width'],
  template: `
    <div class="el-dialog-stub">
      <div class="el-dialog-stub__title">{{ title }}</div>
      <slot />
    </div>
  `
}

function mountDialog(report) {
  return mount(ReportDetailDialog, {
    props: {
      visible: true,
      report,
      detailJson: JSON.stringify(report || {}, null, 2)
    },
    global: {
      stubs: {
        ElDialog: ElDialogStub
      }
    }
  })
}

const enhancedReport = {
  id: 9,
  run_id: 21,
  asset_id: 178,
  model_name: 'transfer3_cnnlstm_normalproto',
  prediction_label: 'motor',
  confidence: 0.93,
  created_at: '2026-04-10 17:29:01',
  summary: {
    severity: {
      level: 'heavy',
      evidence: [
        {
          metric: 'motor_imbalance_ratio',
          status: 'used',
          value: 0.41,
          score: 1
        }
      ]
    },
    data_quality: {
      level: 'good',
      sample_count: 120,
      window_count: 40,
      feature_count: 32,
      warnings: []
    },
    traceability: {
      run_id: 21,
      asset_filename: 'TS-148898.csv',
      input_csv: '/tmp/TS-148898.csv',
      model_name: 'transfer3_cnnlstm_normalproto',
      rules_version: 'rule_v1',
      generated_at: '2026-04-10 17:29:01'
    },
    model_card: {
      task: 'offline_multiclass',
      class_set: ['normal', 'motor', 'sensor'],
      intended_use: '离线日志故障诊断辅助分析',
      limitations: []
    },
    report_confidence: {
      level: 'high',
      score: 0.93,
      reasons: ['输入数据质量可接受', '模型主分类置信度较高']
    },
    review: {
      status: 'pending',
      reviewer: '',
      reviewed_at: '',
      conclusion: '',
      maintenance_notes: ''
    },
    recommendations: [
      { title: '必要时停飞并安排动力系统复检', priority: 'P0' }
    ],
    topk: [
      { label: 'motor', prob: 0.93 },
      { label: 'sensor', prob: 0.04 }
    ]
  }
}

const legacyReport = {
  id: 3,
  prediction_label: 'sensor',
  confidence: 0.71,
  model_name: 'cnn_lstm',
  created_at: '2026-01-01 00:00:00',
  summary: {
    severity: { level: 'medium' },
    recommendations: [{ title: '检查 IMU 校准', priority: 'P1' }]
  }
}

describe('ReportDetailDialog regression', () => {
  it('renders enhanced report sections for data quality, traceability, confidence and review', () => {
    const wrapper = mountDialog(enhancedReport)
    const text = wrapper.text()

    expect(text).toContain('数据质量')
    expect(text).toContain('报告可信度')
    expect(text).toContain('模型与规则追溯')
    expect(text).toContain('人工复核')
    expect(text).toContain('待复核')
    expect(text).toContain('normal / motor / sensor')
    expect(text).toContain('样本数量')
  })

  it('keeps legacy reports readable with placeholder values', () => {
    const wrapper = mountDialog(legacyReport)
    const text = wrapper.text()

    expect(text).toContain('数据质量')
    expect(text).toContain('未知')
    expect(text).toContain('待复核')
    expect(text).toContain('待补充')
    expect(text).not.toContain('undefined')
  })
})
