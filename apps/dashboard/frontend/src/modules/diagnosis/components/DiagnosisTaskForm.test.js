import { mount } from '@vue/test-utils'
import { describe, expect, it } from 'vitest'

import DiagnosisTaskForm from './DiagnosisTaskForm.vue'

function mountTaskForm(overrides = {}) {
  return mount(DiagnosisTaskForm, {
    props: {
      assets: [
        {
          id: 7,
          filename: 'flight_case.csv'
        }
      ],
      models: [
        {
          name: 'transfer3_cnnlstm_normalproto',
          display_name: 'CNN-LSTM Normal Prototype'
        }
      ],
      assetSummaryItems: [
        { label: '类型', value: 'CSV' },
        { label: '大小', value: '1.2 MB' },
        { label: '字段完整性', value: '96%' },
        { label: '特征覆盖率', value: '89%' }
      ],
      modelContextText: '正常 / 电机故障 / 传感器故障 · 31 项输入特征',
      running: false,
      stage: {
        key: 'idle',
        label: '待开始'
      },
      selectedAssetId: 7,
      selectedModel: 'transfer3_cnnlstm_normalproto',
      ...overrides
    }
  })
}

describe('DiagnosisTaskForm focused task launcher', () => {
  it('renders diagnosis inputs without internal model metrics', () => {
    const wrapper = mountTaskForm()

    expect(wrapper.text()).toContain('离线诊断')
    expect(wrapper.text()).toContain('正常 / 电机故障 / 传感器故障')
    expect(wrapper.text()).toContain('31 项输入特征')
    expect(wrapper.text()).not.toContain('模型概览')
    expect(wrapper.text()).not.toContain('模型原始 ID')
    expect(wrapper.text()).not.toContain('Macro F1')
    expect(wrapper.findAll('.diagnosis-param-card')).toHaveLength(0)
  })

  it('does not render model context when no model is selected', () => {
    const wrapper = mountTaskForm({
      selectedModel: '',
      modelContextText: ''
    })

    expect(wrapper.find('.diagnosis-model-context').exists()).toBe(false)
    expect(wrapper.text()).not.toContain('transfer3_cnnlstm_normalproto')
  })

  it('shows a compact running state without internal step timestamps', () => {
    const wrapper = mountTaskForm({
      running: true,
      stage: { key: 'running', label: '诊断进行中' }
    })

    expect(wrapper.text()).toContain('正在分析日志')
    expect(wrapper.text()).not.toContain('报告生成')
    expect(wrapper.findAll('.diagnosis-flow__step')).toHaveLength(0)
  })
})
