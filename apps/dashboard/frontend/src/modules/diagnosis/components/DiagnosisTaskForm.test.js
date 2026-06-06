import { shallowMount } from '@vue/test-utils'
import { describe, expect, it } from 'vitest'

import DiagnosisTaskForm from './DiagnosisTaskForm.vue'

function mountTaskForm(overrides = {}) {
  return shallowMount(DiagnosisTaskForm, {
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
        { label: '文件类型', value: 'CSV' },
        { label: '文件大小', value: '1.2 MB' },
        { label: '字段完整性', value: '96%' },
        { label: '特征覆盖率', value: '89%' }
      ],
      modelSummaryItems: [
        { label: '模型族', value: 'CNN-LSTM' },
        { label: '模型原始 ID', value: 'transfer3_cnnlstm_normalproto' },
        { label: '输入特征', value: 256 },
        { label: 'Macro F1', value: '86.00%' }
      ],
      flowSteps: [
        { label: '数据加载', time: '10:22:14', status: 'ok' },
        { label: '特征处理', time: '10:23:31', status: 'ok' },
        { label: '模型推理', time: '待执行', status: 'offline' }
      ],
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

describe('DiagnosisTaskForm industrial task launcher', () => {
  it('renders compact diagnosis parameters with readable model name and raw id', () => {
    const wrapper = mountTaskForm()

    expect(wrapper.findAll('.diagnosis-param-card')).toHaveLength(8)
    expect(wrapper.findAll('.diagnosis-flow__step')).toHaveLength(3)
    expect(wrapper.text()).toContain('可用资产')
    expect(wrapper.text()).toContain('1')
    expect(wrapper.text()).toContain('日志概览')
    expect(wrapper.text()).toContain('模型概览')
    expect(wrapper.text()).toContain('CNN-LSTM')
    expect(wrapper.text()).toContain('transfer3_cnnlstm_normalproto')
  })

  it('does not render raw model id when no model is selected', () => {
    const wrapper = mountTaskForm({
      selectedModel: '',
      modelSummaryItems: [
        { label: '模型族', value: '尚未选择模型' },
        { label: '模型原始 ID', value: '待选择' },
        { label: '输入特征', value: '暂无' },
        { label: 'Macro F1', value: '暂无' }
      ]
    })

    expect(wrapper.text()).toContain('尚未选择模型')
    expect(wrapper.text()).not.toContain('transfer3_cnnlstm_normalproto')
  })
})
