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
      running: false,
      selectedAssetId: 7,
      selectedModel: 'transfer3_cnnlstm_normalproto',
      ...overrides
    }
  })
}

describe('DiagnosisTaskForm industrial task launcher', () => {
  it('renders compact diagnosis parameters with readable model name and raw id', () => {
    const wrapper = mountTaskForm()

    expect(wrapper.findAll('.diagnosis-param-card')).toHaveLength(3)
    expect(wrapper.text()).toContain('可用资产')
    expect(wrapper.text()).toContain('1')
    expect(wrapper.text()).toContain('当前资产')
    expect(wrapper.text()).toContain('flight_case.csv')
    expect(wrapper.text()).toContain('诊断模型')
    expect(wrapper.text()).toContain('CNN-LSTM Normal Prototype')
    expect(wrapper.text()).toContain('transfer3_cnnlstm_normalproto')
  })

  it('does not render raw model id when no model is selected', () => {
    const wrapper = mountTaskForm({
      selectedModel: ''
    })

    expect(wrapper.text()).toContain('尚未选择模型')
    expect(wrapper.text()).not.toContain('transfer3_cnnlstm_normalproto')
  })
})
