import { mount } from '@vue/test-utils'
import { describe, expect, it, vi } from 'vitest'

const workspaceState = vi.hoisted(() => ({
  activeSessionId: '',
  assets: [],
  confidenceText: '92.40%',
  createReport: vi.fn(),
  currentRun: null,
  latestReportId: null,
  loadMeta: vi.fn(),
  models: [],
  openReportAssistant: vi.fn(),
  openReportsCenter: vi.fn(),
  probabilityRows: [],
  progressPercentage: 0,
  reportCreating: false,
  resultData: null,
  running: false,
  selectedAssetId: null,
  selectedAssetName: '尚未选择资产',
  selectedModel: '',
  selectedModelName: '尚未选择模型',
  stage: {
    key: 'idle',
    label: '待开始',
    description: '锁定日志资产与诊断模型后即可发起任务。',
    cue: '准备启动离线诊断'
  },
  startDiagnosis: vi.fn()
}))

vi.mock('@/modules/diagnosis/composables/useDiagnosisWorkspace', () => ({
  useDiagnosisWorkspace: () => workspaceState
}))

import DiagnosisPage from './DiagnosisPage.vue'

describe('DiagnosisPage competition redesign', () => {
  it('renders the command-center storyline for the diagnosis workspace', () => {
    const wrapper = mount(DiagnosisPage, {
      global: {
        stubs: {
          DiagnosisSessionBanner: { template: '<div />' },
          DiagnosisCommandHero: { template: '<section>离线诊断任务指挥台</section>' },
          DiagnosisTaskForm: { template: '<section>任务发起区</section>' },
          DiagnosisStatusPanel: { template: '<section>任务状态雷达</section>' },
          DiagnosisResultPanel: { template: '<section>诊断结果主舞台</section>' }
        }
      }
    })

    const text = wrapper.text()
    expect(text).toContain('离线诊断任务指挥台')
    expect(text).toContain('任务发起区')
    expect(text).toContain('任务状态雷达')
    expect(text).toContain('诊断结果主舞台')
  })
})
