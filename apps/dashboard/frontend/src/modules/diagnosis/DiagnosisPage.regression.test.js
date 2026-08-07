import { mount } from '@vue/test-utils'
import { describe, expect, it, vi } from 'vitest'

const workspaceState = vi.hoisted(() => ({
  activeSessionId: 'TS-262735',
  assetSummaryItems: [{ label: '文件类型', value: 'CSV' }],
  assets: [],
  confidenceText: '89.00%',
  createReport: vi.fn(),
  currentRun: {
    id: 1024,
    task: 'offline_multiclass',
    status: 'success',
    created_at: '2026-05-05 10:22:14',
    updated_at: '2026-05-05 10:26:18'
  },
  evidenceRows: [{ name: '故障类别置信度', source: 'Top-K 概率', window: '当前任务', note: '规则参考' }],
  explanationSummary:
    '在当前分析窗口内，电机故障为最高概率类别；规则解释基于诊断结果摘要与规则匹配，不代表真实特征贡献算法。',
  latestReportId: null,
  loadMeta: vi.fn(),
  modelContextText: '正常 / 电机故障 / 传感器故障 · 31 项输入特征',
  models: [],
  openReportAssistant: vi.fn(),
  openReportsCenter: vi.fn(),
  openRunResult: vi.fn(),
  probabilityRows: [{ label: 'motor', probability: '89.00%', value: 0.89 }],
  progressPercentage: 100,
  recentReports: [{ id: 8, prediction_label: 'motor', confidence: 0.89 }],
  recentRuns: [{ id: 1024, status: 'success', task: 'offline_multiclass' }],
  reportCreating: false,
  reportStatusText: '待生成',
  resultData: {
    prediction: { label: 'motor', confidence: 0.89 },
    probs: { motor: 0.89, sensor: 0.11 }
  },
  resultMetaItems: [{ label: '诊断完成时间', value: '2026-05-05 10:26:18' }],
  riskLevel: { label: '高风险', tone: 'danger', description: '置信度较高' },
  ruleRows: [{ title: '规则 1', detail: 'Top-1 故障类别概率最高。' }],
  runErrorText: '',
  running: false,
  selectedAssetId: 7,
  selectedAssetName: 'UAV-1023 / LOG-20240429-018',
  selectedModel: 'online_detector_kg_mstca_rfly',
  selectedModelName: 'KG-MSTCA',
  stage: {
    key: 'completed',
    label: '诊断完成',
    description: '结果已就绪。',
    cue: '生成报告并进入 AI 解读'
  },
  startDiagnosis: vi.fn()
}))

vi.mock('@/modules/diagnosis/composables/useDiagnosisWorkspace', () => ({
  useDiagnosisWorkspace: () => workspaceState
}))

import DiagnosisPage from './DiagnosisPage.vue'
import DiagnosisEvidencePanel from './components/DiagnosisEvidencePanel.vue'
import DiagnosisRecentRunsPanel from './components/DiagnosisRecentRunsPanel.vue'
import DiagnosisResultPanel from './components/DiagnosisResultPanel.vue'

describe('DiagnosisPage industrial diagnosis workbench', () => {
  it('renders the three-column diagnosis workbench and traceable explanation panels', () => {
    const wrapper = mount(DiagnosisPage, {
      global: {
        stubs: {
          DiagnosisSessionBanner: { template: '<div>会话承接</div>' },
          DiagnosisCommandHero: { template: '<section>顶部状态条报告状态最近更新</section>' },
          DiagnosisTaskForm: { template: '<section>离线诊断日志资产诊断模型</section>' },
          DiagnosisResultPanel: { template: '<section>诊断结果类别概率</section>' },
          DiagnosisEvidencePanel: { template: '<aside>规则解释 / 诊断摘要可追溯证据 TOP5历史参考报告</aside>' },
          DiagnosisRecentRunsPanel: { template: '<section>最近诊断任务 / 报告出口</section>' }
        }
      }
    })

    const text = wrapper.text()
    expect(text).toContain('顶部状态条')
    expect(text).toContain('离线诊断')
    expect(text).toContain('诊断结果')
    expect(text).toContain('规则解释 / 诊断摘要')
    expect(text).toContain('可追溯证据 TOP5')
    expect(text).toContain('历史参考报告')
    expect(text).toContain('最近诊断任务 / 报告出口')
    expect(text).not.toContain('SHAP')
    expect(text).not.toContain('feature importance')
  })

  it('renders a compact preparation state before any diagnosis result exists', () => {
    const wrapper = mount(DiagnosisResultPanel, {
      props: {
        confidenceText: '0.00%',
        latestReportId: null,
        probabilityRows: [],
        reportCreating: false,
        resultData: null,
        resultMetaItems: [],
        riskLevel: { label: '待确认', tone: 'pending', description: '暂无风险等级。' },
        stage: { key: 'idle', label: '待开始' }
      }
    })

    const text = wrapper.text()
    expect(text).toContain('等待诊断任务')
    expect(text).toContain('选择日志资产并启动诊断')
    expect(text).not.toContain('下一步操作')
    expect(text).not.toContain('Top-K 概率分布')
    expect(text).not.toContain('风险等级')
  })

  it('renders one concise result panel without the duplicate risk meter', () => {
    const wrapper = mount(DiagnosisResultPanel, {
      props: {
        confidenceText: '89.00%',
        latestReportId: null,
        probabilityRows: [
          { label: '电机故障', probability: '89.00%' },
          { label: '状态正常', probability: '11.00%' }
        ],
        reportCreating: false,
        resultData: { prediction: { label: 'motor', confidence: 0.89 } },
        resultMetaItems: [
          { label: '诊断来源', value: 'flight_case.csv' },
          { label: '完成时间', value: '2026-05-05 10:26:18' }
        ],
        riskLevel: { label: '高风险', tone: 'danger', description: '置信度较高' },
        stage: { key: 'completed', label: '诊断完成' }
      }
    })

    const text = wrapper.text()
    expect(text).toContain('类别概率')
    expect(text).toContain('生成报告')
    expect(text).not.toContain('Top-K')
    expect(wrapper.find('.risk-meter').exists()).toBe(false)
  })

  it('renders an explicit failed state when the run status is error', () => {
    const wrapper = mount(DiagnosisResultPanel, {
      props: {
        confidenceText: '0.00%',
        latestReportId: null,
        probabilityRows: [],
        reportCreating: false,
        resultData: null,
        resultMetaItems: [],
        runErrorText: "离线检测失败: No module named 'build_sequence_cache'",
        riskLevel: { label: '待确认', tone: 'pending', description: '暂无风险等级。' },
        stage: { key: 'error', label: '诊断失败', description: '离线诊断任务执行失败，请查看错误原因并重新提交。' }
      }
    })

    const text = wrapper.text()
    expect(text).toContain('诊断失败')
    expect(text).toContain('离线诊断任务执行失败')
    expect(text).toContain("No module named 'build_sequence_cache'")
    expect(text).not.toContain('下一步操作')
    expect(text).not.toContain('诊断准备状态')
  })

  it('keeps the explanation panel lightweight when no result is available', () => {
    const wrapper = mount(DiagnosisEvidencePanel, {
      props: {
        evidenceRows: [],
        explanationSummary: '暂无可追溯证据；完成诊断后将根据结果摘要生成规则解释。',
        recentReports: [],
        resultData: null,
        riskLevel: { label: '待确认', tone: 'pending' },
        ruleRows: []
      }
    })

    const text = wrapper.text()
    expect(text).toContain('完成诊断后生成规则解释')
    expect(text).toContain('证据链待生成')
    expect(text).toContain('历史参考待匹配')
    expect(text).not.toContain('SHAP')
    expect(text).not.toContain('feature importance')
  })

  it('emits row-specific report and AI actions from recent runs', async () => {
    const run = {
      id: 1024,
      asset_id: 7,
      report_id: 88,
      status: 'success',
      task: 'offline_multiclass',
      result: {
        prediction: { label: 'motor', confidence: 0.89 }
      },
      created_at: '2026-05-05 10:22:14',
      updated_at: '2026-05-05 10:26:18'
    }
    const wrapper = mount(DiagnosisRecentRunsPanel, {
      props: {
        recentRuns: [run],
        reportStatusText: '已生成'
      }
    })

    const buttons = wrapper.findAllComponents({ name: 'ElButton' })
    await buttons[1].trigger('click')
    await buttons[2].trigger('click')

    expect(wrapper.emitted('open-reports-center')?.[0]).toEqual([run])
    expect(wrapper.emitted('open-ai-assistant')?.[0]).toEqual([run])
  })
})
