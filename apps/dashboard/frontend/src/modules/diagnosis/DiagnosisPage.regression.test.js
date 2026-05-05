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
  flowSteps: [{ label: '数据加载', time: '10:22:14', status: 'ok' }],
  latestReportId: null,
  loadMeta: vi.fn(),
  modelSummaryItems: [{ label: '模型族', value: 'KG-MSTCA' }],
  models: [],
  nextActionItems: ['生成诊断报告', '进入 AI 解读'],
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
import DiagnosisResultPanel from './components/DiagnosisResultPanel.vue'

describe('DiagnosisPage industrial diagnosis workbench', () => {
  it('renders the three-column diagnosis workbench and traceable explanation panels', () => {
    const wrapper = mount(DiagnosisPage, {
      global: {
        stubs: {
          DiagnosisSessionBanner: { template: '<div>会话承接</div>' },
          DiagnosisCommandHero: { template: '<section>顶部状态条报告状态最近更新</section>' },
          DiagnosisTaskForm: { template: '<section>任务发起 / 输入配置日志概览模型概览</section>' },
          DiagnosisResultPanel: { template: '<section>诊断结果Top-K 概率分布风险等级下一步操作</section>' },
          DiagnosisEvidencePanel: { template: '<aside>规则解释 / 诊断摘要可追溯证据 TOP5历史参考报告AI 解读入口</aside>' },
          DiagnosisRecentRunsPanel: { template: '<section>最近诊断任务 / 报告出口</section>' }
        }
      }
    })

    const text = wrapper.text()
    expect(text).toContain('顶部状态条')
    expect(text).toContain('任务发起 / 输入配置')
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
        nextActionItems: ['选择日志资产与诊断模型', '启动离线诊断任务'],
        probabilityRows: [],
        reportCreating: false,
        resultData: null,
        resultMetaItems: [],
        riskLevel: { label: '待确认', tone: 'pending', description: '暂无风险等级。' },
        stage: { key: 'idle', label: '待开始' }
      }
    })

    const text = wrapper.text()
    expect(text).toContain('诊断准备状态')
    expect(text).toContain('待选择资产')
    expect(text).toContain('待启动任务')
    expect(text).toContain('报告待生成')
    expect(text).toContain('AI 解读待解锁')
    expect(text).toContain('选择日志资产与诊断模型')
    expect(text).toContain('启动离线诊断任务')
    expect(text).not.toContain('Top-K 概率分布')
    expect(text).not.toContain('风险等级')
  })

  it('keeps the explanation panel lightweight when no result is available', () => {
    const wrapper = mount(DiagnosisEvidencePanel, {
      props: {
        evidenceRows: [],
        explanationSummary: '暂无可追溯证据；完成诊断后将根据结果摘要生成规则解释。',
        latestReportId: null,
        recentReports: [],
        recommendationItems: ['选择日志资产与诊断模型', '启动离线诊断任务'],
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
})
