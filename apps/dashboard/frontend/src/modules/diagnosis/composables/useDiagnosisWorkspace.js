import { computed, onMounted, onUnmounted, ref } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { ElMessage } from 'element-plus'
import { assetApi, diagnosisReportApi, modelApi, runApi, telemetryApi } from '@/shared/services'
import {
  buildRuleExplanationRows,
  buildTraceableEvidenceRows,
  diagnosisCopy,
  formatDiagnosisModelName,
  formatPercentage,
  getDiagnosisLabelMeta,
  normalizeDiagnosisResult,
  resolveRiskLevel,
  resolveDiagnosisStage
} from '@/modules/diagnosis/diagnosisView'

export function useDiagnosisWorkspace() {
  const route = useRoute()
  const router = useRouter()

  const assets = ref([])
  const models = ref([])
  const selectedAssetId = ref(null)
  const selectedModel = ref('')
  const currentRun = ref(null)
  const resultData = ref(null)
  const running = ref(false)
  const reportCreating = ref(false)
  const activeSessionId = ref('')
  const latestReportId = ref(null)
  const recentRuns = ref([])
  const recentReports = ref([])

  let pollTimer = null

  const progressPercentage = computed(() => Math.round(Number(currentRun.value?.progress || 0) * 100))
  const confidenceText = computed(() => formatPercentage(resultData.value?.prediction?.confidence))
  const stage = computed(() => resolveDiagnosisStage({
    currentRun: currentRun.value,
    resultData: resultData.value
  }))
  const selectedAssetName = computed(() => {
    const asset = assets.value.find((item) => item.id === selectedAssetId.value)
    return asset?.filename || diagnosisCopy.unselectedAsset
  })
  const selectedModelName = computed(() => formatDiagnosisModelName(selectedModel.value || diagnosisCopy.unselectedModel))
  const probabilityRows = computed(() => {
    const topk = Array.isArray(resultData.value?.topk) ? resultData.value.topk : []
    const probabilities = topk.length
      ? Object.fromEntries(topk.map((item) => [item.label, item.prob ?? item.probability]))
      : (resultData.value?.probs || {})

    return Object.entries(probabilities)
      .map(([label, probability]) => ({
        rawLabel: label,
        label: getDiagnosisLabelMeta(label).text,
        value: Number(probability || 0),
        probability: formatPercentage(probability)
      }))
      .sort((a, b) => b.value - a.value)
  })
  const selectedAsset = computed(() => assets.value.find((item) => item.id === selectedAssetId.value) || null)
  const selectedModelMeta = computed(() => {
    if (!selectedModel.value) return null
    return models.value.find((item) => item.name === selectedModel.value) || {
      name: selectedModel.value,
      display_name: selectedModel.value
    }
  })
  const prediction = computed(() => resultData.value?.prediction || {})
  const labelMeta = computed(() => getDiagnosisLabelMeta(prediction.value?.label))
  const riskLevel = computed(() => resolveRiskLevel({
    label: prediction.value?.label,
    confidence: prediction.value?.confidence,
    severity: resultData.value?.severity
  }))
  const reportStatusText = computed(() => {
    if (reportCreating.value) return '生成中'
    if (latestReportId.value || currentRun.value?.report_id) return '已生成'
    if (resultData.value) return '待生成'
    return '待诊断'
  })
  const latestUpdatedText = computed(() => {
    return currentRun.value?.updated_at || currentRun.value?.finished_at || currentRun.value?.created_at || '暂无'
  })
  const resultMetaItems = computed(() => [
    { label: '诊断完成时间', value: currentRun.value?.finished_at || currentRun.value?.updated_at || '暂无' },
    { label: '分析窗口', value: activeSessionId.value ? `会话 ${activeSessionId.value}` : '完整日志' },
    { label: '来源资产', value: selectedAssetName.value },
    { label: '任务 ID', value: currentRun.value?.id ? `DG-${currentRun.value.id}` : '暂无' }
  ])
  const assetSummaryItems = computed(() => {
    const asset = selectedAsset.value
    const meta = asset?.meta || {}
    return [
      { label: '文件类型', value: asset?.kind ? String(asset.kind).toUpperCase() : '待选择' },
      { label: '文件大小', value: formatBytes(asset?.size_bytes) },
      { label: '字段完整性', value: meta.field_completeness || meta.input_completeness || '暂无' },
      { label: '特征覆盖率', value: meta.feature_coverage || meta.coverage || '暂无' }
    ]
  })
  const modelSummaryItems = computed(() => {
    const model = selectedModelMeta.value
    return [
      { label: '模型族', value: formatDiagnosisModelName(model?.display_name || model?.name || selectedModel.value) },
      { label: '模型原始 ID', value: model?.name || selectedModel.value || '待选择' },
      { label: '输入特征', value: model?.feature_count || model?.metadata?.feature_count || '暂无' },
      { label: 'Avg Confidence', value: model?.avg_confidence || model?.metrics?.avg_confidence || '暂无' }
    ]
  })
  const flowSteps = computed(() => [
    { label: '数据加载', time: currentRun.value?.created_at || '待开始', status: currentRun.value ? 'ok' : 'offline' },
    { label: '特征处理', time: currentRun.value?.started_at || '待执行', status: currentRun.value ? 'ok' : 'offline' },
    { label: '模型推理', time: resultData.value ? latestUpdatedText.value : '待执行', status: resultData.value ? 'ok' : (running.value ? 'active' : 'offline') },
    { label: '结果整理', time: resultData.value ? latestUpdatedText.value : '待完成', status: resultData.value ? 'ok' : 'offline' },
    { label: '报告生成', time: latestReportId.value ? `#${latestReportId.value}` : '待生成', status: latestReportId.value ? 'ok' : 'offline' }
  ])
  const evidenceRows = computed(() => buildTraceableEvidenceRows({
    resultData: resultData.value,
    probabilityRows: probabilityRows.value,
    activeSessionId: activeSessionId.value,
    selectedAssetName: selectedAssetName.value
  }))
  const ruleRows = computed(() => buildRuleExplanationRows({
    resultData: resultData.value,
    probabilityRows: probabilityRows.value,
    activeSessionId: activeSessionId.value
  }))
  const explanationSummary = computed(() => {
    if (!resultData.value) return diagnosisCopy.emptyEvidence
    return `${labelMeta.value.summary} 规则解释基于诊断结果摘要、概率分布与规则匹配，不代表真实特征贡献算法。`
  })
  const nextActionItems = computed(() => {
    if (stage.value.key === 'error') return ['查看任务错误原因', '确认资产与模型可用', '重新启动离线诊断任务']
    if (!resultData.value) return ['选择日志资产与诊断模型', '启动离线诊断任务']
    return latestReportId.value
      ? ['查看报告中心', '进入 AI 解读']
      : ['生成诊断报告', '复核可追溯证据', '进入 AI 解读']
  })
  const runErrorText = computed(() => {
    return currentRun.value?.error || currentRun.value?.error_detail || ''
  })
  const filteredRecentReports = computed(() => {
    const label = String(prediction.value?.label || '').trim()
    if (!label) return recentReports.value
    const matched = recentReports.value.filter((item) => item.prediction_label === label)
    return matched.length ? matched : recentReports.value
  })

  function formatBytes(value) {
    const bytes = Number(value || 0)
    if (!Number.isFinite(bytes) || bytes <= 0) return '暂无'
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
    return `${(bytes / 1024 / 1024).toFixed(1)} MB`
  }

  function stopPolling() {
    if (pollTimer) {
      clearInterval(pollTimer)
      pollTimer = null
    }
  }

  async function loadMeta() {
    try {
      const [assetResponse, modelResponse] = await Promise.all([
        assetApi.list('all'),
        modelApi.getModels()
      ])

      assets.value = assetResponse.data || []
      models.value = modelResponse.data || []

      if (!selectedModel.value && models.value.length > 0) {
        selectedModel.value = models.value[0].name
      }
    } catch (error) {
      ElMessage.error(`加载资产或模型失败：${error.message || '未知错误'}`)
    }
  }

  async function loadRecent() {
    try {
      const [runResponse, reportResponse] = await Promise.all([
        runApi.list({ task: 'offline_multiclass', limit: 8 }),
        diagnosisReportApi.list({ page: 1, per_page: 8 })
      ])
      recentRuns.value = runResponse.data || []
      recentReports.value = reportResponse.data || []
      if (currentRun.value?.id && !latestReportId.value) {
        latestReportId.value = resolveReportIdForRun(currentRun.value.id)
      }
    } catch {
      recentRuns.value = []
      recentReports.value = []
    }
  }

  function resolveReportIdForRun(runId) {
    const matched = recentReports.value.find((item) => String(item.run_id) === String(runId))
    return matched?.id || currentRun.value?.report_id || null
  }

  async function pollRun(runId) {
    stopPolling()

    pollTimer = setInterval(async () => {
      try {
        const detailResponse = await runApi.detail(runId)
        currentRun.value = detailResponse.data || null

        if (detailResponse.data?.status === 'success') {
          stopPolling()
          running.value = false

          const resultResponse = await runApi.result(runId)
          resultData.value = normalizeDiagnosisResult(resultResponse.data)
          await loadRecent()
        }

        if (detailResponse.data?.status === 'error') {
          stopPolling()
          running.value = false
          ElMessage.error(detailResponse.data.error || diagnosisCopy.runExecutionFailed)
        }
      } catch (error) {
        stopPolling()
        running.value = false
        ElMessage.error(`查询任务状态失败：${error.message || '未知错误'}`)
      }
    }, 2000)
  }

  async function startDiagnosis(options = {}) {
    if (!selectedModel.value) {
      ElMessage.warning(diagnosisCopy.noModelWarning)
      return
    }

    if (!options.sessionId && !selectedAssetId.value) {
      ElMessage.warning(diagnosisCopy.noAssetWarning)
      return
    }

    running.value = true
    resultData.value = null
    latestReportId.value = null

    try {
      const response = await runApi.createOfflineDetection({
        assetId: selectedAssetId.value,
        sessionId: options.sessionId,
        model: selectedModel.value
      })
      currentRun.value = response.data || null
      await loadRecent()

      if (!currentRun.value?.id) {
        throw new Error(diagnosisCopy.createRunFailed)
      }

      await pollRun(currentRun.value.id)
    } catch (error) {
      running.value = false
      ElMessage.error(`提交诊断任务失败：${error.message || '未知错误'}`)
    }
  }

  async function handleAutoRoute() {
    const sessionId = String(route.query.session_id || '')
    const shouldAutoStart = route.query.auto === '1'

    if (!sessionId || !shouldAutoStart) {
      return
    }

    activeSessionId.value = sessionId

    try {
      await telemetryApi.sessionDetail(sessionId)
      router.replace({
        path: '/diagnosis',
        query: {
          session_id: sessionId
        }
      })
      await startDiagnosis({ sessionId })
    } catch (error) {
      ElMessage.error(`承接监测会话失败：${error.message || '未知错误'}`)
    }
  }

  async function createReport() {
    if (!currentRun.value?.id) {
      return
    }

    reportCreating.value = true

    try {
      const response = await diagnosisReportApi.create(currentRun.value.id)
      latestReportId.value = response.data?.id || latestReportId.value
      await loadRecent()
      ElMessage.success(response.message || diagnosisCopy.reportCreated)
    } catch (error) {
      ElMessage.error(`生成报告失败：${error.message || '未知错误'}`)
    } finally {
      reportCreating.value = false
    }
  }

  function openReportsCenter(reportOrRun = null) {
    const assetId = reportOrRun?.asset_id || currentRun.value?.asset_id || selectedAssetId.value
    router.push({
      path: '/reports',
      query: assetId ? { asset_id: String(assetId) } : {}
    })
  }

  function openReportAssistant(reportOrRun = null) {
    const reportId = reportOrRun?.report_id || latestReportId.value
    if (!reportId) return

    router.push({
      path: '/ai',
      query: {
        scene: 'report_analysis',
        report_id: String(reportId),
        auto_analyze: '1'
      }
    })
  }

  async function openRunResult(runId) {
    try {
      const detailResponse = await runApi.detail(runId)
      currentRun.value = detailResponse.data || null
      latestReportId.value = currentRun.value?.report_id || resolveReportIdForRun(runId)
      if (currentRun.value?.status === 'success') {
        const resultResponse = await runApi.result(runId)
        resultData.value = normalizeDiagnosisResult(resultResponse.data)
      } else {
        resultData.value = null
      }
    } catch (error) {
      ElMessage.error(`加载诊断结果失败：${error.message || '未知错误'}`)
    }
  }

  onMounted(async () => {
    await Promise.all([loadMeta(), loadRecent()])
    await handleAutoRoute()
  })

  onUnmounted(stopPolling)

  return {
    activeSessionId,
    assetSummaryItems,
    assets,
    confidenceText,
    createReport,
    currentRun,
    evidenceRows,
    explanationSummary,
    filteredRecentReports,
    flowSteps,
    latestReportId,
    latestUpdatedText,
    loadMeta,
    models,
    modelSummaryItems,
    nextActionItems,
    openRunResult,
    openReportAssistant,
    openReportsCenter,
    probabilityRows,
    progressPercentage,
    recentReports: filteredRecentReports,
    recentRuns,
    reportCreating,
    reportStatusText,
    resultData,
    resultMetaItems,
    riskLevel,
    ruleRows,
    runErrorText,
    running,
    selectedAssetId,
    selectedAssetName,
    selectedModel,
    selectedModelName,
    stage,
    startDiagnosis
  }
}
