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
    if (asset?.filename) return asset.filename
    if (activeSessionId.value) return `实时监测会话 ${activeSessionId.value}`
    return diagnosisCopy.unselectedAsset
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
    { label: '诊断来源', value: resolveDiagnosisSourceName() },
    { label: '完成时间', value: currentRun.value?.finished_at || currentRun.value?.updated_at || '暂无' }
  ])
  const assetSummaryItems = computed(() => {
    const asset = selectedAsset.value
    if (!asset) return []
    const meta = asset?.meta || {}
    const items = [
      { label: '类型', value: asset?.kind ? String(asset.kind).toUpperCase() : '未知' },
      { label: '大小', value: formatBytes(asset?.size_bytes) }
    ]
    const completeness = meta.field_completeness || meta.input_completeness
    const coverage = meta.feature_coverage || meta.coverage
    if (completeness) items.push({ label: '字段完整性', value: completeness })
    if (coverage) items.push({ label: '特征覆盖率', value: coverage })
    return items
  })
  const modelContextText = computed(() => {
    const model = selectedModelMeta.value
    if (!model) return ''
    const labelNames = Array.isArray(model.label_names) ? model.label_names : []
    const labelText = labelNames.length
      ? labelNames.map(formatTaskLabel).join(' / ')
      : '离线故障分类'
    const featureCount = model.feature_count || model.metadata?.feature_count
    return featureCount ? `${labelText} · ${featureCount} 项输入特征` : labelText
  })
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

  function formatTaskLabel(value) {
    const labels = {
      normal: '正常',
      motor: '电机故障',
      motor_failure: '电机故障',
      sensor: '传感器故障'
    }
    return labels[String(value || '').toLowerCase()] || String(value || '')
  }

  function resolveDiagnosisSourceName() {
    const runAssetId = currentRun.value?.asset_id
    const runAsset = assets.value.find((item) => String(item.id) === String(runAssetId || ''))
    if (runAsset?.filename) return runAsset.filename
    const sessionId = currentRun.value?.session_id || activeSessionId.value
    if (sessionId) return `实时监测会话 ${sessionId}`
    return selectedAssetName.value
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
    latestReportId,
    latestUpdatedText,
    loadMeta,
    models,
    modelContextText,
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
