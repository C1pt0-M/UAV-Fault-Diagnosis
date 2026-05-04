import { computed, onMounted, onUnmounted, ref } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { ElMessage } from 'element-plus'
import { assetApi, diagnosisReportApi, modelApi, runApi, telemetryApi } from '@/shared/services'
import {
  diagnosisCopy,
  formatPercentage,
  normalizeDiagnosisResult,
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
  const selectedModelName = computed(() => selectedModel.value || diagnosisCopy.unselectedModel)
  const probabilityRows = computed(() => {
    const probabilities = resultData.value?.probs || {}

    return Object.entries(probabilities).map(([label, probability]) => ({
      label,
      probability: formatPercentage(probability)
    }))
  })

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
      ElMessage.success(response.message || diagnosisCopy.reportCreated)
    } catch (error) {
      ElMessage.error(`生成报告失败：${error.message || '未知错误'}`)
    } finally {
      reportCreating.value = false
    }
  }

  function openReportsCenter() {
    const assetId = currentRun.value?.asset_id || selectedAssetId.value
    router.push({
      path: '/reports',
      query: assetId ? { asset_id: String(assetId) } : {}
    })
  }

  function openReportAssistant() {
    if (!latestReportId.value) return

    router.push({
      path: '/ai',
      query: {
        scene: 'report_analysis',
        report_id: String(latestReportId.value)
      }
    })
  }

  onMounted(async () => {
    await loadMeta()
    await handleAutoRoute()
  })

  onUnmounted(stopPolling)

  return {
    activeSessionId,
    assets,
    confidenceText,
    createReport,
    currentRun,
    latestReportId,
    loadMeta,
    models,
    openReportAssistant,
    openReportsCenter,
    probabilityRows,
    progressPercentage,
    reportCreating,
    resultData,
    running,
    selectedAssetId,
    selectedAssetName,
    selectedModel,
    selectedModelName,
    stage,
    startDiagnosis
  }
}
