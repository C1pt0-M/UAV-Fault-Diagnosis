import { computed, onMounted, onUnmounted, reactive, ref } from 'vue'
import { useRouter } from 'vue-router'
import { ElMessage, ElMessageBox } from 'element-plus'
import { assetApi, modelApi, telemetryApi } from '@/shared/services'
import {
  formatLatestValue,
  monitoringCopy,
  monitoringStateLabels,
  resolveMonitoringState
} from '@/modules/monitoring/monitoringView'

const actionKeys = [
  'startCollection',
  'startDetection',
  'startBlindTest',
  'stopDetection',
  'stopCollection'
]

function createActionLoadingState() {
  return actionKeys.reduce((state, key) => {
    state[key] = false
    return state
  }, {})
}

export function useMonitoringDashboard() {
  const router = useRouter()

  const runStatus = reactive({})
  const actionLoading = reactive(createActionLoadingState())
  const onlineModels = ref([])
  const replayAssets = ref([])
  const sessions = ref([])
  const sessionDetail = ref(null)
  const detailVisible = ref(false)
  const realtime = ref({ times: [] })
  const selectedBackend = ref('px4')
  const selectedDuration = ref(60)
  const selectedModel = ref('')
  const selectedReplayAssetId = ref(null)

  let pollTimer = null

  const currentState = computed(() => resolveMonitoringState(runStatus))
  const stateLabel = computed(() => monitoringStateLabels[currentState.value] || monitoringStateLabels.IDLE)
  const alerts = computed(() => runStatus.fault_events || [])
  const riskScore = computed(() => Number(runStatus.risk_score || 0).toFixed(2))
  const realtimeCount = computed(() => realtime.value?.times?.length || 0)
  const latestTimestamp = computed(() => {
    return formatLatestValue(realtime.value?.times, (value) => `${value} s`)
  })
  const latestAltitude = computed(() => {
    return formatLatestValue(realtime.value?.pos_z, (value) => `${(-Number(value)).toFixed(2)} m`)
  })
  const latestRoll = computed(() => {
    return formatLatestValue(realtime.value?.roll, (value) => `${Number(value).toFixed(2)}°`)
  })
  const latestPitch = computed(() => {
    return formatLatestValue(realtime.value?.pitch, (value) => `${Number(value).toFixed(2)}°`)
  })

  const metrics = computed(() => [
    {
      label: '当前状态',
      value: stateLabel.value,
      meta: `当前会话：${runStatus.current_session_id || monitoringCopy.noSession}`
    },
    {
      label: '风险评分',
      value: riskScore.value,
      meta: `累计故障事件：${runStatus.fault_events_count || 0}`
    },
    {
      label: '在线模型',
      value: selectedModel.value || monitoringCopy.noSelection,
      meta: `数据来源：${selectedBackend.value === 'replay' ? 'Replay CSV' : 'PX4 SITL'}`
    },
    {
      label: '最近样本',
      value: latestTimestamp.value,
      meta: '最近一次拉取的实时数据时间点'
    }
  ])

  const sampleSummaryItems = computed(() => [
    { label: '采样点数量', value: realtimeCount.value },
    { label: '最新高度', value: latestAltitude.value },
    { label: '最新滚转角', value: latestRoll.value },
    { label: '最新俯仰角', value: latestPitch.value }
  ])

  function stopPolling() {
    if (pollTimer) {
      clearInterval(pollTimer)
      pollTimer = null
    }
  }

  async function runAction(actionKey, handler) {
    actionLoading[actionKey] = true
    try {
      await handler()
    } finally {
      actionLoading[actionKey] = false
    }
  }

  async function fetchStatus() {
    const response = await telemetryApi.status()
    Object.assign(runStatus, response.data || {})
  }

  async function fetchRealtime() {
    const response = await telemetryApi.realtimeData()
    realtime.value = response.data || { times: [] }
  }

  async function loadModels() {
    const response = await modelApi.getModels({ capability: 'realtime_detection' })
    onlineModels.value = response.data || []

    if (!selectedModel.value && onlineModels.value.length > 0) {
      selectedModel.value = onlineModels.value[0].name
    }
  }

  async function loadReplayAssets() {
    const response = await assetApi.list('all')
    replayAssets.value = (response.data || []).filter((item) => ['csv', 'ulg'].includes(item.kind))
  }

  async function loadSessions() {
    const response = await telemetryApi.sessions({ page: 1, per_page: 20 })
    sessions.value = response.data?.items || []
  }

  async function refreshDashboard() {
    await Promise.all([fetchStatus(), loadSessions(), fetchRealtime()])
  }

  function startPolling() {
    stopPolling()
    pollTimer = setInterval(async () => {
      try {
        await refreshDashboard()
      } catch {
        // 轮询错误在用户主动操作时再提示，避免持续刷屏。
      }
    }, 4000)
  }

  async function startCollection() {
    await runAction('startCollection', async () => {
      if (selectedBackend.value === 'replay' && !selectedReplayAssetId.value) {
        ElMessage.warning(monitoringCopy.replayAssetRequired)
        return
      }

      const replayAsset = replayAssets.value.find((item) => item.id === selectedReplayAssetId.value)

      try {
        await telemetryApi.startCollection({
          drone_id: 'DRONE-001',
          session_type: 'simulation',
          sim_time: selectedDuration.value,
          backend: selectedBackend.value,
          replay_path: selectedBackend.value === 'replay'
            ? replayAsset?.meta?.prepared_csv_path || replayAsset?.stored_path
            : undefined
        })

        ElMessage.success(monitoringCopy.startCollectionSuccess)
        await refreshDashboard()
        startPolling()
      } catch (error) {
        ElMessage.error(`启动采集失败：${error.message || '未知错误'}`)
      }
    })
  }

  async function startDetection() {
    await runAction('startDetection', async () => {
      try {
        await telemetryApi.startDetection({ model: selectedModel.value || undefined })
        ElMessage.success(monitoringCopy.startDetectionSuccess)
        await refreshDashboard()
        startPolling()
      } catch (error) {
        ElMessage.error(`启动检测失败：${error.message || '未知错误'}`)
      }
    })
  }

  async function stopDetection() {
    await runAction('stopDetection', async () => {
      try {
        await telemetryApi.stopDetection()
        ElMessage.success(monitoringCopy.stopDetectionSuccess)
        await refreshDashboard()
      } catch (error) {
        ElMessage.error(`停止检测失败：${error.message || '未知错误'}`)
      }
    })
  }

  async function stopCollection() {
    await runAction('stopCollection', async () => {
      try {
        const response = await telemetryApi.stopCollection()
        ElMessage.success(response.message || monitoringCopy.stopCollectionFallback)
        stopPolling()
        await refreshDashboard()

        if (Number(response.data?.fault_events_count || 0) <= 0) {
          return
        }

        try {
          await ElMessageBox.confirm(
            monitoringCopy.stopCollectionConfirmMessage(response.data.fault_events_count),
            monitoringCopy.stopCollectionConfirmTitle,
            {
              type: 'warning',
              confirmButtonText: '立即跳转',
              cancelButtonText: '稍后处理'
            }
          )

          router.push({
            path: '/diagnosis',
            query: {
              session_id: response.data.session_id,
              auto: '1'
            }
          })
        } catch (error) {
          if (error?.message && error.message !== 'cancel') {
            throw error
          }
        }
      } catch (error) {
        if (error?.message !== 'cancel') {
          ElMessage.error(`停止采集失败：${error.message || '未知错误'}`)
        }
      }
    })
  }

  async function startBlindTest() {
    await runAction('startBlindTest', async () => {
      try {
        await telemetryApi.startBlindTest({
          drone_id: 'DRONE-001',
          sim_time: selectedDuration.value
        })
        ElMessage.success(monitoringCopy.startBlindTestSuccess)
        await refreshDashboard()
        startPolling()
      } catch (error) {
        ElMessage.error(`启动盲测失败：${error.message || '未知错误'}`)
      }
    })
  }

  async function openSessionDetail(sessionId) {
    try {
      const response = await telemetryApi.sessionDetail(sessionId)
      sessionDetail.value = response.data || null
      detailVisible.value = true
    } catch (error) {
      ElMessage.error(`加载会话详情失败：${error.message || '未知错误'}`)
    }
  }

  function continueDiagnosis(sessionId) {
    router.push({
      path: '/diagnosis',
      query: {
        session_id: sessionId,
        auto: '1'
      }
    })
  }

  onMounted(async () => {
    try {
      await Promise.all([loadModels(), loadReplayAssets(), refreshDashboard()])
      if (currentState.value !== 'IDLE') {
        startPolling()
      }
    } catch (error) {
      ElMessage.error(`初始化监测页失败：${error.message || '未知错误'}`)
    }
  })

  onUnmounted(stopPolling)

  return {
    actionLoading,
    alerts,
    continueDiagnosis,
    currentState,
    detailVisible,
    loadSessions,
    metrics,
    onlineModels,
    openSessionDetail,
    replayAssets,
    sampleSummaryItems,
    selectedBackend,
    selectedDuration,
    selectedModel,
    selectedReplayAssetId,
    sessionDetail,
    sessions,
    startBlindTest,
    startCollection,
    startDetection,
    stopCollection,
    stopDetection
  }
}
