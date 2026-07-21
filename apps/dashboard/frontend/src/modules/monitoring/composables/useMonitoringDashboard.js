import { computed, onMounted, onUnmounted, reactive, ref, watch } from 'vue'
import { useRouter } from 'vue-router'
import { ElMessage } from 'element-plus'
import { assetApi, modelApi, telemetryApi } from '@/shared/services'
import {
  formatLatestValue,
  formatMonitoringModelName,
  monitoringBackendOptions,
  monitoringCopy,
  rflyInferenceLocationOptions,
  rflyNetworkProfileOptions,
  monitoringStateLabels,
  resolveMonitoringBackendOptions,
  resolveMonitoringState,
  shouldShowBlindTest
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

function formatCount(value) {
  const parsed = Number(value || 0)
  return Number.isFinite(parsed) ? parsed.toLocaleString('zh-CN') : '0'
}

function formatDuration(seconds) {
  const parsed = Number(seconds || 0)
  if (!Number.isFinite(parsed) || parsed <= 0) return monitoringCopy.noValue
  if (parsed < 60) return `${parsed.toFixed(0)} s`
  const minutes = Math.floor(parsed / 60)
  const remain = Math.floor(parsed % 60)
  return `${minutes}m ${String(remain).padStart(2, '0')}s`
}

function resolveSeverityTone(severity) {
  const value = String(severity || '').toLowerCase()
  if (['critical', 'high', 'danger'].includes(value)) return 'danger'
  if (['medium', 'warning', 'warn'].includes(value)) return 'warn'
  if (['low', 'info'].includes(value)) return 'ok'
  return 'pending'
}

function formatMs(value) {
  if (value === null || value === undefined || value === '') return monitoringCopy.noValue
  const parsed = Number(value)
  if (!Number.isFinite(parsed)) return monitoringCopy.noValue
  return `${parsed.toFixed(parsed >= 100 ? 0 : 1)} ms`
}

function formatRate(value) {
  if (value === null || value === undefined || value === '') return monitoringCopy.noValue
  const parsed = Number(value)
  if (!Number.isFinite(parsed)) return monitoringCopy.noValue
  return `${(parsed * 100).toFixed(1)}%`
}

function formatInputContractZeroGroups(groups) {
  const labels = {
    actuator_outputs: '动力输出',
    imu_accel_z: 'Z 轴加速度',
    motion_reference: '运动参考'
  }
  return (Array.isArray(groups) ? groups : [])
    .map((item) => labels[item] || item)
    .filter(Boolean)
    .join('、')
}

export function useMonitoringDashboard() {
  const router = useRouter()

  const runStatus = reactive({})
  const actionLoading = reactive(createActionLoadingState())
  const onlineModels = ref([])
  const replayAssets = ref([])
  const rflyDiagnostics = ref(null)
  const rflyDiagnosticsLoading = ref(false)
  const sessions = ref([])
  const sessionDetail = ref(null)
  const detailVisible = ref(false)
  const realtime = ref({ times: [] })
  const networkComparison = ref({ profiles: [], complete: false })
  const selectedBackend = ref('replay')
  const selectedDuration = ref(60)
  const selectedModel = ref('')
  const selectedReplayAssetId = ref(null)
  const rflyPx4mavctrlPath = ref('')
  const rflyNetworkProfile = ref('enhanced_5g')
  const rflyInferenceLocation = ref('mec')
  const faultHandoffDialogVisible = ref(false)
  const faultHandoffPayload = ref({
    session_id: null,
    fault_events_count: 0
  })

  let pollTimer = null

  const currentState = computed(() => resolveMonitoringState(runStatus))
  const telemetryCapabilities = computed(() => runStatus.telemetry_capabilities || {})
  const backendOptions = computed(() => resolveMonitoringBackendOptions(telemetryCapabilities.value))
  const blindTestEnabled = computed(() => shouldShowBlindTest(telemetryCapabilities.value))
  const rflyReady = computed(() => rflyDiagnostics.value?.ready === true)
  const stateLabel = computed(() => monitoringStateLabels[currentState.value] || monitoringStateLabels.IDLE)
  const alerts = computed(() => {
    return (runStatus.fault_events || []).map((item) => ({
      ...item,
      tone: resolveSeverityTone(item.diagnosis?.severity || item.severity)
    }))
  })
  const riskScore = computed(() => Number(runStatus.risk_score || 0).toFixed(2))
  const realtimeCount = computed(() => realtime.value?.times?.length || 0)
  const latestTimestamp = computed(() => formatLatestValue(realtime.value?.times, (value) => `${value} s`))
  const currentSessionId = computed(() => runStatus.current_session_id || monitoringCopy.noSession)
  const selectedBackendLabel = computed(() => {
    const option = monitoringBackendOptions.find((item) => item.value === selectedBackend.value)
    return option?.label || selectedBackend.value || monitoringCopy.noSelection
  })
  const activeModelLabel = computed(() => {
    if (!selectedModel.value) return monitoringCopy.noSelection
    const option = onlineModels.value.find((item) => item.name === selectedModel.value)
    return option?.display_name || option?.name || selectedModel.value
  })
  const compactModelLabel = computed(() => formatMonitoringModelName(activeModelLabel.value))
  const networkLink = computed(() => runStatus.network_link || {})
  const networkProcesses = computed(() => runStatus.processes || {})
  const mec = computed(() => runStatus.mec || {})
  const inference = computed(() => runStatus.inference || {})
  const networkLatencySeries = computed(() => (
    Array.isArray(realtime.value?.net_latency_ms) ? realtime.value.net_latency_ms : []
  ))
  const networkLossSeries = computed(() => (
    Array.isArray(realtime.value?.net_loss_rate) ? realtime.value.net_loss_rate : []
  ))
  const networkIntegritySeries = computed(() => (
    Array.isArray(realtime.value?.net_data_integrity) ? realtime.value.net_data_integrity : []
  ))
  const rflyNetworkProfileLabel = computed(() => {
    const activeProfile = networkLink.value?.enabled
      ? networkLink.value?.profile
      : rflyNetworkProfile.value
    const option = rflyNetworkProfileOptions.find((item) => item.value === activeProfile)
    return option?.label || activeProfile || monitoringCopy.noValue
  })
  const networkLatencyText = computed(() => formatMs(networkLink.value?.avg_latency_ms))
  const networkP95Text = computed(() => formatMs(networkLink.value?.p95_latency_ms))
  const networkLossText = computed(() => (
    networkLink.value?.enabled
      ? formatRate(networkLink.value?.loss_rate)
      : monitoringCopy.noValue
  ))
  const networkIntegrityText = computed(() => (
    networkLink.value?.enabled
      ? formatRate(networkLink.value?.data_integrity)
      : monitoringCopy.noValue
  ))
  const networkTone = computed(() => {
    if (!networkLink.value?.enabled) return 'offline'
    const lossRate = Number(networkLink.value?.loss_rate || 0)
    const p95Latency = Number(networkLink.value?.p95_latency_ms || 0)
    if (lossRate >= 0.03 || p95Latency >= 120) return 'danger'
    if (lossRate >= 0.01 || p95Latency >= 60) return 'warn'
    return 'ok'
  })
  const faultPeakText = computed(() => {
    const values = Array.isArray(realtime.value?.fault_prob) ? realtime.value.fault_prob : []
    const numeric = values.map((item) => Number(item)).filter((item) => Number.isFinite(item))
    if (!numeric.length) return '0.00'
    return Math.max(...numeric).toFixed(2)
  })
  const inputContractStatus = computed(() => runStatus.input_contract_status || {})
  const inputContractLabel = computed(() => {
    if (inputContractStatus.value?.detection_blocked) return '检测暂停'
    const status = String(inputContractStatus.value?.status || '').toLowerCase()
    if (status === 'healthy') return '字段正常'
    if (status === 'degraded') return '字段降级'
    return '等待输入'
  })
  const inputContractMeta = computed(() => {
    const actuatorSource = inputContractStatus.value?.actuator_source || 'unknown'
    const reason = inputContractStatus.value?.reason || 'waiting'
    const zeroGroups = formatInputContractZeroGroups(inputContractStatus.value?.zero_signal_groups)
    const blocked = inputContractStatus.value?.detection_blocked
      ? ' / 检测已暂停'
      : ''
    const zeroSummary = zeroGroups
      ? ` / 零信号：${zeroGroups}`
      : ''
    return `动力字段：${actuatorSource} / ${reason}${blocked}${zeroSummary}`
  })
  const stateTone = computed(() => {
    if (currentState.value === 'DETECTING') return 'ok'
    if (currentState.value === 'COLLECTING') return 'active'
    if (currentState.value === 'BLIND_TEST') return 'pending'
    return 'offline'
  })
  const contractTone = computed(() => {
    if (inputContractStatus.value?.detection_blocked) return 'danger'
    const status = String(inputContractStatus.value?.status || '').toLowerCase()
    if (status === 'healthy') return 'ok'
    if (status === 'degraded') return 'warn'
    return 'offline'
  })
  const riskTone = computed(() => {
    const score = Number(riskScore.value)
    if (score >= 0.7) return 'danger'
    if (score >= 0.45) return 'warn'
    return 'ok'
  })

  const estimatedSampleRate = computed(() => {
    const times = Array.isArray(realtime.value?.times)
      ? realtime.value.times.map((item) => Number(item)).filter((item) => Number.isFinite(item))
      : []
    if (times.length < 2) return monitoringCopy.noValue

    const deltas = []
    for (let index = 1; index < times.length; index += 1) {
      const delta = times[index] - times[index - 1]
      if (delta > 0) deltas.push(delta)
    }
    if (!deltas.length) return monitoringCopy.noValue

    const averageDelta = deltas.reduce((sum, item) => sum + item, 0) / deltas.length
    if (!Number.isFinite(averageDelta) || averageDelta <= 0) return monitoringCopy.noValue
    return `${(1 / averageDelta).toFixed(1)} Hz`
  })

  function resolveAltitudeFromPosZ(rawValue) {
    const raw = Number(rawValue)
    if (Number.isNaN(raw)) return NaN
    if (raw === 0) return 0

    const backend = String(runStatus.scenario_backend || '').toLowerCase()
    if (backend === 'px4' || backend === 'rfly') return -raw

    if (backend === 'replay') {
      const series = (Array.isArray(realtime.value?.pos_z) ? realtime.value.pos_z : [])
        .map((item) => Number(item))
        .filter((item) => !Number.isNaN(item))
      if (series.length) {
        const negatives = series.filter((item) => item < 0).length
        const positives = series.filter((item) => item > 0).length
        if (negatives > positives) return -raw
      }
      return raw
    }

    return raw
  }

  const latestAltitude = computed(() => {
    return formatLatestValue(realtime.value?.pos_z, (value) => {
      const altitude = resolveAltitudeFromPosZ(value)
      if (Number.isNaN(altitude)) return monitoringCopy.noValue
      return `${altitude.toFixed(2)} m`
    })
  })

  const latestRoll = computed(() => formatLatestValue(realtime.value?.roll, (value) => `${Number(value).toFixed(2)}°`))
  const latestPitch = computed(() => formatLatestValue(realtime.value?.pitch, (value) => `${Number(value).toFixed(2)}°`))
  const realtimeChartTimes = computed(() => Array.isArray(realtime.value?.times) ? realtime.value.times : [])
  const realtimeFaultProb = computed(() => Array.isArray(realtime.value?.fault_prob) ? realtime.value.fault_prob : [])

  const realtimeMotorSeries = computed(() => {
    return [
      { name: 'Motor 1', field: 'motor1' },
      { name: 'Motor 2', field: 'motor2' },
      { name: 'Motor 3', field: 'motor3' },
      { name: 'Motor 4', field: 'motor4' }
    ]
      .map((item) => ({
        name: item.name,
        data: Array.isArray(realtime.value?.[item.field]) ? realtime.value[item.field] : []
      }))
      .filter((item) => item.data.length > 0)
  })

  const realtimeAttitudeSeries = computed(() => {
    return [
      { name: 'Roll', field: 'roll' },
      { name: 'Pitch', field: 'pitch' },
      { name: 'Yaw', field: 'yaw' }
    ]
      .map((item) => ({
        name: item.name,
        data: Array.isArray(realtime.value?.[item.field]) ? realtime.value[item.field] : []
      }))
      .filter((item) => item.data.length > 0)
  })

  const realtimePositionSeries = computed(() => {
    const source = Array.isArray(realtime.value?.pos_z) ? realtime.value.pos_z : []
    if (!source.length) return []

    return [{
      name: 'Altitude',
      data: source.map((value) => {
        const altitude = resolveAltitudeFromPosZ(value)
        return Number.isNaN(altitude) ? null : altitude
      })
    }]
  })

  const metrics = computed(() => {
    const cards = [
      {
        label: monitoringCopy.heroStatusLabel,
        value: stateLabel.value,
        meta: `会话 ${currentSessionId.value}`,
        tone: stateTone.value,
        icon: 'Monitor'
      },
      {
        label: '风险评分',
        value: riskScore.value,
        meta: `故障概率峰值 ${faultPeakText.value}`,
        tone: riskTone.value,
        icon: 'Warning'
      },
      {
        label: '在线检测模型',
        value: compactModelLabel.value,
        meta: selectedModel.value || monitoringCopy.noSelection,
        tone: 'active',
        icon: 'Cpu'
      },
      {
        label: '字段合同',
        value: inputContractLabel.value,
        meta: inputContractMeta.value,
        tone: contractTone.value,
        icon: 'DocumentChecked'
      },
      {
        label: '实时样本',
        value: formatCount(realtimeCount.value),
        meta: `最近时间 ${latestTimestamp.value}`,
        tone: realtimeCount.value > 0 ? 'ok' : 'offline',
        icon: 'DataLine'
      },
      {
        label: '估算采样率',
        value: estimatedSampleRate.value,
        meta: '基于实时 times 间隔计算',
        tone: estimatedSampleRate.value === monitoringCopy.noValue ? 'offline' : 'ok',
        icon: 'Timer'
      }
    ]
    if (selectedBackend.value === 'rfly') {
      cards.push(
        {
          label: '5G链路场景',
          value: rflyNetworkProfileLabel.value,
          meta: `均值 ${networkLatencyText.value} / P95 ${networkP95Text.value}`,
          tone: networkTone.value,
          icon: 'DataLine'
        },
        {
          label: '链路完整率',
          value: networkIntegrityText.value,
          meta: `丢包 ${networkLossText.value}`,
          tone: networkTone.value,
          icon: 'DocumentChecked'
        }
      )
    }
    return cards
  })

  const sampleSummaryItems = computed(() => [
    { label: '采样点数量', value: formatCount(realtimeCount.value) },
    { label: '估算采样率', value: estimatedSampleRate.value },
    { label: '最新高度', value: latestAltitude.value },
    { label: '最新滚转角', value: latestRoll.value },
    { label: '最新俯仰角', value: latestPitch.value },
    { label: '最近样本时间', value: latestTimestamp.value }
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
    const allowedValues = new Set(backendOptions.value.map((item) => item.value))
    if (!allowedValues.has(selectedBackend.value)) {
      selectedBackend.value = backendOptions.value[0]?.value || 'replay'
    }
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

  async function loadRflyDiagnostics() {
    rflyDiagnosticsLoading.value = true
    try {
      const response = await telemetryApi.rflyDiagnostics()
      rflyDiagnostics.value = response.data || null
    } finally {
      rflyDiagnosticsLoading.value = false
    }
  }

  async function testRflyDiagnostics() {
    rflyDiagnosticsLoading.value = true
    try {
      const rflyPath = String(rflyPx4mavctrlPath.value || '').trim()
      const response = await telemetryApi.testRflyDiagnostics({
        rfly_px4mavctrl_path: rflyPath || undefined
      })
      rflyDiagnostics.value = response.data || null
      if (response.data?.ready) {
        ElMessage.success(monitoringCopy.rflyDiagnosticsReady)
      } else {
        ElMessage.warning(response.data?.summary?.message || monitoringCopy.rflyNotReady)
      }
      return response.data
    } catch (error) {
      ElMessage.error(`RflySim 配置诊断失败：${error.message || '未知错误'}`)
      throw error
    } finally {
      rflyDiagnosticsLoading.value = false
    }
  }

  async function loadSessions() {
    const response = await telemetryApi.sessions({ page: 1, per_page: 20 })
    sessions.value = response.data?.items || []
  }

  async function loadNetworkComparison() {
    const response = await telemetryApi.networkComparison()
    networkComparison.value = response.data || { profiles: [], complete: false }
  }

  async function refreshDashboard() {
    await Promise.all([
      fetchStatus(),
      loadSessions(),
      fetchRealtime(),
      loadNetworkComparison()
    ])
  }

  function startPolling() {
    stopPolling()
    pollTimer = setInterval(async () => {
      try {
        await refreshDashboard()
      } catch {
        // Polling errors are surfaced on user-triggered actions.
      }
    }, 4000)
  }

  async function startCollection() {
    await runAction('startCollection', async () => {
      const allowedBackends = telemetryCapabilities.value?.allowed_backends || []
      if (allowedBackends.length && !allowedBackends.includes(selectedBackend.value)) {
        ElMessage.warning(monitoringCopy.backendNotAllowed)
        return
      }
      if (selectedBackend.value === 'replay' && !selectedReplayAssetId.value) {
        ElMessage.warning(monitoringCopy.replayAssetRequired)
        return
      }
      if (selectedBackend.value === 'rfly') {
        const diagnostics = rflyDiagnostics.value?.ready
          ? rflyDiagnostics.value
          : await testRflyDiagnostics()
        if (diagnostics?.ready !== true) {
          ElMessage.warning(diagnostics?.summary?.message || monitoringCopy.rflyNotReady)
          return
        }
      }

      const replayAsset = replayAssets.value.find((item) => item.id === selectedReplayAssetId.value)
      const rflyPath = String(rflyPx4mavctrlPath.value || '').trim()
      try {
        await telemetryApi.startCollection({
          drone_id: 'DRONE-001',
          session_type: 'simulation',
          sim_time: selectedBackend.value === 'replay' ? 0 : selectedDuration.value,
          backend: selectedBackend.value,
          replay_path: selectedBackend.value === 'replay'
            ? replayAsset?.meta?.prepared_csv_path || replayAsset?.stored_path
            : undefined,
          replay_asset_id: selectedBackend.value === 'replay'
            ? replayAsset?.id
            : undefined,
          rfly_px4mavctrl_path: selectedBackend.value === 'rfly' && rflyPath
            ? rflyPath
            : undefined,
          rfly_network_profile: selectedBackend.value === 'rfly'
            ? rflyNetworkProfile.value
            : undefined,
          rfly_inference_location: selectedBackend.value === 'rfly'
            ? rflyInferenceLocation.value
            : undefined,
          model: selectedBackend.value === 'rfly'
            ? selectedModel.value || undefined
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

        if (Number(response.data?.fault_events_count || 0) <= 0) return

        openFaultHandoffDialog(response.data)
      } catch (error) {
        ElMessage.error(`停止采集失败：${error.message || '未知错误'}`)
      }
    })
  }

  function openFaultHandoffDialog(payload = {}) {
    faultHandoffPayload.value = {
      session_id: payload.session_id || null,
      fault_events_count: Number(payload.fault_events_count || 0)
    }
    faultHandoffDialogVisible.value = true
  }

  function dismissFaultHandoff() {
    faultHandoffDialogVisible.value = false
  }

  function confirmFaultHandoff() {
    const sessionId = faultHandoffPayload.value?.session_id
    faultHandoffDialogVisible.value = false
    router.push({
      path: '/diagnosis',
      query: {
        ...(sessionId ? { session_id: sessionId } : {}),
        auto: '1'
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
      await Promise.all([loadModels(), loadReplayAssets(), loadRflyDiagnostics(), refreshDashboard()])
      if (currentState.value !== 'IDLE') {
        startPolling()
      }
    } catch (error) {
      ElMessage.error(`初始化监测页失败：${error.message || '未知错误'}`)
    }
  })

  watch(selectedBackend, (backend) => {
    if (backend === 'rfly' && !rflyDiagnostics.value) {
      loadRflyDiagnostics().catch(() => {})
    }
  })

  watch(telemetryCapabilities, (capabilities) => {
    const defaultProfile = capabilities?.rfly_default_network_profile
    const currentProfileOk = rflyNetworkProfileOptions.some(
      (item) => item.value === rflyNetworkProfile.value
    )
    if (
      defaultProfile &&
      !currentProfileOk &&
      rflyNetworkProfileOptions.some((item) => item.value === defaultProfile)
    ) {
      rflyNetworkProfile.value = defaultProfile
    }
    const defaultLocation = capabilities?.rfly_default_inference_location
    if (
      defaultLocation &&
      rflyInferenceLocationOptions.some((item) => item.value === defaultLocation)
    ) {
      rflyInferenceLocation.value = defaultLocation
    }
  })

  onUnmounted(stopPolling)

  return {
    actionLoading,
    activeModelLabel,
    alerts,
    backendOptions,
    blindTestEnabled,
    continueDiagnosis,
    currentSessionId,
    currentState,
    detailVisible,
    estimatedSampleRate,
    faultHandoffDialogVisible,
    faultHandoffPayload,
    inputContractLabel,
    inputContractMeta,
    inference,
    loadNetworkComparison,
    loadSessions,
    metrics,
    mec,
    networkComparison,
    networkIntegritySeries,
    networkLatencySeries,
    networkLink,
    networkLossSeries,
    networkProcesses,
    onlineModels,
    openFaultHandoffDialog,
    openSessionDetail,
    replayAssets,
    realtimeChartTimes,
    realtimeFaultProb,
    realtimeAttitudeSeries,
    realtimeMotorSeries,
    realtimePositionSeries,
    rflyDiagnostics,
    rflyDiagnosticsLoading,
    rflyInferenceLocation,
    rflyNetworkProfile,
    rflyPx4mavctrlPath,
    rflyReady,
    riskScore,
    sampleSummaryItems,
    selectedBackend,
    selectedBackendLabel,
    selectedDuration,
    selectedModel,
    selectedReplayAssetId,
    sessionDetail,
    sessions,
    startBlindTest,
    startCollection,
    startDetection,
    stateLabel,
    confirmFaultHandoff,
    dismissFaultHandoff,
    stopCollection,
    stopDetection,
    testRflyDiagnostics
  }
}
