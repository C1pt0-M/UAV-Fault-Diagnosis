import { computed, onMounted, onUnmounted, reactive, ref, watch } from 'vue'
import { useRouter } from 'vue-router'
import { ElMessage, ElMessageBox } from 'element-plus'
import { analysisApi, assetApi, runApi } from '@/shared/services'
import { assetsCopy } from '@/modules/assets/assetsView'

export function useAssetsWorkspace() {
  const router = useRouter()
  const activeKind = ref('all')
  const assets = ref([])
  const loading = ref(false)
  const uploading = ref(false)
  const detailVisible = ref(false)
  const detailTarget = ref(null)
  const preparingMap = reactive({})
  const deletingMap = reactive({})
  const pollTimers = new Map()
  const analysisAssetId = ref(null)
  const analysisFields = ref([])
  const analysisFieldOptions = ref([])
  const analysisStartT = ref()
  const analysisEndT = ref()
  const analysisMaxPoints = ref(2000)
  const analysisChartTimes = ref([])
  const analysisChartSeries = ref([])
  const analysisLoadingFields = ref(false)
  const analysisLoadingSeries = ref(false)
  const analysisErrorMessage = ref('')
  const analysisDisplayMap = ref({})

  let activeUploads = 0

  const detailJson = computed(() => JSON.stringify(detailTarget.value || {}, null, 2))
  const analyzableAssets = computed(() => {
    return (assets.value || []).filter((item) => ['csv', 'ulg'].includes(String(item.kind || '').toLowerCase()))
  })

  function resetAnalysisSeries() {
    analysisChartTimes.value = []
    analysisChartSeries.value = []
  }

  function pickDefaultFields(options = []) {
    const preferred = ['motor1', 'motor2', 'motor3', 'motor4', 'roll', 'pitch', 'yaw', 'pos_z']
    const available = new Set(options.map((item) => item.value))
    const selected = preferred.filter((field) => available.has(field))
    if (selected.length > 0) {
      return selected.slice(0, 4)
    }
    return options.slice(0, 4).map((item) => item.value)
  }

  async function loadAssets() {
    loading.value = true

    try {
      const response = await assetApi.list(activeKind.value)
      assets.value = response.data || []
      if (
        analysisAssetId.value === null ||
        !analyzableAssets.value.some((item) => item.id === analysisAssetId.value)
      ) {
        analysisAssetId.value = analyzableAssets.value[0]?.id ?? null
      }
    } catch (error) {
      ElMessage.error(`加载资产失败：${error.message || '未知错误'}`)
    } finally {
      loading.value = false
    }
  }

  async function loadAnalysisFields() {
    if (!analysisAssetId.value) {
      analysisFieldOptions.value = []
      analysisFields.value = []
      analysisDisplayMap.value = {}
      resetAnalysisSeries()
      return
    }
    analysisLoadingFields.value = true
    analysisErrorMessage.value = ''

    try {
      const response = await analysisApi.fields(analysisAssetId.value)
      const groups = response.data?.field_groups || []
      const flatOptions = groups.flatMap((group) =>
        (group.fields || []).map((field) => ({
          label: `${group.group} / ${field.display_name || field.field}`,
          value: field.field,
          displayName: field.display_name || field.field,
          group: group.group
        }))
      )
      analysisFieldOptions.value = flatOptions
      analysisDisplayMap.value = flatOptions.reduce((mapping, item) => {
        mapping[item.value] = item.displayName
        return mapping
      }, {})

      const validFields = analysisFields.value.filter((field) =>
        flatOptions.some((item) => item.value === field)
      )
      analysisFields.value = validFields.length > 0 ? validFields : pickDefaultFields(flatOptions)
      analysisStartT.value = response.data?.time?.min
      analysisEndT.value = response.data?.time?.max
      await loadAnalysisSeries()
    } catch (error) {
      analysisFieldOptions.value = []
      analysisFields.value = []
      analysisDisplayMap.value = {}
      resetAnalysisSeries()
      analysisErrorMessage.value = error.message || assetsCopy.analysisLoadFieldsFailed
    } finally {
      analysisLoadingFields.value = false
    }
  }

  async function loadAnalysisSeries() {
    if (!analysisAssetId.value || analysisFields.value.length === 0) {
      resetAnalysisSeries()
      return
    }
    analysisLoadingSeries.value = true
    analysisErrorMessage.value = ''

    try {
      const response = await analysisApi.timeseries({
        assetId: analysisAssetId.value,
        fields: analysisFields.value,
        startT: analysisStartT.value,
        endT: analysisEndT.value,
        maxPoints: analysisMaxPoints.value
      })
      const payload = response.data || {}
      analysisChartTimes.value = payload.x || []
      analysisChartSeries.value = (payload.series || []).map((series) => ({
        name: analysisDisplayMap.value[series.field] || series.field,
        data: series.values || []
      }))
      if (analysisChartSeries.value.length === 0) {
        analysisErrorMessage.value = assetsCopy.analysisNoSeries
      }
    } catch (error) {
      resetAnalysisSeries()
      analysisErrorMessage.value = error.message || assetsCopy.analysisLoadSeriesFailed
    } finally {
      analysisLoadingSeries.value = false
    }
  }

  async function handleUpload(options) {
    activeUploads += 1
    uploading.value = true

    try {
      await assetApi.upload(options.file)
      ElMessage.success(`文件 ${options.file.name} 上传成功`)
      options.onSuccess?.()
    } catch (error) {
      ElMessage.error(`上传 ${options.file.name} 失败：${error.message || '未知错误'}`)
      options.onError?.(error)
    } finally {
      activeUploads -= 1

      if (activeUploads === 0) {
        uploading.value = false
        await loadAssets()
      }
    }
  }

  function stopPreparePolling(runId) {
    const timer = pollTimers.get(runId)
    if (timer) {
      clearInterval(timer)
      pollTimers.delete(runId)
    }
  }

  function pollPrepareRun(runId, assetId) {
    if (pollTimers.has(runId)) {
      return
    }

    const timer = setInterval(async () => {
      try {
        const response = await runApi.detail(runId)

        if (response.data?.status === 'success') {
          stopPreparePolling(runId)
          preparingMap[assetId] = false
          ElMessage.success(assetsCopy.prepareSuccess)
          await loadAssets()
        }

        if (response.data?.status === 'error') {
          stopPreparePolling(runId)
          preparingMap[assetId] = false
          ElMessage.error(response.data.error || '资产预处理失败')
        }
      } catch (error) {
        stopPreparePolling(runId)
        preparingMap[assetId] = false
        ElMessage.error(`查询预处理任务失败：${error.message || '未知错误'}`)
      }
    }, 2000)

    pollTimers.set(runId, timer)
  }

  async function prepareAsset(row) {
    preparingMap[row.id] = true

    try {
      const response = await assetApi.prepare(row.id)

      if (response.data?.id) {
        pollPrepareRun(response.data.id, row.id)
        return
      }

      preparingMap[row.id] = false
      ElMessage.success(assetsCopy.prepareAlreadyDone)
      await loadAssets()
    } catch (error) {
      preparingMap[row.id] = false
      ElMessage.error(`资产预处理失败：${error.message || '未知错误'}`)
    }
  }

  async function downloadAsset(row) {
    try {
      await assetApi.download(row.id, row.filename)
    } catch (error) {
      ElMessage.error(`下载失败：${error.message || '未知错误'}`)
    }
  }

  function openReportsCenter() {
    router.push('/reports')
  }

  function openReports(row) {
    router.push({
      path: '/reports',
      query: {
        asset_id: String(row.id)
      }
    })
  }

  function showDetail(row) {
    detailTarget.value = row
    detailVisible.value = true
  }

  async function removeAsset(row) {
    try {
      await ElMessageBox.confirm(
        assetsCopy.removeConfirmMessage(row),
        assetsCopy.removeConfirmTitle,
        {
          type: 'warning',
          confirmButtonText: '删除',
          cancelButtonText: '取消'
        }
      )
    } catch {
      return
    }

    deletingMap[row.id] = true

    try {
      await assetApi.remove(row.id)
      ElMessage.success(assetsCopy.removeSuccess)
      await loadAssets()
    } catch (error) {
      ElMessage.error(`删除失败：${error.message || '未知错误'}`)
    } finally {
      deletingMap[row.id] = false
    }
  }

  watch(analysisAssetId, async (value, oldValue) => {
    if (value === oldValue) return
    await loadAnalysisFields()
  })

  onMounted(loadAssets)

  onUnmounted(() => {
    pollTimers.forEach((timer) => clearInterval(timer))
    pollTimers.clear()
  })

  return {
    activeKind,
    analysisAssetId,
    analysisChartSeries,
    analysisChartTimes,
    analysisEndT,
    analysisErrorMessage,
    analysisFieldOptions,
    analysisFields,
    analysisLoadingFields,
    analysisLoadingSeries,
    analysisMaxPoints,
    analysisStartT,
    assets,
    deletingMap,
    detailJson,
    detailVisible,
    downloadAsset,
    handleUpload,
    loadAssets,
    loadAnalysisFields,
    loadAnalysisSeries,
    loading,
    openReports,
    openReportsCenter,
    prepareAsset,
    preparingMap,
    removeAsset,
    showDetail,
    uploading
  }
}
