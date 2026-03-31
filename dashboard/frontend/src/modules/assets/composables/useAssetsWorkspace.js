import { computed, onMounted, onUnmounted, reactive, ref } from 'vue'
import { useRouter } from 'vue-router'
import { ElMessage, ElMessageBox } from 'element-plus'
import { assetApi, runApi } from '@/shared/services'
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

  let activeUploads = 0

  const detailJson = computed(() => JSON.stringify(detailTarget.value || {}, null, 2))

  async function loadAssets() {
    loading.value = true

    try {
      const response = await assetApi.list(activeKind.value)
      assets.value = response.data || []
    } catch (error) {
      ElMessage.error(`加载资产失败：${error.message || '未知错误'}`)
    } finally {
      loading.value = false
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

  onMounted(loadAssets)

  onUnmounted(() => {
    pollTimers.forEach((timer) => clearInterval(timer))
    pollTimers.clear()
  })

  return {
    activeKind,
    assets,
    deletingMap,
    detailJson,
    detailVisible,
    downloadAsset,
    handleUpload,
    loadAssets,
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
