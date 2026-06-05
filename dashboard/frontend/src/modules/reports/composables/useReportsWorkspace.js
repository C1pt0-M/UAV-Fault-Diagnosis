import { computed, onMounted, ref } from 'vue'
import { useRouter } from 'vue-router'
import { ElMessage, ElMessageBox } from 'element-plus'
import { assetApi, diagnosisReportApi } from '@/shared/services'
import { createDefaultReportFilters, reportsCopy } from '@/modules/reports/reportsView'

export function useReportsWorkspace() {
  const router = useRouter()
  const loading = ref(false)
  const reports = ref([])
  const assetOptions = ref([])
  const filters = ref(createDefaultReportFilters())
  const detailVisible = ref(false)
  const detailTarget = ref(null)

  const assetNameMap = computed(() => {
    return new Map(assetOptions.value.map((item) => [item.id, item.filename]))
  })

  const detailJson = computed(() => JSON.stringify(detailTarget.value || {}, null, 2))

  async function loadAssets() {
    const response = await assetApi.list('all')
    assetOptions.value = response.data || []
  }

  async function loadReports() {
    loading.value = true

    try {
      const response = await diagnosisReportApi.list({
        page: 1,
        per_page: 20,
        asset_id: filters.value.assetId,
        label: filters.value.label
      })
      reports.value = response.data || []
    } catch (error) {
      ElMessage.error(`加载报告失败：${error.message || '未知错误'}`)
    } finally {
      loading.value = false
    }
  }

  function handleReset() {
    filters.value = createDefaultReportFilters()
    loadReports()
  }

  function showDetail(row) {
    detailTarget.value = row
    detailVisible.value = true
  }

  function openInAssistant(row) {
    router.push({
      path: '/ai',
      query: {
        scene: 'report_analysis',
        report_id: String(row.id)
      }
    })
  }

  async function downloadReport(row) {
    try {
      await diagnosisReportApi.download(row.id, `diagnosis_report_${row.id}.pdf`)
    } catch (error) {
      ElMessage.error(`下载报告失败：${error.message || '未知错误'}`)
    }
  }

  async function removeReport(row) {
    try {
      await ElMessageBox.confirm(
        reportsCopy.removeConfirmMessage(row),
        reportsCopy.removeConfirmTitle,
        {
          type: 'warning',
          confirmButtonText: '删除',
          cancelButtonText: '取消'
        }
      )
    } catch {
      return
    }

    try {
      await diagnosisReportApi.remove(row.id)
      ElMessage.success(reportsCopy.removeSuccess)
      await loadReports()
    } catch (error) {
      ElMessage.error(`删除报告失败：${error.message || '未知错误'}`)
    }
  }

  onMounted(async () => {
    try {
      await loadAssets()
      await loadReports()
    } catch (error) {
      ElMessage.error(`初始化报告页失败：${error.message || '未知错误'}`)
    }
  })

  return {
    assetNameMap,
    assetOptions,
    detailJson,
    detailVisible,
    downloadReport,
    filters,
    handleReset,
    loadReports,
    loading,
    openInAssistant,
    removeReport,
    reports,
    showDetail
  }
}
