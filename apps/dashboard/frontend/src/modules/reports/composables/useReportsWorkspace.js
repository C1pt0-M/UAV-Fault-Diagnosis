import { computed, onMounted, ref } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { ElMessage, ElMessageBox } from 'element-plus'
import { assetApi, diagnosisReportApi } from '@/shared/services'
import { createDefaultReportFilters, reportsCopy } from '@/modules/reports/reportsView'

export function useReportsWorkspace() {
  const route = useRoute()
  const router = useRouter()
  const loading = ref(false)
  const reports = ref([])
  const assetOptions = ref([])
  const filters = ref(createDefaultReportFilters())
  const pagination = ref({
    page: 1,
    perPage: 20,
    total: 0,
    pages: 0
  })
  const detailVisible = ref(false)
  const detailTarget = ref(null)

  const assetNameMap = computed(() => {
    return new Map(assetOptions.value.map((item) => [item.id, item.filename]))
  })

  const detailJson = computed(() => JSON.stringify(detailTarget.value || {}, null, 2))
  const currentFilterSummary = computed(() => {
    const parts = []
    if (filters.value.assetId) parts.push(`资产 #${filters.value.assetId}`)
    if (filters.value.label) parts.push(`标签 ${filters.value.label}`)
    const filterText = parts.length ? parts.join(' / ') : '当前查看全部报告'
    return `${filterText}，共 ${pagination.value.total || 0} 份`
  })

  async function loadAssets() {
    const response = await assetApi.list('all')
    assetOptions.value = response.data || []
  }

  async function loadReports() {
    loading.value = true

    try {
      const response = await diagnosisReportApi.list({
        page: filters.value.page,
        per_page: filters.value.perPage,
        asset_id: filters.value.assetId,
        label: filters.value.label
      })
      reports.value = response.data || []
      const responsePagination = response.pagination || {}
      pagination.value = {
        page: Number(responsePagination.page || filters.value.page || 1),
        perPage: Number(responsePagination.per_page || filters.value.perPage || 20),
        total: Number(responsePagination.total || 0),
        pages: Number(responsePagination.pages || 0)
      }
    } catch (error) {
      ElMessage.error(`加载报告失败：${error.message || '未知错误'}`)
    } finally {
      loading.value = false
    }
  }

  function syncFiltersFromRoute() {
    const routeAssetId = Number(route.query.asset_id || 0)
    if (routeAssetId > 0) {
      filters.value = {
        ...filters.value,
        assetId: routeAssetId,
        page: 1
      }
    }
  }

  function handleReset() {
    filters.value = createDefaultReportFilters()
    loadReports()
  }

  function handlePageChange(page) {
    filters.value = {
      ...filters.value,
      page
    }
    loadReports()
  }

  function handlePageSizeChange(perPage) {
    filters.value = {
      ...filters.value,
      page: 1,
      perPage
    }
    loadReports()
  }

  function handleFilterRefresh() {
    filters.value = {
      ...filters.value,
      page: 1
    }
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
        report_id: String(row.id),
        auto_analyze: '1'
      }
    })
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
      syncFiltersFromRoute()
      await loadReports()
    } catch (error) {
      ElMessage.error(`初始化报告页失败：${error.message || '未知错误'}`)
    }
  })

  return {
    assetNameMap,
    assetOptions,
    currentFilterSummary,
    detailTarget,
    detailJson,
    detailVisible,
    filters,
    handleReset,
    handleFilterRefresh,
    handlePageChange,
    handlePageSizeChange,
    loadReports,
    loading,
    openInAssistant,
    pagination,
    removeReport,
    reports,
    showDetail
  }
}
