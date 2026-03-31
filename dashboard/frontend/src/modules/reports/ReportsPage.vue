<template>
  <div class="page-stack">
    <ReportsWorkspacePanel
      v-model:filters="filters"
      :asset-options="assetOptions"
      :asset-name-map="assetNameMap"
      :loading="loading"
      :reports="reports"
      @refresh="loadReports"
      @reset="handleReset"
      @show-detail="showDetail"
      @open-ai="openInAssistant"
      @download="downloadReport"
      @remove="removeReport"
    />

    <ReportDetailDialog
      v-model:visible="detailVisible"
      :detail-json="detailJson"
    />
  </div>
</template>

<script setup>
import ReportDetailDialog from '@/modules/reports/components/ReportDetailDialog.vue'
import ReportsWorkspacePanel from '@/modules/reports/components/ReportsWorkspacePanel.vue'
import { useReportsWorkspace } from '@/modules/reports/composables/useReportsWorkspace'

const {
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
} = useReportsWorkspace()
</script>
