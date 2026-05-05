<template>
  <div class="page-stack">
    <div class="reports-statusbar">
      <div class="reports-statusbar__item">
        <span class="reports-statusbar__label">当前筛选</span>
        <span class="reports-statusbar__value">{{ currentFilterSummary }}</span>
      </div>
      <div class="reports-statusbar__divider"></div>
      <div class="reports-statusbar__item">
        <span class="reports-statusbar__label">当前页报告数</span>
        <span class="reports-statusbar__value reports-statusbar__value--mono">{{ reports.length }}</span>
      </div>
      <div class="reports-statusbar__divider"></div>
      <div class="reports-statusbar__item">
        <span class="reports-statusbar__hint">通过资产或标签快速收拢报告范围</span>
      </div>
    </div>

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
      :report="detailTarget"
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
  currentFilterSummary,
  detailTarget,
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

<style scoped>
.reports-statusbar {
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: var(--space-sm);
  padding: var(--space-sm) var(--space-lg);
  border: 1px solid var(--line-soft);
  border-radius: var(--radius-md);
  background: var(--bg-panel-raised);
  font-size: 12px;
  color: var(--text-secondary);
}

.reports-statusbar__item {
  display: flex;
  align-items: center;
  gap: 6px;
  min-width: 0;
}

.reports-statusbar__label {
  color: var(--text-tertiary);
  font-size: 11px;
  white-space: nowrap;
}

.reports-statusbar__value {
  color: var(--text-primary);
  font-weight: 600;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  max-width: 360px;
}

.reports-statusbar__value--mono {
  font-family: var(--font-mono);
  font-size: 11px;
}

.reports-statusbar__hint {
  color: var(--text-tertiary);
  font-size: 11px;
  white-space: nowrap;
}

.reports-statusbar__divider {
  width: 1px;
  height: 14px;
  background: var(--line-strong);
  flex-shrink: 0;
}
</style>
