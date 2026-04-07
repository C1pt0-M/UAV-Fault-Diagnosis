<template>
  <div class="page-stack">
    <section class="reports-hero">
      <div class="reports-hero__main">
        <span class="reports-hero__eyebrow">报告中心</span>
        <h1>诊断结果与报告复核</h1>
        <p>{{ reportsCopy.description }}</p>
      </div>

      <div class="reports-hero__summary">
        <article class="reports-summary-card reports-summary-card--accent">
          <span>当前筛选</span>
          <strong>{{ currentFilterSummary }}</strong>
          <small>通过资产或标签快速收拢报告范围</small>
        </article>
        <article class="reports-summary-card">
          <span>当前页报告数</span>
          <strong>{{ reports.length }}</strong>
          <small>延续诊断页的结果中心语义</small>
        </article>
      </div>
    </section>

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
import { reportsCopy } from '@/modules/reports/reportsView'
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
.reports-hero {
  display: grid;
  grid-template-columns: minmax(0, 1.3fr) minmax(320px, 0.95fr);
  gap: 20px;
  padding: 28px;
  border-radius: 28px;
  background:
    radial-gradient(circle at right top, rgba(255, 255, 255, 0.18), transparent 30%),
    linear-gradient(140deg, #081f2f 0%, #0d4656 56%, #0f766e 100%);
  color: #f4fbfb;
  box-shadow: 0 24px 60px rgba(8, 31, 47, 0.22);
}

.reports-hero__eyebrow {
  display: inline-flex;
  padding: 7px 12px;
  border-radius: 999px;
  background: rgba(255, 255, 255, 0.14);
  font-size: 12px;
  font-weight: 800;
  letter-spacing: 0.04em;
}

.reports-hero__main h1 {
  margin: 18px 0 0;
  font-size: 42px;
  line-height: 1.04;
}

.reports-hero__main p {
  margin: 16px 0 0;
  max-width: 660px;
  line-height: 1.9;
  color: rgba(244, 251, 251, 0.86);
}

.reports-hero__summary {
  display: grid;
  gap: 12px;
}

.reports-summary-card {
  padding: 16px 18px;
  border-radius: 20px;
  background: rgba(255, 255, 255, 0.1);
  border: 1px solid rgba(255, 255, 255, 0.14);
}

.reports-summary-card--accent {
  background: rgba(255, 255, 255, 0.14);
}

.reports-summary-card span {
  font-size: 12px;
  color: rgba(244, 251, 251, 0.72);
}

.reports-summary-card strong {
  display: block;
  margin-top: 8px;
  font-size: 22px;
  line-height: 1.3;
}

.reports-summary-card small {
  display: block;
  margin-top: 6px;
  line-height: 1.7;
  color: rgba(244, 251, 251, 0.8);
}

@media (max-width: 1480px) {
  .reports-hero {
    grid-template-columns: 1fr;
  }
}
</style>
