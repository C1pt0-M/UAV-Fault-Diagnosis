<template>
  <div class="page-stack diagnosis-workbench">
    <DiagnosisSessionBanner :active-session-id="activeSessionId" />

    <DiagnosisCommandHero
      :active-session-id="activeSessionId"
      :current-run="currentRun"
      :latest-updated-text="latestUpdatedText"
      :report-status-text="reportStatusText"
      :selected-asset-name="selectedAssetName"
      :selected-model-name="selectedModelName"
      :stage="stage"
    />

    <section class="diagnosis-workbench__grid">
      <DiagnosisTaskForm
        v-model:selected-asset-id="selectedAssetId"
        v-model:selected-model="selectedModel"
        :asset-summary-items="assetSummaryItems"
        :assets="assets"
        :flow-steps="flowSteps"
        :model-summary-items="modelSummaryItems"
        :models="models"
        :running="running"
        :stage="stage"
        @submit="startDiagnosis"
        @refresh="loadMeta"
      />

      <DiagnosisResultPanel
        :confidence-text="confidenceText"
        :latest-report-id="latestReportId"
        :probability-rows="probabilityRows"
        :report-creating="reportCreating"
        :result-data="resultData"
        :result-meta-items="resultMetaItems"
        :risk-level="riskLevel"
        :run-error-text="runErrorText"
        :stage="stage"
        @create-report="createReport"
        @open-reports-center="openReportsCenter"
        @open-ai-assistant="openReportAssistant"
      />

      <DiagnosisEvidencePanel
        :evidence-rows="evidenceRows"
        :explanation-summary="explanationSummary"
        :recent-reports="recentReports"
        :result-data="resultData"
        :risk-level="riskLevel"
        :rule-rows="ruleRows"
        @open-reports-center="openReportsCenter"
      />
    </section>

    <DiagnosisRecentRunsPanel
      :recent-runs="recentRuns"
      :report-status-text="reportStatusText"
      @open-run-result="openRunResult"
      @open-reports-center="openReportsCenter"
      @open-ai-assistant="openReportAssistant"
    />
  </div>
</template>

<script setup>
import DiagnosisCommandHero from '@/modules/diagnosis/components/DiagnosisCommandHero.vue'
import DiagnosisEvidencePanel from '@/modules/diagnosis/components/DiagnosisEvidencePanel.vue'
import DiagnosisRecentRunsPanel from '@/modules/diagnosis/components/DiagnosisRecentRunsPanel.vue'
import DiagnosisResultPanel from '@/modules/diagnosis/components/DiagnosisResultPanel.vue'
import DiagnosisSessionBanner from '@/modules/diagnosis/components/DiagnosisSessionBanner.vue'
import DiagnosisTaskForm from '@/modules/diagnosis/components/DiagnosisTaskForm.vue'
import { useDiagnosisWorkspace } from '@/modules/diagnosis/composables/useDiagnosisWorkspace'

const {
  activeSessionId,
  assetSummaryItems,
  assets,
  confidenceText,
  createReport,
  currentRun,
  evidenceRows,
  explanationSummary,
  flowSteps,
  latestReportId,
  latestUpdatedText,
  loadMeta,
  modelSummaryItems,
  models,
  openReportAssistant,
  openReportsCenter,
  openRunResult,
  probabilityRows,
  recentReports,
  recentRuns,
  reportCreating,
  reportStatusText,
  resultData,
  resultMetaItems,
  riskLevel,
  ruleRows,
  runErrorText,
  running,
  selectedAssetId,
  selectedAssetName,
  selectedModel,
  selectedModelName,
  stage,
  startDiagnosis
} = useDiagnosisWorkspace()
</script>

<style scoped>
.diagnosis-workbench {
  gap: var(--space-lg);
}

.diagnosis-workbench__grid {
  display: grid;
  grid-template-columns: minmax(320px, 34fr) minmax(360px, 34fr) minmax(340px, 32fr);
  gap: var(--space-lg);
  align-items: start;
}

@media (max-width: 1480px) {
  .diagnosis-workbench__grid {
    grid-template-columns: minmax(0, 1fr) minmax(380px, 1fr);
  }

  .diagnosis-workbench__grid > :last-child {
    grid-column: 1 / -1;
  }
}

@media (max-width: 1120px) {
  .diagnosis-workbench__grid {
    grid-template-columns: 1fr;
  }

  .diagnosis-workbench__grid > :last-child {
    grid-column: auto;
  }
}
</style>
