<template>
  <div class="page-stack">
    <DiagnosisSessionBanner :active-session-id="activeSessionId" />

    <DiagnosisCommandHero
      :active-session-id="activeSessionId"
      :current-run="currentRun"
      :selected-asset-name="selectedAssetName"
      :selected-model-name="selectedModelName"
      :stage="stage"
    />

    <section class="two-column diagnosis-command-grid">
      <DiagnosisTaskForm
        v-model:selected-asset-id="selectedAssetId"
        v-model:selected-model="selectedModel"
        :assets="assets"
        :models="models"
        :running="running"
        @submit="startDiagnosis"
        @refresh="loadMeta"
      />

      <DiagnosisStatusPanel
        :current-run="currentRun"
        :progress-percentage="progressPercentage"
        :stage="stage"
      />
    </section>

    <DiagnosisResultPanel
      :result-data="resultData"
      :confidence-text="confidenceText"
      :latest-report-id="latestReportId"
      :probability-rows="probabilityRows"
      :report-creating="reportCreating"
      :stage="stage"
      @create-report="createReport"
      @open-reports-center="openReportsCenter"
      @open-ai-assistant="openReportAssistant"
    />
  </div>
</template>

<script setup>
import DiagnosisCommandHero from '@/modules/diagnosis/components/DiagnosisCommandHero.vue'
import DiagnosisResultPanel from '@/modules/diagnosis/components/DiagnosisResultPanel.vue'
import DiagnosisSessionBanner from '@/modules/diagnosis/components/DiagnosisSessionBanner.vue'
import DiagnosisStatusPanel from '@/modules/diagnosis/components/DiagnosisStatusPanel.vue'
import DiagnosisTaskForm from '@/modules/diagnosis/components/DiagnosisTaskForm.vue'
import { useDiagnosisWorkspace } from '@/modules/diagnosis/composables/useDiagnosisWorkspace'

const {
  activeSessionId,
  assets,
  confidenceText,
  createReport,
  currentRun,
  latestReportId,
  loadMeta,
  models,
  openReportAssistant,
  openReportsCenter,
  probabilityRows,
  progressPercentage,
  reportCreating,
  resultData,
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
.diagnosis-command-grid {
  align-items: start;
}
</style>
