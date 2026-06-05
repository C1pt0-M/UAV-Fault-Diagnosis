<template>
  <div class="page-stack">
    <DiagnosisSessionBanner :active-session-id="activeSessionId" />

    <section class="two-column">
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
        :report-creating="reportCreating"
        :result-ready="Boolean(resultData)"
        @create-report="createReport"
      />
    </section>

    <DiagnosisResultPanel
      v-if="resultData"
      :result-data="resultData"
      :confidence-text="confidenceText"
      :probability-rows="probabilityRows"
    />
  </div>
</template>

<script setup>
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
  loadMeta,
  models,
  probabilityRows,
  progressPercentage,
  reportCreating,
  resultData,
  running,
  selectedAssetId,
  selectedModel,
  startDiagnosis
} = useDiagnosisWorkspace()
</script>
