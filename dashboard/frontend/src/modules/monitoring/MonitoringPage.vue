<template>
  <div class="page-stack">
    <MonitoringControlPanel
      v-model:selected-backend="selectedBackend"
      v-model:selected-duration="selectedDuration"
      v-model:selected-model="selectedModel"
      v-model:selected-replay-asset-id="selectedReplayAssetId"
      :action-loading="actionLoading"
      :backend-options="backendOptions"
      :blind-test-enabled="blindTestEnabled"
      :current-state="currentState"
      :online-models="onlineModels"
      :replay-assets="replayAssets"
      @start-collection="startCollection"
      @start-detection="startDetection"
      @start-blind-test="startBlindTest"
      @stop-detection="stopDetection"
      @stop-collection="stopCollection"
    />

    <MonitoringMetrics :metrics="metrics" />

    <MonitoringRealtimeChartPanel
      v-model:selected-fields="realtimeChartFields"
      :field-options="realtimeFieldOptions"
      :chart-times="realtimeChartTimes"
      :chart-series="realtimeChartSeries"
    />

    <MonitoringInsightsPanel
      :alerts="alerts"
      :sample-summary-items="sampleSummaryItems"
    />

    <MonitoringSessionsPanel
      :sessions="sessions"
      @refresh="loadSessions"
      @open-detail="openSessionDetail"
      @continue-diagnosis="continueDiagnosis"
    />

    <MonitoringSessionDialog
      v-model:visible="detailVisible"
      :session-detail="sessionDetail"
    />
  </div>
</template>

<script setup>
import MonitoringControlPanel from '@/modules/monitoring/components/MonitoringControlPanel.vue'
import MonitoringInsightsPanel from '@/modules/monitoring/components/MonitoringInsightsPanel.vue'
import MonitoringMetrics from '@/modules/monitoring/components/MonitoringMetrics.vue'
import MonitoringRealtimeChartPanel from '@/modules/monitoring/components/MonitoringRealtimeChartPanel.vue'
import MonitoringSessionDialog from '@/modules/monitoring/components/MonitoringSessionDialog.vue'
import MonitoringSessionsPanel from '@/modules/monitoring/components/MonitoringSessionsPanel.vue'
import { useMonitoringDashboard } from '@/modules/monitoring/composables/useMonitoringDashboard'

const {
  actionLoading,
  alerts,
  backendOptions,
  blindTestEnabled,
  currentState,
  detailVisible,
  loadSessions,
  metrics,
  onlineModels,
  openSessionDetail,
  replayAssets,
  realtimeChartFields,
  realtimeChartSeries,
  realtimeChartTimes,
  realtimeFieldOptions,
  sampleSummaryItems,
  selectedBackend,
  selectedDuration,
  selectedModel,
  selectedReplayAssetId,
  sessionDetail,
  sessions,
  startBlindTest,
  startCollection,
  startDetection,
  stopCollection,
  stopDetection,
  continueDiagnosis
} = useMonitoringDashboard()
</script>
