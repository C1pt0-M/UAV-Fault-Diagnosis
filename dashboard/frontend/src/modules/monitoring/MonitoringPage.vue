<template>
  <div class="page-stack monitoring-command">
    <section class="monitoring-hero">
      <div class="monitoring-hero__main">
        <span class="monitoring-hero__eyebrow">在线监测</span>
        <h1>实时监测与在线预警工作台</h1>
        <p>{{ monitoringCopy.heroDescription }}</p>
      </div>

      <div class="monitoring-hero__summary">
        <article class="monitoring-summary-card monitoring-summary-card--accent">
          <span>{{ monitoringCopy.heroStatusLabel }}</span>
          <strong>{{ stateLabel }}</strong>
          <small>当前会话：{{ currentSessionId }}</small>
        </article>
        <article class="monitoring-summary-card">
          <span>{{ monitoringCopy.heroBackendLabel }}</span>
          <strong>{{ selectedBackendLabel }}</strong>
          <small>{{ sessionBridgeText }}</small>
        </article>
        <article class="monitoring-summary-card">
          <span>{{ monitoringCopy.heroModelLabel }}</span>
          <strong>{{ activeModelLabel }}</strong>
          <small>故障概率峰值：{{ faultPeakText }}</small>
        </article>
      </div>
    </section>

    <MonitoringControlPanel
      v-model:selected-backend="selectedBackend"
      v-model:selected-duration="selectedDuration"
      v-model:selected-model="selectedModel"
      v-model:selected-replay-asset-id="selectedReplayAssetId"
      v-model:rfly-px4mavctrl-path="rflyPx4mavctrlPath"
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
      :chart-times="realtimeChartTimes"
      :fault-prob-data="realtimeFaultProb"
      :motor-series="realtimeMotorSeries"
      :attitude-series="realtimeAttitudeSeries"
      :position-series="realtimePositionSeries"
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
import { monitoringCopy } from '@/modules/monitoring/monitoringView'
import { useMonitoringDashboard } from '@/modules/monitoring/composables/useMonitoringDashboard'

const {
  actionLoading,
  activeModelLabel,
  alerts,
  backendOptions,
  blindTestEnabled,
  continueDiagnosis,
  currentSessionId,
  currentState,
  detailVisible,
  faultPeakText,
  loadSessions,
  metrics,
  onlineModels,
  openSessionDetail,
  replayAssets,
  realtimeChartTimes,
  realtimeFaultProb,
  realtimeAttitudeSeries,
  realtimeMotorSeries,
  realtimePositionSeries,
  rflyPx4mavctrlPath,
  sampleSummaryItems,
  selectedBackend,
  selectedBackendLabel,
  selectedDuration,
  selectedModel,
  selectedReplayAssetId,
  sessionBridgeText,
  sessionDetail,
  sessions,
  startBlindTest,
  startCollection,
  startDetection,
  stateLabel,
  stopCollection,
  stopDetection
} = useMonitoringDashboard()
</script>

<style scoped>
.monitoring-hero {
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

.monitoring-hero__eyebrow {
  display: inline-flex;
  padding: 7px 12px;
  border-radius: 999px;
  background: rgba(255, 255, 255, 0.14);
  font-size: 12px;
  font-weight: 800;
  letter-spacing: 0.04em;
}

.monitoring-hero__main h1 {
  margin: 18px 0 0;
  font-size: 44px;
  line-height: 1.04;
}

.monitoring-hero__main p {
  margin: 16px 0 0;
  max-width: 660px;
  line-height: 1.9;
  color: rgba(244, 251, 251, 0.86);
}

.monitoring-hero__summary {
  display: grid;
  gap: 12px;
}

.monitoring-summary-card {
  padding: 16px 18px;
  border-radius: 20px;
  background: rgba(255, 255, 255, 0.1);
  border: 1px solid rgba(255, 255, 255, 0.14);
}

.monitoring-summary-card--accent {
  background: rgba(255, 255, 255, 0.14);
}

.monitoring-summary-card span {
  font-size: 12px;
  color: rgba(244, 251, 251, 0.72);
}

.monitoring-summary-card strong {
  display: block;
  margin-top: 8px;
  font-size: 22px;
  line-height: 1.3;
}

.monitoring-summary-card small {
  display: block;
  margin-top: 6px;
  line-height: 1.7;
  color: rgba(244, 251, 251, 0.8);
}

@media (max-width: 1480px) {
  .monitoring-hero {
    grid-template-columns: 1fr;
  }
}
</style>
