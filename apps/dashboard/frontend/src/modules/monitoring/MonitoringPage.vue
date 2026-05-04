<template>
  <div class="page-stack monitoring-command">
    <!-- 监测状态摘要条：替代原大渐变 Hero -->
    <div class="monitoring-statusbar">
      <div class="monitoring-statusbar__item">
        <span class="monitoring-statusbar__label">{{ monitoringCopy.heroStatusLabel }}</span>
        <span class="monitoring-statusbar__value">{{ stateLabel }}</span>
      </div>
      <div class="monitoring-statusbar__divider"></div>
      <div class="monitoring-statusbar__item">
        <span class="monitoring-statusbar__label">当前会话</span>
        <span class="monitoring-statusbar__value monitoring-statusbar__value--mono">{{ currentSessionId || '-' }}</span>
      </div>
      <div class="monitoring-statusbar__divider"></div>
      <div class="monitoring-statusbar__item">
        <span class="monitoring-statusbar__label">{{ monitoringCopy.heroBackendLabel }}</span>
        <span class="monitoring-statusbar__value">{{ selectedBackendLabel }}</span>
      </div>
      <div class="monitoring-statusbar__divider"></div>
      <div class="monitoring-statusbar__item">
        <span class="monitoring-statusbar__label">{{ monitoringCopy.heroModelLabel }}</span>
        <span class="monitoring-statusbar__value">{{ activeModelLabel }}</span>
      </div>
      <div class="monitoring-statusbar__divider"></div>
      <div class="monitoring-statusbar__item">
        <span class="monitoring-statusbar__label">故障概率峰值</span>
        <span class="monitoring-statusbar__value monitoring-statusbar__value--mono">{{ faultPeakText }}</span>
      </div>
      <div class="monitoring-statusbar__divider"></div>
      <div class="monitoring-statusbar__item" :title="inputContractMeta">
        <span class="monitoring-statusbar__label">字段合同</span>
        <span class="monitoring-statusbar__value">{{ inputContractLabel }}</span>
      </div>
    </div>

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
      :rfly-diagnostics="rflyDiagnostics"
      :rfly-diagnostics-loading="rflyDiagnosticsLoading"
      @start-collection="startCollection"
      @start-detection="startDetection"
      @start-blind-test="startBlindTest"
      @stop-detection="stopDetection"
      @stop-collection="stopCollection"
      @test-rfly-diagnostics="testRflyDiagnostics"
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
  inputContractLabel,
  inputContractMeta,
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
  rflyDiagnostics,
  rflyDiagnosticsLoading,
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
  stopDetection,
  testRflyDiagnostics
} = useMonitoringDashboard()
</script>

<style scoped>
/* 监测状态摘要条 */
.monitoring-statusbar {
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

.monitoring-statusbar__item {
  display: flex;
  align-items: center;
  gap: 6px;
}

.monitoring-statusbar__label {
  color: var(--text-tertiary);
  font-size: 11px;
  white-space: nowrap;
}

.monitoring-statusbar__value {
  color: var(--text-primary);
  font-weight: 600;
  white-space: nowrap;
}

.monitoring-statusbar__value--mono {
  font-family: var(--font-mono);
  font-size: 11px;
}

.monitoring-statusbar__divider {
  width: 1px;
  height: 14px;
  background: var(--line-strong);
  flex-shrink: 0;
}
</style>
