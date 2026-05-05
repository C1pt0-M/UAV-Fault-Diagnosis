<template>
  <div class="page-stack monitoring-cockpit">
    <header class="monitoring-cockpit__heading">
      <div>
        <h1>{{ monitoringCopy.pageTitle }}</h1>
        <p>{{ monitoringCopy.pageDescription }}</p>
      </div>
      <div class="monitoring-cockpit__sync">
        <span class="status-dot" :class="statusDotClass"></span>
        <span>{{ stateLabel }}</span>
      </div>
    </header>

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
      <div class="monitoring-statusbar__item monitoring-statusbar__item--model">
        <span class="monitoring-statusbar__label">{{ monitoringCopy.heroModelLabel }}</span>
        <span
          class="monitoring-statusbar__value monitoring-statusbar__value--truncate"
          data-test="monitoring-status-model"
          :title="activeModelLabel"
        >
          {{ activeModelLabel }}
        </span>
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

    <div class="monitoring-workbench">
      <main class="monitoring-workbench__main">
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

        <MonitoringSessionsPanel
          :sessions="sessions"
          @refresh="loadSessions"
          @open-detail="openSessionDetail"
          @continue-diagnosis="continueDiagnosis"
        />
      </main>

      <aside class="monitoring-workbench__side">
        <MonitoringInsightsPanel
          :alerts="alerts"
          :sample-summary-items="sampleSummaryItems"
          :status-overview-items="statusOverviewItems"
          :pipeline-items="pipelineItems"
        />
      </aside>
    </div>

    <MonitoringSessionDialog
      v-model:visible="detailVisible"
      :session-detail="sessionDetail"
    />
  </div>
</template>

<script setup>
import { computed } from 'vue'
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
  pipelineItems,
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
  sessionDetail,
  sessions,
  startBlindTest,
  startCollection,
  startDetection,
  stateLabel,
  statusOverviewItems,
  stopCollection,
  stopDetection,
  testRflyDiagnostics
} = useMonitoringDashboard()

const statusDotClass = computed(() => {
  if (currentState.value === 'DETECTING') return 'status-dot--ok'
  if (currentState.value === 'COLLECTING') return 'status-dot--active'
  if (currentState.value === 'BLIND_TEST') return 'status-dot--pending'
  return 'status-dot--offline'
})
</script>

<style scoped>
.monitoring-cockpit {
  gap: var(--space-lg);
}

.monitoring-cockpit__heading {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: var(--space-lg);
}

.monitoring-cockpit__heading h1 {
  margin: 0;
  font-size: 24px;
  line-height: 1.25;
  font-weight: 800;
  color: var(--text-primary);
  letter-spacing: 0;
}

.monitoring-cockpit__heading p {
  margin: 6px 0 0;
  color: var(--text-tertiary);
  font-size: 13px;
}

.monitoring-cockpit__sync {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  padding: 8px 12px;
  border: 1px solid var(--line-soft);
  border-radius: var(--radius-md);
  background: var(--bg-panel-raised);
  color: var(--text-secondary);
  font-size: 12px;
  font-weight: 700;
  white-space: nowrap;
}

.monitoring-statusbar {
  display: flex;
  align-items: center;
  flex-wrap: nowrap;
  gap: var(--space-sm);
  min-width: 0;
  padding: 10px 14px;
  border: 1px solid var(--line-soft);
  border-radius: var(--radius-md);
  background: var(--bg-panel-raised);
  font-size: 12px;
  color: var(--text-secondary);
  overflow: hidden;
}

.monitoring-statusbar__item {
  display: flex;
  align-items: center;
  gap: 6px;
  min-width: 0;
  flex-shrink: 0;
}

.monitoring-statusbar__item--model {
  flex: 1 1 260px;
}

.monitoring-statusbar__label {
  color: var(--text-tertiary);
  font-size: 11px;
  white-space: nowrap;
}

.monitoring-statusbar__value {
  min-width: 0;
  color: var(--text-primary);
  font-weight: 700;
  white-space: nowrap;
}

.monitoring-statusbar__value--mono {
  font-family: var(--font-mono);
  font-size: 11px;
}

.monitoring-statusbar__value--truncate {
  display: inline-block;
  max-width: min(360px, 100%);
  overflow: hidden;
  text-overflow: ellipsis;
  vertical-align: bottom;
}

.monitoring-statusbar__divider {
  width: 1px;
  height: 14px;
  background: var(--line-strong);
  flex-shrink: 0;
}

.monitoring-workbench {
  display: grid;
  grid-template-columns: minmax(0, 8fr) minmax(340px, 4fr);
  gap: var(--space-lg);
  align-items: start;
}

.monitoring-workbench__main,
.monitoring-workbench__side {
  min-width: 0;
  display: flex;
  flex-direction: column;
  gap: var(--space-lg);
}

@media (max-width: 1480px) {
  .monitoring-workbench {
    grid-template-columns: 1fr;
  }
}

@media (max-width: 960px) {
  .monitoring-cockpit__heading {
    align-items: flex-start;
    flex-direction: column;
  }

  .monitoring-statusbar {
    flex-wrap: wrap;
  }

  .monitoring-statusbar__item--model {
    flex-basis: 100%;
  }

  .monitoring-statusbar__divider {
    display: none;
  }
}
</style>
