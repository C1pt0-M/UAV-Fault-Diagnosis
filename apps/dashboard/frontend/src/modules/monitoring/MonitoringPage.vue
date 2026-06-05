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
        <span class="monitoring-statusbar__label">风险评分</span>
        <span class="monitoring-statusbar__value monitoring-statusbar__value--mono">{{ riskScore }}</span>
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

        <MonitoringInsightsPanel
          :alerts="alerts"
          :sample-summary-items="sampleSummaryItems"
        />
      </main>
    </div>

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

    <ElDialog
      v-model="faultHandoffDialogVisible"
      align-center
      class="fault-handoff-dialog"
      width="440px"
      :show-close="false"
    >
      <template #header>
        <div class="fault-handoff__header">
          <span class="fault-handoff__icon">!</span>
          <div>
            <strong>检测到故障事件</strong>
            <p>
              当前会话已确认 {{ faultHandoffPayload.fault_events_count || 0 }} 个故障事件，
              是否承接到诊断中心继续分析？
            </p>
          </div>
        </div>
      </template>

      <div class="fault-handoff__body">
        <div class="fault-handoff__row">
          <span>当前会话 ID</span>
          <strong>{{ faultHandoffPayload.session_id || '暂无' }}</strong>
        </div>
        <div class="fault-handoff__row">
          <span>故障事件数</span>
          <strong>{{ faultHandoffPayload.fault_events_count || 0 }}</strong>
        </div>
        <div class="fault-handoff__row">
          <span>建议动作</span>
          <strong>进入诊断中心生成离线诊断任务</strong>
        </div>
      </div>

      <template #footer>
        <div class="fault-handoff__footer">
          <ElButton @click="dismissFaultHandoff">稍后处理</ElButton>
          <ElButton type="primary" @click="confirmFaultHandoff">进入诊断中心</ElButton>
        </div>
      </template>
    </ElDialog>
  </div>
</template>

<script setup>
import { computed } from 'vue'
import { ElButton, ElDialog } from 'element-plus'
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
  dismissFaultHandoff,
  faultHandoffDialogVisible,
  faultHandoffPayload,
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
  riskScore,
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
  confirmFaultHandoff,
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
  min-width: 0;
}

.monitoring-workbench__main {
  min-width: 0;
  display: flex;
  flex-direction: column;
  gap: var(--space-lg);
}

.fault-handoff__header {
  display: flex;
  align-items: flex-start;
  gap: 12px;
}

.fault-handoff__header strong {
  display: block;
  color: var(--text-primary);
  font-size: 16px;
  line-height: 1.4;
  font-weight: 800;
}

.fault-handoff__header p {
  margin: 6px 0 0;
  color: var(--text-secondary);
  font-size: 13px;
  line-height: 1.7;
}

.fault-handoff__icon {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 28px;
  height: 28px;
  flex: 0 0 28px;
  border: 1px solid rgba(245, 158, 11, 0.34);
  border-radius: var(--radius-sm);
  background: rgba(245, 158, 11, 0.12);
  color: var(--warning);
  font-weight: 900;
}

.fault-handoff__body {
  display: grid;
  gap: 10px;
  padding: 4px 0 2px;
}

.fault-handoff__row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  padding: 10px 12px;
  border: 1px solid var(--line-soft);
  border-radius: var(--radius-sm);
  background: var(--bg-panel-muted);
}

.fault-handoff__row span {
  color: var(--text-tertiary);
  font-size: 12px;
}

.fault-handoff__row strong {
  min-width: 0;
  color: var(--text-primary);
  font-size: 12px;
  font-weight: 800;
  text-align: right;
}

.fault-handoff__footer {
  display: flex;
  justify-content: flex-end;
  gap: 10px;
}

:deep(.fault-handoff-dialog .el-dialog) {
  border: 1px solid var(--line-soft);
  border-radius: var(--radius-md);
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
