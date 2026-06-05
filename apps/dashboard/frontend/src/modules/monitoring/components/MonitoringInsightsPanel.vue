<template>
  <div class="monitoring-insights">
    <section class="section-card">
      <div class="section-card__header">
        <div>
          <div class="section-card__title">{{ monitoringCopy.statusOverviewTitle }}</div>
          <div class="section-card__desc">{{ monitoringCopy.statusOverviewDescription }}</div>
        </div>
      </div>
      <div class="section-card__body monitoring-insights__body">
        <div class="status-overview-list">
          <div
            v-for="item in statusOverviewItems"
            :key="item.label"
            class="status-overview-row"
          >
            <span class="status-overview-row__label">{{ item.label }}</span>
            <span class="status-badge" :class="`status-badge--${item.status || 'offline'}`">
              <span class="status-dot" :class="`status-dot--${item.status || 'offline'}`"></span>
              {{ item.value }}
            </span>
          </div>
        </div>
      </div>
    </section>

    <section class="section-card">
      <div class="section-card__header">
        <div>
          <div class="section-card__title">{{ monitoringCopy.pipelineTitle }}</div>
          <div class="section-card__desc">{{ monitoringCopy.pipelineDescription }}</div>
        </div>
      </div>
      <div class="section-card__body monitoring-insights__body">
        <div class="pipeline-list">
          <div
            v-for="item in pipelineItems"
            :key="item.label"
            class="pipeline-row"
          >
            <span class="pipeline-row__rail" :class="`pipeline-row__rail--${item.status || 'offline'}`"></span>
            <div>
              <strong>{{ item.label }}</strong>
              <small :title="item.value">{{ item.value }}</small>
            </div>
          </div>
        </div>
      </div>
    </section>

    <section class="section-card">
      <div class="section-card__header">
        <div>
          <div class="section-card__title">{{ monitoringCopy.alertsTitle }}</div>
          <div class="section-card__desc">{{ monitoringCopy.alertsDescription }}</div>
        </div>
      </div>
      <div class="section-card__body monitoring-insights__body">
        <div
          v-if="alerts.length === 0"
          class="monitoring-alert-empty"
        >
          <el-empty
            :description="monitoringCopy.alertsEmpty"
            :image-size="44"
          />
        </div>
        <div v-else class="alert-table">
          <div class="alert-table__head">
            <span>时间</span>
            <span>事件 / 信号</span>
            <span>级别</span>
          </div>
          <div
            v-for="item in alerts.slice(0, 8)"
            :key="item.id || `${item.start_time}-${item.end_time}`"
            class="alert-table__row"
          >
            <span class="alert-table__time">{{ item.start_time || '-' }}</span>
            <strong :title="item.diagnosis?.label || item.label || monitoringCopy.unknownFault">
              {{ item.diagnosis?.label || item.label || monitoringCopy.unknownFault }}
            </strong>
            <span class="status-badge" :class="`status-badge--${item.tone || resolveAlertTone(item)}`">
              {{ severityLabel(item) }}
            </span>
          </div>
        </div>
      </div>
    </section>

    <section class="section-card">
      <div class="section-card__header">
        <div>
          <div class="section-card__title">{{ monitoringCopy.sampleTitle }}</div>
          <div class="section-card__desc">{{ monitoringCopy.sampleDescription }}</div>
        </div>
      </div>
      <div class="section-card__body monitoring-insights__body">
        <dl class="sample-list">
          <div
            v-for="item in sampleSummaryItems"
            :key="item.label"
            class="sample-list__row"
          >
            <dt>{{ item.label }}</dt>
            <dd>{{ item.value }}</dd>
          </div>
        </dl>
      </div>
    </section>
  </div>
</template>

<script setup>
import ElEmpty from 'element-plus/es/components/empty/index.mjs'
import { monitoringCopy } from '@/modules/monitoring/monitoringView'

defineProps({
  alerts: { type: Array, default: () => [] },
  sampleSummaryItems: { type: Array, default: () => [] },
  statusOverviewItems: { type: Array, default: () => [] },
  pipelineItems: { type: Array, default: () => [] }
})

function resolveAlertTone(item) {
  const value = String(item.diagnosis?.severity || item.severity || '').toLowerCase()
  if (['critical', 'high', 'danger'].includes(value)) return 'danger'
  if (['medium', 'warning', 'warn'].includes(value)) return 'warn'
  if (['low', 'info'].includes(value)) return 'ok'
  return 'pending'
}

function severityLabel(item) {
  const value = String(item.diagnosis?.severity || item.severity || '').toLowerCase()
  if (['critical', 'high', 'danger'].includes(value)) return '高危'
  if (['medium', 'warning', 'warn'].includes(value)) return '中危'
  if (['low', 'info'].includes(value)) return '低危'
  return monitoringCopy.pendingSeverity
}
</script>

<style scoped>
.monitoring-insights {
  display: flex;
  flex-direction: column;
  gap: var(--space-lg);
}

.monitoring-insights__body {
  padding: var(--space-lg);
}

.monitoring-alert-empty {
  min-height: 132px;
  display: grid;
  place-items: center;
}

.monitoring-alert-empty :deep(.el-empty) {
  --el-empty-padding: 10px 0;
}

.status-overview-list,
.pipeline-list,
.sample-list {
  margin: 0;
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.status-overview-row,
.sample-list__row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: var(--space-md);
  min-width: 0;
}

.status-overview-row__label,
.sample-list__row dt {
  color: var(--text-secondary);
  font-size: 12px;
  font-weight: 700;
}

.sample-list__row dd {
  margin: 0;
  color: var(--text-primary);
  font-family: var(--font-mono);
  font-size: 12px;
  font-weight: 800;
}

.pipeline-row {
  display: grid;
  grid-template-columns: 4px minmax(0, 1fr);
  gap: 10px;
  min-width: 0;
}

.pipeline-row__rail {
  border-radius: 999px;
  background: var(--status-offline);
}

.pipeline-row__rail--ok {
  background: var(--status-ok);
}

.pipeline-row__rail--active {
  background: var(--status-active);
}

.pipeline-row__rail--warn {
  background: var(--status-warn);
}

.pipeline-row__rail--danger {
  background: var(--status-danger);
}

.pipeline-row__rail--pending {
  background: var(--status-pending);
}

.pipeline-row strong {
  display: block;
  font-size: 12px;
  color: var(--text-primary);
}

.pipeline-row small {
  display: block;
  margin-top: 3px;
  max-width: 100%;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  color: var(--text-tertiary);
  font-size: 11px;
}

.alert-table {
  display: flex;
  flex-direction: column;
  gap: 2px;
}

.alert-table__head,
.alert-table__row {
  display: grid;
  grid-template-columns: 72px minmax(0, 1fr) 58px;
  align-items: center;
  gap: 8px;
}

.alert-table__head {
  padding: 0 0 8px;
  color: var(--text-tertiary);
  font-size: 11px;
  font-weight: 700;
  border-bottom: 1px solid var(--line-soft);
}

.alert-table__row {
  min-height: 34px;
  border-bottom: 1px solid var(--line-soft);
  font-size: 12px;
}

.alert-table__row strong {
  min-width: 0;
  color: var(--text-primary);
  font-weight: 700;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.alert-table__time {
  font-family: var(--font-mono);
  color: var(--text-secondary);
}
</style>
