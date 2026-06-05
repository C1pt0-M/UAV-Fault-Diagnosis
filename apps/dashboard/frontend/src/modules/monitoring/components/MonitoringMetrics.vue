<template>
  <section class="monitoring-metrics">
    <article
      v-for="item in metrics"
      :key="item.label"
      class="monitoring-metric"
      :class="`monitoring-metric--${item.tone || 'active'}`"
    >
      <div class="monitoring-metric__icon">
        <el-icon>
          <component :is="iconMap[item.icon] || DataLine" />
        </el-icon>
      </div>
      <div class="monitoring-metric__content">
        <span class="monitoring-metric__label">{{ item.label }}</span>
        <div class="monitoring-metric__value" :title="item.value">{{ item.value }}</div>
        <div class="monitoring-metric__meta" :title="item.meta">{{ item.meta }}</div>
      </div>
    </article>
  </section>
</template>

<script setup>
import {
  Cpu,
  DataLine,
  DocumentChecked,
  Monitor,
  Timer,
  Warning
} from '@element-plus/icons-vue'
import ElIcon from 'element-plus/es/components/icon/index.mjs'

defineProps({
  metrics: {
    type: Array,
    default: () => []
  }
})

const iconMap = {
  Cpu,
  DataLine,
  DocumentChecked,
  Monitor,
  Timer,
  Warning
}
</script>

<style scoped>
.monitoring-metrics {
  display: grid;
  grid-template-columns: repeat(6, minmax(0, 1fr));
  gap: var(--space-md);
}

.monitoring-metric {
  min-width: 0;
  min-height: 102px;
  padding: 12px;
  border: 1px solid var(--line-soft);
  border-top: 3px solid var(--brand);
  border-radius: var(--radius-md);
  background: var(--bg-panel);
  box-shadow: var(--shadow-sm);
  display: flex;
  align-items: flex-start;
  gap: 10px;
  overflow: hidden;
}

.monitoring-metric--ok {
  border-top-color: var(--status-ok);
}

.monitoring-metric--warn {
  border-top-color: var(--status-warn);
}

.monitoring-metric--danger {
  border-top-color: var(--status-danger);
}

.monitoring-metric--offline {
  border-top-color: var(--status-offline);
}

.monitoring-metric--pending {
  border-top-color: var(--status-pending);
}

.monitoring-metric__icon {
  width: 30px;
  height: 30px;
  border-radius: var(--radius-sm);
  background: var(--brand-soft);
  color: var(--brand-strong);
  display: grid;
  place-items: center;
  flex-shrink: 0;
}

.monitoring-metric--ok .monitoring-metric__icon {
  background: var(--status-ok-soft);
  color: var(--status-ok);
}

.monitoring-metric--warn .monitoring-metric__icon {
  background: var(--status-warn-soft);
  color: var(--status-warn);
}

.monitoring-metric--danger .monitoring-metric__icon {
  background: var(--status-danger-soft);
  color: var(--status-danger);
}

.monitoring-metric--offline .monitoring-metric__icon {
  background: var(--status-offline-soft);
  color: var(--status-offline);
}

.monitoring-metric--pending .monitoring-metric__icon {
  background: var(--status-pending-soft);
  color: var(--status-pending);
}

.monitoring-metric__content {
  min-width: 0;
  flex: 1;
}

.monitoring-metric__label {
  display: block;
  font-size: 11px;
  color: var(--text-tertiary);
  font-weight: 700;
}

.monitoring-metric__value {
  margin-top: 6px;
  min-width: 0;
  font-size: 18px;
  line-height: 1.2;
  font-weight: 800;
  color: var(--text-primary);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.monitoring-metric__meta {
  margin-top: 5px;
  min-width: 0;
  font-size: 11px;
  line-height: 1.3;
  color: var(--text-tertiary);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

@media (max-width: 1680px) {
  .monitoring-metrics {
    grid-template-columns: repeat(3, minmax(0, 1fr));
  }
}

@media (max-width: 760px) {
  .monitoring-metrics {
    grid-template-columns: 1fr;
  }
}
</style>
