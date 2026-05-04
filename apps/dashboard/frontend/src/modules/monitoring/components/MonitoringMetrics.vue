<template>
  <section class="metric-grid monitoring-metrics">
    <article
      v-for="item in metrics"
      :key="item.label"
      class="metric-card monitoring-metric"
    >
      <span class="metric-card__label">{{ item.label }}</span>
      <div class="metric-card__value">{{ item.value }}</div>
      <div class="metric-card__meta">{{ item.meta }}</div>
    </article>
  </section>
</template>

<script setup>
defineProps({
  metrics: {
    type: Array,
    default: () => []
  }
})
</script>

<style scoped>
.monitoring-metric {
  position: relative;
  overflow: hidden;
  min-height: 156px;
}

.monitoring-metric::before {
  content: '';
  position: absolute;
  inset: 0 auto auto 0;
  width: 100%;
  height: 4px;
  background: var(--brand);
}

.monitoring-metric:nth-child(2)::before {
  background: var(--status-active);
}

.monitoring-metric:nth-child(3)::before {
  background: var(--status-warn);
}

.monitoring-metric:nth-child(4)::before {
  background: var(--status-danger);
}
</style>
