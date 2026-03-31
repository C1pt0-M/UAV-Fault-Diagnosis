<template>
  <section class="metric-grid">
    <article
      v-for="item in metrics"
      :key="item.label"
      class="metric-card"
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
