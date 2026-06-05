<template>
  <section class="section-card ai-hero">
    <div class="section-card__body">
      <span class="ai-hero__tag">{{ aiCopy.heroTag }}</span>
      <h2>{{ aiCopy.heroTitle }}</h2>
      <p class="page-description">{{ aiCopy.heroDescription }}</p>
    </div>
  </section>
</template>

<script setup>
import { aiCopy } from '@/modules/ai/aiView'
</script>

<style scoped>
.ai-hero {
  background: linear-gradient(135deg, #103240 0%, #0f766e 100%);
  color: #f4fbfb;
}

.ai-hero__tag {
  display: inline-flex;
  padding: 6px 10px;
  border-radius: 999px;
  background: rgba(255, 255, 255, 0.14);
  font-size: 12px;
  font-weight: 700;
}

.ai-hero h2 {
  margin: 14px 0 0;
  font-size: 32px;
}
</style>
