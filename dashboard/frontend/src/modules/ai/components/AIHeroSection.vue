<template>
  <section class="section-card ai-hero">
    <div class="ai-hero__main">
      <div class="ai-hero__content">
        <span class="ai-hero__tag">{{ aiCopy.heroTag }}</span>
        <h2>{{ aiCopy.heroTitle }}</h2>
        <p class="page-description">{{ aiCopy.heroDescription }}</p>
      </div>

      <div class="ai-hero__cards">
        <article class="ai-hero__card ai-hero__card--accent">
          <span>{{ aiCopy.heroPrimaryCardTitle }}</span>
          <strong>报告解释 / 风险追问</strong>
          <small>{{ aiCopy.heroPrimaryCardText }}</small>
        </article>
        <article class="ai-hero__card">
          <span>{{ aiCopy.heroSecondaryCardTitle }}</span>
          <strong>检索 / Embedding / 回退</strong>
          <small>{{ aiCopy.heroSecondaryCardText }}</small>
        </article>
      </div>
    </div>
  </section>
</template>

<script setup>
import { aiCopy } from '@/modules/ai/aiView'
</script>

<style scoped>
.ai-hero {
  overflow: hidden;
  background:
    radial-gradient(circle at right top, rgba(255, 255, 255, 0.18), transparent 30%),
    linear-gradient(140deg, #081f2f 0%, #0d4656 56%, #0f766e 100%);
  color: #f4fbfb;
}

.ai-hero__main {
  display: grid;
  grid-template-columns: minmax(0, 1.25fr) minmax(320px, 0.95fr);
  gap: 18px;
  align-items: stretch;
  padding: 26px 28px 24px;
}

.ai-hero__content {
  padding: 10px 0 0 8px;
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
  font-size: 34px;
  line-height: 1.1;
}

.ai-hero :deep(.page-description) {
  color: rgba(244, 251, 251, 0.84);
}

.ai-hero__cards {
  display: grid;
  gap: 12px;
}

.ai-hero__card {
  padding: 16px 18px;
  border-radius: 20px;
  background: rgba(255, 255, 255, 0.1);
  border: 1px solid rgba(255, 255, 255, 0.14);
}

.ai-hero__card--accent {
  background: rgba(255, 255, 255, 0.14);
}

.ai-hero__card span {
  font-size: 12px;
  color: rgba(244, 251, 251, 0.72);
}

.ai-hero__card strong {
  display: block;
  margin-top: 8px;
  font-size: 22px;
  line-height: 1.3;
}

.ai-hero__card small {
  display: block;
  margin-top: 6px;
  line-height: 1.7;
  color: rgba(244, 251, 251, 0.8);
}

@media (max-width: 1480px) {
  .ai-hero__main {
    grid-template-columns: 1fr;
    padding: 24px 24px 22px;
  }

  .ai-hero__content {
    padding-left: 4px;
  }
}
</style>
