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
  background: var(--bg-panel);
  border-left: 3px solid var(--brand);
}

.ai-hero__main {
  display: grid;
  grid-template-columns: minmax(0, 1.25fr) minmax(320px, 0.95fr);
  gap: var(--space-xl);
  align-items: stretch;
  padding: var(--space-xl);
}

.ai-hero__content {
  padding: var(--space-sm) 0 0 var(--space-sm);
}

.ai-hero__tag {
  display: inline-flex;
  padding: 5px 10px;
  border-radius: var(--radius-sm);
  background: var(--brand-soft);
  color: var(--brand-strong);
  font-size: 12px;
  font-weight: 700;
  letter-spacing: 0.04em;
}

.ai-hero h2 {
  margin: 14px 0 0;
  font-size: 24px;
  line-height: 1.2;
  color: var(--text-primary);
}

.ai-hero :deep(.page-description) {
  color: var(--text-secondary);
}

.ai-hero__cards {
  display: grid;
  gap: var(--space-md);
}

.ai-hero__card {
  padding: var(--space-lg);
  border-radius: var(--radius-md);
  background: var(--bg-panel-raised);
  border: 1px solid var(--line-soft);
}

.ai-hero__card--accent {
  border-left: 3px solid var(--brand);
  padding-left: calc(var(--space-lg) - 3px);
}

.ai-hero__card span {
  font-size: 12px;
  color: var(--text-tertiary);
  text-transform: uppercase;
  letter-spacing: 0.08em;
}

.ai-hero__card strong {
  display: block;
  margin-top: 8px;
  font-size: 17px;
  line-height: 1.3;
  color: var(--text-primary);
}

.ai-hero__card small {
  display: block;
  margin-top: 6px;
  line-height: 1.6;
  color: var(--text-secondary);
}

@media (max-width: 1480px) {
  .ai-hero__main {
    grid-template-columns: 1fr;
  }

  .ai-hero__content {
    padding-left: 4px;
  }
}
</style>
