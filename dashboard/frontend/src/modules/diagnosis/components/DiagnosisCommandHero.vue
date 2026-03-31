<template>
  <section class="diagnosis-hero">
    <div class="diagnosis-hero__main">
      <span class="diagnosis-hero__eyebrow">{{ diagnosisCopy.commandTitle }}</span>
      <h1>{{ stage.label }}</h1>
      <p>{{ diagnosisCopy.commandDescription }}</p>
      <div class="diagnosis-hero__cue">{{ stage.cue }}</div>
    </div>

    <div class="diagnosis-hero__summary">
      <div class="diagnosis-summary-card">
        <span>{{ diagnosisCopy.stageLabel }}</span>
        <strong>{{ stage.label }}</strong>
        <small>{{ stage.description }}</small>
      </div>
      <div class="diagnosis-summary-card">
        <span>{{ diagnosisCopy.assetSummaryLabel }}</span>
        <strong>{{ selectedAssetName }}</strong>
        <small>{{ diagnosisCopy.linkedSessionLabel }}：{{ activeSessionId || '-' }}</small>
      </div>
      <div class="diagnosis-summary-card">
        <span>{{ diagnosisCopy.modelSummaryLabel }}</span>
        <strong>{{ selectedModelName }}</strong>
        <small>{{ currentRun?.task || 'offline_multiclass' }}</small>
      </div>
    </div>
  </section>
</template>

<script setup>
import { diagnosisCopy } from '@/modules/diagnosis/diagnosisView'

defineProps({
  activeSessionId: {
    type: String,
    default: ''
  },
  currentRun: {
    type: Object,
    default: null
  },
  selectedAssetName: {
    type: String,
    required: true
  },
  selectedModelName: {
    type: String,
    required: true
  },
  stage: {
    type: Object,
    required: true
  }
})
</script>

<style scoped>
.diagnosis-hero {
  display: grid;
  grid-template-columns: minmax(0, 1.25fr) minmax(320px, 0.95fr);
  gap: 18px;
  padding: 24px;
  border-radius: 28px;
  background:
    radial-gradient(circle at top right, rgba(255, 255, 255, 0.14), transparent 30%),
    linear-gradient(135deg, #081f2f 0%, #0f4b57 52%, #0f766e 100%);
  color: #f4fbfb;
  box-shadow: 0 24px 60px rgba(8, 31, 47, 0.24);
}

.diagnosis-hero__eyebrow {
  display: inline-flex;
  padding: 6px 10px;
  border-radius: 999px;
  background: rgba(255, 255, 255, 0.14);
  font-size: 12px;
  font-weight: 700;
  letter-spacing: 0.04em;
}

.diagnosis-hero__main h1 {
  margin: 16px 0 0;
  font-size: 42px;
  line-height: 1.05;
}

.diagnosis-hero__main p {
  margin: 14px 0 0;
  max-width: 640px;
  line-height: 1.9;
  color: rgba(244, 251, 251, 0.84);
}

.diagnosis-hero__cue {
  margin-top: 18px;
  display: inline-flex;
  padding: 10px 14px;
  border-radius: 16px;
  background: rgba(255, 255, 255, 0.12);
  border: 1px solid rgba(255, 255, 255, 0.16);
  font-weight: 700;
}

.diagnosis-hero__summary {
  display: grid;
  gap: 12px;
}

.diagnosis-summary-card {
  padding: 16px 18px;
  border-radius: 18px;
  background: rgba(255, 255, 255, 0.1);
  border: 1px solid rgba(255, 255, 255, 0.14);
}

.diagnosis-summary-card span {
  font-size: 12px;
  color: rgba(244, 251, 251, 0.74);
}

.diagnosis-summary-card strong {
  display: block;
  margin-top: 8px;
  font-size: 22px;
  line-height: 1.3;
}

.diagnosis-summary-card small {
  display: block;
  margin-top: 6px;
  color: rgba(244, 251, 251, 0.78);
  line-height: 1.6;
}
</style>
