<template>
  <article class="section-card diagnosis-status" :class="`is-${stage.key}`">
    <div class="section-card__header">
      <div>
        <div class="section-card__title">{{ diagnosisCopy.statusTitle }}</div>
        <div class="section-card__desc">{{ diagnosisCopy.statusDescription }}</div>
      </div>
    </div>
    <div class="section-card__body">
      <div class="diagnosis-status__hero">
        <div>
          <span class="diagnosis-status__eyebrow">{{ diagnosisCopy.stageLabel }}</span>
          <h3>{{ stage.label }}</h3>
          <p>{{ stage.description }}</p>
        </div>
        <div class="diagnosis-status__progress">
          <strong>{{ currentRun ? `${progressPercentage}%` : '00%' }}</strong>
          <span>任务完成度</span>
        </div>
      </div>

      <template v-if="currentRun">
        <el-progress :percentage="progressPercentage" />
        <div class="run-meta">
          <span>{{ diagnosisCopy.taskIdLabel }}：{{ currentRun.id }}</span>
          <span>{{ diagnosisCopy.statusLabel }}：{{ currentRun.status }}</span>
        </div>
        <div class="run-meta">
          <span>{{ diagnosisCopy.taskTypeLabel }}：{{ currentRun.task || 'offline_multiclass' }}</span>
          <span>{{ diagnosisCopy.createdAtLabel }}：{{ currentRun.created_at || '-' }}</span>
        </div>
      </template>

      <el-empty v-else :description="diagnosisCopy.emptyRun" :image-size="60" />
    </div>
  </article>
</template>

<script setup>
import ElEmpty from 'element-plus/es/components/empty/index.mjs'
import ElProgress from 'element-plus/es/components/progress/index.mjs'
import { diagnosisCopy } from '@/modules/diagnosis/diagnosisView'

defineProps({
  currentRun: {
    type: Object,
    default: null
  },
  progressPercentage: {
    type: Number,
    default: 0
  },
  stage: {
    type: Object,
    required: true
  }
})
</script>

<style scoped>
.diagnosis-status {
  overflow: hidden;
}

.diagnosis-status__hero {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
  padding: 18px;
  border-radius: 18px;
  background: linear-gradient(180deg, #f8fbfc 0%, #eef6f7 100%);
  border: 1px solid var(--line-soft);
  margin-bottom: 18px;
}

.diagnosis-status__eyebrow {
  display: inline-flex;
  padding: 5px 9px;
  border-radius: 999px;
  background: var(--brand-soft);
  color: var(--brand-strong);
  font-size: 12px;
  font-weight: 700;
}

.diagnosis-status__hero h3 {
  margin: 12px 0 0;
  font-size: 30px;
}

.diagnosis-status__hero p {
  margin: 10px 0 0;
  color: var(--text-secondary);
  line-height: 1.7;
}

.diagnosis-status__progress {
  min-width: 150px;
  padding: 16px;
  border-radius: 18px;
  background: #ffffff;
  border: 1px solid var(--line-soft);
  text-align: center;
}

.diagnosis-status__progress strong {
  display: block;
  font-size: 34px;
  line-height: 1;
}

.diagnosis-status__progress span {
  display: block;
  margin-top: 8px;
  color: var(--text-tertiary);
  font-size: 12px;
}

.diagnosis-status.is-completed .diagnosis-status__hero {
  background: linear-gradient(180deg, #effaf6 0%, #e0f3eb 100%);
}

.diagnosis-status.is-running .diagnosis-status__hero {
  background: linear-gradient(180deg, #eef6fb 0%, #dfeef7 100%);
}

.run-meta {
  display: flex;
  gap: 18px;
  flex-wrap: wrap;
  margin-top: 12px;
  color: var(--text-secondary);
}
</style>
