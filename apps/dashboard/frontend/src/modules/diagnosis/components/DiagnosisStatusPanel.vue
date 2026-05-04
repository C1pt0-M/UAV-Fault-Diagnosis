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

      <div class="diagnosis-status__steps">
        <div class="diagnosis-step" :class="{ 'is-active': stage.key === 'idle' }">
          <span>01</span>
          <strong>任务排布</strong>
        </div>
        <div class="diagnosis-step" :class="{ 'is-active': stage.key === 'running' }">
          <span>02</span>
          <strong>推理执行</strong>
        </div>
        <div class="diagnosis-step" :class="{ 'is-active': stage.key === 'completed' }">
          <span>03</span>
          <strong>结果出舱</strong>
        </div>
      </div>

      <template v-if="currentRun">
        <el-progress :percentage="progressPercentage" />
        <div class="run-meta">
          <span>{{ diagnosisCopy.taskIdLabel }}：#{{ currentRun.id }}</span>
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
  padding: 16px;
  border-radius: var(--radius-lg);
  background: var(--bg-panel-raised);
  border: 1px solid var(--line-soft);
  border-left: 3px solid var(--brand);
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
  font-size: 26px;
  color: var(--text-primary);
}

.diagnosis-status__hero p {
  margin: 10px 0 0;
  color: var(--text-secondary);
  line-height: 1.7;
}

.diagnosis-status__progress {
  min-width: 150px;
  padding: 14px;
  border-radius: var(--radius-md);
  background: var(--bg-panel);
  border: 1px solid var(--line-strong);
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

.diagnosis-status__steps {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 12px;
  margin-bottom: 18px;
}

.diagnosis-step {
  padding: 10px 14px;
  border-radius: var(--radius-md);
  border: 1px solid var(--line-soft);
  background: var(--bg-panel-raised);
}

.diagnosis-step.is-active {
  border-color: var(--brand);
  background: var(--brand-glow);
}

.diagnosis-step span {
  font-size: 12px;
  color: var(--text-tertiary);
}

.diagnosis-step strong {
  display: block;
  margin-top: 8px;
}

.diagnosis-status.is-completed .diagnosis-status__hero {
  border-left-color: var(--status-ok);
}

.diagnosis-status.is-running .diagnosis-status__hero {
  border-left-color: var(--status-active);
}

.run-meta {
  display: flex;
  gap: 18px;
  flex-wrap: wrap;
  margin-top: 12px;
  color: var(--text-secondary);
}
</style>
