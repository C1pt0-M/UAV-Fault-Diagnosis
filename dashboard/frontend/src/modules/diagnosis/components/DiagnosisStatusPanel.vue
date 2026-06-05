<template>
  <article class="section-card">
    <div class="section-card__header">
      <div>
        <div class="section-card__title">{{ diagnosisCopy.statusTitle }}</div>
        <div class="section-card__desc">{{ diagnosisCopy.statusDescription }}</div>
      </div>
    </div>
    <div class="section-card__body">
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

        <el-button
          v-if="resultReady && !reportCreating"
          type="success"
          plain
          @click="$emit('create-report')"
        >
          {{ diagnosisCopy.reportButton }}
        </el-button>
        <el-button v-else-if="reportCreating" type="success" plain loading>
          {{ diagnosisCopy.reportLoadingButton }}
        </el-button>
      </template>

      <el-empty v-else :description="diagnosisCopy.emptyRun" :image-size="60" />
    </div>
  </article>
</template>

<script setup>
import ElButton from 'element-plus/es/components/button/index.mjs'
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
  reportCreating: {
    type: Boolean,
    default: false
  },
  resultReady: {
    type: Boolean,
    default: false
  }
})

defineEmits(['create-report'])
</script>

<style scoped>
.run-meta {
  display: flex;
  gap: 18px;
  flex-wrap: wrap;
  margin-top: 12px;
  color: var(--text-secondary);
}
</style>
