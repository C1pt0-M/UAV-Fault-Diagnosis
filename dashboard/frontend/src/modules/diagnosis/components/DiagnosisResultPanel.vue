<template>
  <section class="section-card">
    <div class="section-card__header">
      <div>
        <div class="section-card__title">{{ diagnosisCopy.resultTitle }}</div>
        <div class="section-card__desc">{{ diagnosisCopy.resultDescription }}</div>
      </div>
    </div>
    <div class="section-card__body">
      <div class="metric-grid diagnosis-result-grid">
        <article class="metric-card">
          <span class="metric-card__label">{{ diagnosisCopy.predictionLabel }}</span>
          <div class="metric-card__value">{{ resultData.prediction?.label || '-' }}</div>
          <div class="metric-card__meta">当前离线诊断的主要结论</div>
        </article>

        <article class="metric-card">
          <span class="metric-card__label">{{ diagnosisCopy.confidenceLabel }}</span>
          <div class="metric-card__value">{{ confidenceText }}</div>
          <div class="metric-card__meta">来源于模型输出的主分类概率</div>
        </article>
      </div>

      <el-table :data="probabilityRows" stripe :empty-text="diagnosisCopy.emptyProbability">
        <el-table-column prop="label" label="标签" />
        <el-table-column prop="probability" label="概率" />
      </el-table>
    </div>
  </section>
</template>

<script setup>
import { ElTable, ElTableColumn } from 'element-plus/es/components/table/index.mjs'
import { diagnosisCopy } from '@/modules/diagnosis/diagnosisView'

defineProps({
  resultData: {
    type: Object,
    required: true
  },
  confidenceText: {
    type: String,
    required: true
  },
  probabilityRows: {
    type: Array,
    default: () => []
  }
})
</script>

<style scoped>
.diagnosis-result-grid {
  margin-bottom: 18px;
  grid-template-columns: repeat(2, minmax(0, 1fr));
}
</style>
