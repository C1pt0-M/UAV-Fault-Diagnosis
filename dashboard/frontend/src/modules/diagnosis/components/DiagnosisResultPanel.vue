<template>
  <section class="section-card diagnosis-result">
    <div class="section-card__header">
      <div>
        <div class="section-card__title">{{ diagnosisCopy.resultTitle }}</div>
        <div class="section-card__desc">{{ diagnosisCopy.resultDescription }}</div>
      </div>
      <span class="diagnosis-result__status">{{ stage.label }}</span>
    </div>
    <div class="section-card__body">
      <template v-if="resultData">
        <div class="diagnosis-result__hero">
          <div class="diagnosis-result__summary">
            <span class="diagnosis-result__eyebrow">{{ diagnosisCopy.predictionLabel }}</span>
            <h3>{{ labelMeta.text }}</h3>
            <p>{{ labelMeta.summary }}</p>
          </div>
          <div class="diagnosis-result__highlight">
            <span>{{ diagnosisCopy.confidenceLabel }}</span>
            <strong>{{ confidenceText }}</strong>
            <small>主分类置信度</small>
          </div>
        </div>

        <section class="two-column diagnosis-result__grid">
          <article class="section-card diagnosis-result__panel diagnosis-result__panel--focus">
            <div class="section-card__header">
              <div>
                <div class="section-card__title">{{ diagnosisCopy.recommendationTitle }}</div>
                <div class="section-card__desc">{{ labelMeta.riskBadge }}</div>
              </div>
            </div>
            <div class="section-card__body">
              <div class="diagnosis-recommendation">
                <h4>{{ labelMeta.actionTitle }}</h4>
                <p>{{ labelMeta.actionSummary }}</p>
                <ul>
                  <li v-for="item in labelMeta.actionItems" :key="item">{{ item }}</li>
                </ul>
              </div>
            </div>
          </article>

          <article class="section-card diagnosis-result__panel">
            <div class="section-card__header">
              <div class="section-card__title">{{ diagnosisCopy.probabilityTitle }}</div>
            </div>
            <div class="section-card__body">
              <div class="probability-list">
                <div
                  v-for="row in probabilityRows"
                  :key="row.label"
                  class="probability-row"
                >
                  <div class="probability-row__head">
                    <span>{{ row.label }}</span>
                    <strong>{{ row.probability }}</strong>
                  </div>
                  <div class="probability-row__track">
                    <div class="probability-row__fill" :style="{ width: row.probability }" />
                  </div>
                </div>
              </div>
            </div>
          </article>
        </section>

        <section class="section-card diagnosis-result__actions">
          <div class="section-card__header">
            <div class="section-card__title">{{ diagnosisCopy.resultActionTitle }}</div>
          </div>
          <div class="section-card__body">
            <div class="diagnosis-actions">
              <el-button
                v-if="!latestReportId && !reportCreating"
                type="primary"
                size="large"
                @click="$emit('create-report')"
              >
                {{ diagnosisCopy.reportButton }}
              </el-button>
              <el-button
                v-else-if="reportCreating"
                type="primary"
                size="large"
                loading
              >
                {{ diagnosisCopy.reportLoadingButton }}
              </el-button>
              <el-button
                v-else
                type="success"
                size="large"
                plain
              >
                {{ diagnosisCopy.reportReadyButton }}
              </el-button>

              <el-button plain @click="$emit('open-reports-center')">
                {{ diagnosisCopy.reportCenterButton }}
              </el-button>
              <el-button
                type="warning"
                plain
                :disabled="!latestReportId"
                @click="$emit('open-ai-assistant')"
              >
                {{ diagnosisCopy.aiActionButton }}
              </el-button>
            </div>
          </div>
        </section>
      </template>

      <el-empty v-else :description="diagnosisCopy.emptyRun" :image-size="68" />
    </div>
  </section>
</template>

<script setup>
import ElButton from 'element-plus/es/components/button/index.mjs'
import { computed } from 'vue'
import ElEmpty from 'element-plus/es/components/empty/index.mjs'
import { diagnosisCopy, getDiagnosisLabelMeta } from '@/modules/diagnosis/diagnosisView'

const props = defineProps({
  resultData: {
    type: Object,
    default: null
  },
  confidenceText: {
    type: String,
    required: true
  },
  latestReportId: {
    type: Number,
    default: null
  },
  probabilityRows: {
    type: Array,
    default: () => []
  },
  reportCreating: {
    type: Boolean,
    default: false
  },
  stage: {
    type: Object,
    required: true
  }
})

defineEmits(['create-report', 'open-reports-center', 'open-ai-assistant'])

const labelMeta = computed(() => getDiagnosisLabelMeta(props.resultData?.prediction?.label))
</script>

<style scoped>
.diagnosis-result {
  overflow: hidden;
}

.diagnosis-result__status {
  display: inline-flex;
  padding: 6px 10px;
  border-radius: 999px;
  background: var(--brand-soft);
  color: var(--brand-strong);
  font-size: 12px;
  font-weight: 700;
}

.diagnosis-result__hero {
  display: grid;
  grid-template-columns: minmax(0, 1.2fr) minmax(220px, 0.8fr);
  gap: 16px;
  padding: 22px;
  border-radius: 22px;
  background:
    radial-gradient(circle at top right, rgba(255, 255, 255, 0.14), transparent 28%),
    linear-gradient(135deg, #0f2f3d 0%, #0f766e 100%);
  color: #f4fbfb;
}

.diagnosis-result__eyebrow {
  display: inline-flex;
  padding: 6px 10px;
  border-radius: 999px;
  background: rgba(255, 255, 255, 0.14);
  font-size: 12px;
  font-weight: 700;
}

.diagnosis-result__summary h3 {
  margin: 14px 0 0;
  font-size: 38px;
}

.diagnosis-result__summary p {
  margin: 12px 0 0;
  line-height: 1.8;
  color: rgba(244, 251, 251, 0.86);
}

.diagnosis-result__highlight {
  padding: 18px;
  border-radius: 18px;
  background: rgba(255, 255, 255, 0.12);
  border: 1px solid rgba(255, 255, 255, 0.14);
}

.diagnosis-result__highlight span,
.diagnosis-result__highlight small {
  color: rgba(244, 251, 251, 0.78);
}

.diagnosis-result__highlight strong {
  display: block;
  margin-top: 10px;
  font-size: 40px;
  line-height: 1;
}

.diagnosis-result__highlight small {
  display: block;
  margin-top: 10px;
}

.diagnosis-result__grid {
  margin-top: 18px;
  align-items: stretch;
}

.diagnosis-result__panel {
  box-shadow: none;
}

.diagnosis-result__panel--focus {
  background: linear-gradient(180deg, #fbfdfd 0%, #f3f8fa 100%);
}

.diagnosis-recommendation h4 {
  margin: 0;
  font-size: 24px;
  line-height: 1.3;
}

.diagnosis-recommendation p {
  margin: 12px 0 0;
  line-height: 1.8;
  color: var(--text-secondary);
}

.diagnosis-recommendation ul {
  margin: 16px 0 0;
  padding-left: 18px;
  color: var(--text-secondary);
  line-height: 1.9;
}

.probability-list {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.probability-row__head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 6px;
}

.probability-row__track {
  height: 10px;
  border-radius: 999px;
  background: #e8f1f2;
  overflow: hidden;
}

.probability-row__fill {
  height: 100%;
  border-radius: inherit;
  background: linear-gradient(90deg, #0f766e 0%, #1a9c90 100%);
}

.diagnosis-result__actions {
  margin-top: 18px;
}

.diagnosis-actions {
  display: flex;
  gap: 12px;
  flex-wrap: wrap;
}
</style>
