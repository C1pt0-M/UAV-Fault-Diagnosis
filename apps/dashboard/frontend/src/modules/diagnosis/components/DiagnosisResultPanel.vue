<template>
  <section class="section-card diagnosis-result">
    <div class="section-card__header">
      <div>
        <div class="section-card__title">{{ diagnosisCopy.resultTitle }}</div>
        <div class="section-card__desc">{{ diagnosisCopy.resultDescription }}</div>
      </div>
      <span class="status-badge status-badge--active">{{ stage.label }}</span>
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
          <article class="section-card diagnosis-result__panel section-card--accent">
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

.diagnosis-result__hero {
  display: grid;
  grid-template-columns: minmax(0, 1.2fr) minmax(220px, 0.8fr);
  gap: 16px;
  padding: 18px;
  border: 1px solid var(--line-soft);
  border-radius: var(--radius-lg);
  background: var(--bg-panel-raised);
}

.diagnosis-result__eyebrow {
  display: inline-flex;
  padding: 4px 8px;
  border-radius: var(--radius-sm);
  background: var(--brand-soft);
  color: var(--brand);
  font-size: 11px;
  font-weight: 700;
}

.diagnosis-result__summary h3 {
  margin: 12px 0 0;
  font-size: 28px;
  line-height: 1.2;
  color: var(--text-primary);
}

.diagnosis-result__summary p {
  margin: 10px 0 0;
  line-height: 1.7;
  color: var(--text-secondary);
}

.diagnosis-result__highlight {
  padding: 16px;
  border-radius: var(--radius-md);
  background: var(--bg-panel);
  border: 1px solid var(--line-strong);
}

.diagnosis-result__highlight span,
.diagnosis-result__highlight small {
  color: var(--text-tertiary);
}

.diagnosis-result__highlight strong {
  display: block;
  margin-top: 8px;
  font-size: 34px;
  line-height: 1;
  color: var(--text-primary);
  font-family: var(--font-mono);
}

.diagnosis-result__highlight small {
  display: block;
  margin-top: 8px;
}

.diagnosis-result__grid {
  margin-top: 18px;
  align-items: stretch;
}

.diagnosis-result__panel {
  box-shadow: none;
}

.diagnosis-recommendation h4 {
  margin: 0;
  font-size: 20px;
  line-height: 1.3;
  color: var(--text-primary);
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
  line-height: 1.8;
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
  color: var(--text-secondary);
}

.probability-row__head strong {
  color: var(--text-primary);
  font-family: var(--font-mono);
}

.probability-row__track {
  height: 8px;
  border-radius: var(--radius-xs);
  background: var(--bg-panel-strong);
  overflow: hidden;
}

.probability-row__fill {
  height: 100%;
  border-radius: inherit;
  background: var(--brand);
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
