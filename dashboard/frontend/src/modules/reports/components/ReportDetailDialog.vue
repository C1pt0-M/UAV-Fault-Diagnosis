<template>
  <el-dialog v-model="dialogVisible" :title="reportsCopy.detailTitle" width="820px">
    <template v-if="report">
      <div class="report-stack">
        <section class="report-hero">
          <div class="report-hero__main">
            <span class="report-hero__label">{{ view.labelText }}</span>
            <h3>{{ view.headline }}</h3>
            <p>{{ view.riskText }}</p>
          </div>

          <div class="report-hero__metrics">
            <div class="report-hero__metric">
              <span>{{ reportsCopy.confidenceLabel }}</span>
              <strong>{{ topProbabilityText }}</strong>
            </div>
            <div class="report-hero__metric">
              <span>{{ reportsCopy.modelLabel }}</span>
              <strong>{{ report.model_name || '-' }}</strong>
            </div>
            <div class="report-hero__metric">
              <span>{{ reportsCopy.generatedAtLabel }}</span>
              <strong>{{ report.created_at || '-' }}</strong>
            </div>
          </div>
        </section>

        <section class="section-card">
          <div class="section-card__header">
            <div>
              <div class="section-card__title">{{ reportsCopy.summarySection }}</div>
              <div class="section-card__desc">
                {{ reportsCopy.assetLabel }}：#{{ report.asset_id || '-' }}
              </div>
            </div>
          </div>
          <div class="section-card__body">
            <p class="report-text">{{ view.actionSummary }}</p>
          </div>
        </section>

        <section class="two-column">
          <article class="section-card">
            <div class="section-card__header">
              <div class="section-card__title">{{ reportsCopy.recommendationSection }}</div>
            </div>
            <div class="section-card__body">
              <div class="report-list">
                <div
                  v-for="item in view.actionItems"
                  :key="`${item.priorityText}-${item.title}`"
                  class="report-list__item"
                >
                  <span class="report-priority">{{ item.priorityText }}</span>
                  <div>{{ item.title }}</div>
                </div>
              </div>
            </div>
          </article>

          <article class="section-card">
            <div class="section-card__header">
              <div class="section-card__title">{{ reportsCopy.evidenceSection }}</div>
            </div>
            <div class="section-card__body">
              <div class="report-list">
                <div
                  v-for="item in view.whyItems"
                  :key="item.title"
                  class="report-list__item"
                >
                  <strong>{{ item.title }}</strong>
                  <div>{{ item.description }}</div>
                </div>
              </div>
            </div>
          </article>
        </section>

        <section class="section-card">
          <div class="section-card__header">
            <div class="section-card__title">{{ reportsCopy.probabilitySection }}</div>
          </div>
          <div class="section-card__body">
            <div class="probability-list">
              <div
                v-for="row in view.probabilityRows"
                :key="row.label"
                class="probability-row"
              >
                <div class="probability-row__head">
                  <span>{{ row.labelText }}</span>
                  <strong>{{ row.probText }}</strong>
                </div>
                <div class="probability-row__track">
                  <div class="probability-row__fill" :style="{ width: `${Math.max(6, row.prob * 100)}%` }" />
                </div>
              </div>
            </div>
          </div>
        </section>

        <details class="technical-details">
          <summary>{{ reportsCopy.technicalSection }}</summary>
          <pre class="report-json">{{ detailJson }}</pre>
        </details>
      </div>
    </template>

    <div v-else class="report-empty">{{ reportsCopy.noReportDetail }}</div>
  </el-dialog>
</template>

<script setup>
import { computed } from 'vue'
import ElDialog from 'element-plus/es/components/dialog/index.mjs'
import { reportsCopy } from '@/modules/reports/reportsView'
import { buildDiagnosisReportView } from '@/utils/diagnosisReportView'

const props = defineProps({
  visible: {
    type: Boolean,
    default: false
  },
  report: {
    type: Object,
    default: null
  },
  detailJson: {
    type: String,
    default: '{}'
  }
})

const emit = defineEmits(['update:visible'])

const dialogVisible = computed({
  get: () => props.visible,
  set: (value) => emit('update:visible', value)
})

const view = computed(() => buildDiagnosisReportView(props.report || {}))

const topProbabilityText = computed(() => {
  return view.value?.probabilityRows?.[0]?.probText || '-'
})
</script>

<style scoped>
.report-stack {
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.report-hero {
  display: grid;
  grid-template-columns: minmax(0, 1.3fr) minmax(240px, 0.9fr);
  gap: 16px;
  padding: 20px;
  border-radius: 20px;
  background: linear-gradient(135deg, #103240 0%, #0f766e 100%);
  color: #f4fbfb;
}

.report-hero__label {
  display: inline-flex;
  padding: 6px 10px;
  border-radius: 999px;
  background: rgba(255, 255, 255, 0.16);
  font-size: 12px;
  font-weight: 700;
}

.report-hero__main h3 {
  margin: 14px 0 0;
  font-size: 26px;
}

.report-hero__main p {
  margin: 10px 0 0;
  line-height: 1.8;
  color: rgba(244, 251, 251, 0.88);
}

.report-hero__metrics {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.report-hero__metric {
  padding: 14px 16px;
  border-radius: 16px;
  background: rgba(255, 255, 255, 0.12);
  border: 1px solid rgba(255, 255, 255, 0.14);
}

.report-hero__metric span {
  font-size: 12px;
  color: rgba(244, 251, 251, 0.78);
}

.report-hero__metric strong {
  display: block;
  margin-top: 6px;
  font-size: 18px;
  line-height: 1.4;
}

.report-text {
  margin: 0;
  line-height: 1.8;
  color: var(--text-secondary);
}

.report-list {
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.report-list__item {
  padding: 12px 14px;
  border-radius: 14px;
  border: 1px solid var(--line-soft);
  background: #f7fbfb;
  line-height: 1.7;
}

.report-priority {
  display: inline-flex;
  margin-bottom: 6px;
  padding: 4px 8px;
  border-radius: 999px;
  background: var(--brand-soft);
  color: var(--brand-strong);
  font-size: 12px;
  font-weight: 700;
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

.technical-details summary {
  cursor: pointer;
  color: var(--brand-strong);
  font-weight: 700;
}

.technical-details {
  border-top: 1px dashed var(--line-strong);
  padding-top: 12px;
}

.report-json {
  margin: 12px 0 0;
  max-height: 420px;
  overflow: auto;
  padding: 16px;
  border-radius: 16px;
  background: #0f1f26;
  color: #f4fbfb;
}

.report-empty {
  color: var(--text-secondary);
}
</style>
