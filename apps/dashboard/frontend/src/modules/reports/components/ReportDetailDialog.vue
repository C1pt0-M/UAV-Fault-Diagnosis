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
  visible: { type: Boolean, default: false },
  report: { type: Object, default: null },
  detailJson: { type: String, default: '{}' }
})

const emit = defineEmits(['update:visible'])

const dialogVisible = computed({
  get: () => props.visible,
  set: (value) => emit('update:visible', value)
})

const view = computed(() => buildDiagnosisReportView(props.report || {}))
const topProbabilityText = computed(() => view.value?.probabilityRows?.[0]?.probText || '-')
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
  padding: 16px;
  border-radius: var(--radius-lg);
  background: var(--bg-panel-raised);
  border: 1px solid var(--line-soft);
  border-left: 3px solid var(--brand);
}

.report-hero__label {
  display: inline-flex;
  padding: 3px 8px;
  border-radius: var(--radius-sm);
  background: var(--brand-soft);
  color: var(--brand);
  font-size: 11px;
  font-weight: 700;
}

.report-hero__main h3 {
  margin: 12px 0 0;
  font-size: 22px;
  color: var(--text-primary);
}

.report-hero__main p {
  margin: 8px 0 0;
  line-height: 1.7;
  color: var(--text-secondary);
}

.report-hero__metrics {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.report-hero__metric {
  padding: 10px 12px;
  border-radius: var(--radius-md);
  background: var(--bg-panel);
  border: 1px solid var(--line-strong);
}

.report-hero__metric span {
  font-size: 11px;
  color: var(--text-tertiary);
}

.report-hero__metric strong {
  display: block;
  margin-top: 4px;
  font-size: 14px;
  line-height: 1.4;
  color: var(--text-primary);
  font-weight: 600;
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
  padding: 10px 12px;
  border-radius: var(--radius-md);
  border: 1px solid var(--line-soft);
  background: var(--bg-panel-raised);
  line-height: 1.7;
  color: var(--text-secondary);
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
  border-radius: var(--radius-md);
  background: var(--bg-canvas);
  color: var(--text-primary);
  font-family: var(--font-mono);
  font-size: 12px;
  border: 1px solid var(--line-soft);
}

.report-empty {
  color: var(--text-secondary);
}
</style>
