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

        <section class="two-column report-enhanced-grid">
          <article class="section-card">
            <div class="section-card__header">
              <div>
                <div class="section-card__title">{{ reportsCopy.dataQualitySection }}</div>
                <div class="section-card__desc">
                  <span class="report-badge">{{ view.dataQuality.badge }}</span>
                  {{ view.dataQuality.summary }}
                </div>
              </div>
            </div>
            <div class="section-card__body">
              <div class="metric-grid">
                <div
                  v-for="metric in view.dataQuality.metrics"
                  :key="metric.label"
                  class="metric-grid__item"
                >
                  <span>{{ metric.label }}</span>
                  <strong>{{ metric.value }}</strong>
                </div>
              </div>
              <p v-for="warning in view.dataQuality.warnings" :key="warning" class="report-note">
                {{ warning }}
              </p>
            </div>
          </article>

          <article class="section-card">
            <div class="section-card__header">
              <div>
                <div class="section-card__title">{{ reportsCopy.confidenceSection }}</div>
                <div class="section-card__desc">
                  <span class="report-badge report-badge--confidence">{{ view.reportConfidence.badge }}</span>
                  {{ view.reportConfidence.summary }}
                </div>
              </div>
            </div>
            <div class="section-card__body">
              <div class="report-hero__metric report-confidence-score">
                <span>可信度评分</span>
                <strong>{{ view.reportConfidence.scoreText }}</strong>
              </div>
              <ul class="report-reason-list">
                <li v-for="reason in view.reportConfidence.reasons" :key="reason">{{ reason }}</li>
                <li v-if="!view.reportConfidence.reasons.length">依据信息待补充</li>
              </ul>
            </div>
          </article>
        </section>

        <section class="two-column report-enhanced-grid">
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
            <div>
              <div class="section-card__title">{{ reportsCopy.traceabilitySection }}</div>
              <div class="section-card__desc">{{ view.traceability.intendedUse }}</div>
            </div>
          </div>
          <div class="section-card__body">
            <div class="metric-grid metric-grid--wide">
              <div
                v-for="item in view.traceability.items"
                :key="item.label"
                class="metric-grid__item"
              >
                <span>{{ item.label }}</span>
                <strong>{{ item.value }}</strong>
              </div>
            </div>
            <p v-for="limitation in view.traceability.limitations" :key="limitation" class="report-note">
              限制说明：{{ limitation }}
            </p>
          </div>
        </section>

        <section class="section-card">
          <div class="section-card__header">
            <div>
              <div class="section-card__title">{{ reportsCopy.reviewSection }}</div>
              <div class="section-card__desc">
                <span class="report-badge report-badge--review">{{ view.review.badge }}</span>
                {{ view.review.summary }}
              </div>
            </div>
          </div>
          <div class="section-card__body">
            <div class="metric-grid metric-grid--wide">
              <div class="metric-grid__item">
                <span>复核人</span>
                <strong>{{ view.review.reviewer }}</strong>
              </div>
              <div class="metric-grid__item">
                <span>复核时间</span>
                <strong>{{ view.review.reviewedAt }}</strong>
              </div>
              <div class="metric-grid__item metric-grid__item--wide">
                <span>复核结论</span>
                <strong>{{ view.review.conclusion }}</strong>
              </div>
              <div class="metric-grid__item metric-grid__item--wide">
                <span>维修备注</span>
                <strong>{{ view.review.maintenanceNotes }}</strong>
              </div>
            </div>
          </div>
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

.report-enhanced-grid {
  align-items: stretch;
}

.report-badge {
  display: inline-flex;
  margin-right: 8px;
  padding: 2px 8px;
  border-radius: 999px;
  background: var(--brand-soft);
  color: var(--brand-strong);
  font-size: 12px;
  font-weight: 700;
}

.report-badge--confidence {
  background: rgba(46, 125, 50, 0.12);
  color: #2e7d32;
}

.report-badge--review {
  background: rgba(237, 108, 2, 0.12);
  color: #ed6c02;
}

.metric-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 10px;
}

.metric-grid--wide {
  grid-template-columns: repeat(2, minmax(0, 1fr));
}

.metric-grid__item {
  padding: 10px 12px;
  border-radius: var(--radius-md);
  border: 1px solid var(--line-soft);
  background: var(--bg-panel-raised);
}

.metric-grid__item span {
  display: block;
  font-size: 11px;
  color: var(--text-tertiary);
}

.metric-grid__item strong {
  display: block;
  margin-top: 4px;
  font-size: 13px;
  line-height: 1.5;
  color: var(--text-primary);
  font-weight: 600;
  word-break: break-word;
}

.metric-grid__item--wide {
  grid-column: 1 / -1;
}

.report-note {
  margin: 10px 0 0;
  line-height: 1.6;
  color: var(--text-secondary);
  font-size: 13px;
}

.report-reason-list {
  margin: 12px 0 0;
  padding-left: 18px;
  color: var(--text-secondary);
  line-height: 1.7;
}

.report-confidence-score {
  margin-bottom: 4px;
}

@media (max-width: 720px) {
  .report-hero {
    grid-template-columns: 1fr;
  }

  .two-column {
    grid-template-columns: 1fr;
  }

  .metric-grid,
  .metric-grid--wide {
    grid-template-columns: 1fr;
  }
}
</style>
