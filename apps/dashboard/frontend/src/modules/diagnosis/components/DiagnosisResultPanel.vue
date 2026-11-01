<template>
  <section class="section-card diagnosis-result">
    <div class="section-card__header">
      <div>
        <div class="section-card__title">{{ diagnosisCopy.resultTitle }}</div>
        <div class="section-card__desc">
          {{ resultData ? diagnosisCopy.resultDescription : stage.description }}
        </div>
      </div>
      <span class="status-badge" :class="riskBadgeClass">{{ stage.label }}</span>
    </div>

    <div class="section-card__body diagnosis-result__body">
      <template v-if="resultData">
        <div class="diagnosis-result__hero">
          <div class="diagnosis-result__summary">
            <span class="diagnosis-result__eyebrow">{{ diagnosisCopy.predictionLabel }}</span>
            <div class="diagnosis-result__headline">
              <h3>{{ labelMeta.text }}</h3>
              <span class="diagnosis-risk" :class="`diagnosis-risk--${riskLevel.tone}`">
                {{ riskLevel.label }}
              </span>
            </div>
            <p>{{ labelMeta.summary }}</p>
          </div>

          <div class="diagnosis-result__confidence">
            <span>{{ diagnosisCopy.confidenceLabel }}</span>
            <strong>{{ confidenceText }}</strong>
          </div>
        </div>

        <div class="diagnosis-meta-grid">
          <div
            v-for="item in resultMetaItems"
            :key="item.label"
            class="diagnosis-meta-card"
          >
            <span>{{ item.label }}</span>
            <strong :title="String(item.value)">{{ item.value }}</strong>
          </div>
        </div>

        <div class="diagnosis-primary-actions">
          <el-button
            v-if="!latestReportId && !reportCreating"
            type="primary"
            @click="$emit('create-report')"
          >
            {{ diagnosisCopy.reportButton }}
          </el-button>
          <el-button
            v-else-if="reportCreating"
            type="primary"
            loading
          >
            {{ diagnosisCopy.reportLoadingButton }}
          </el-button>
          <el-button
            v-else
            type="primary"
            @click="$emit('open-reports-center')"
          >
            {{ diagnosisCopy.reportCenterButton }}
          </el-button>
          <el-button
            v-if="latestReportId && runtimeCapabilities.ai"
            type="warning"
            plain
            @click="$emit('open-ai-assistant')"
          >
            {{ diagnosisCopy.aiActionButton }}
          </el-button>
        </div>

        <p v-if="!runtimeCapabilities.ai" role="status">AI 解读未启用，仍可生成和查看诊断报告。</p>

        <div class="diagnosis-result__lower">
          <section class="diagnosis-subpanel">
            <div class="diagnosis-subpanel__title">{{ diagnosisCopy.probabilityTitle }}</div>
            <div v-if="probabilityRows.length" class="probability-list">
              <div
                v-for="(row, index) in probabilityRows"
                :key="row.label"
                class="probability-row"
              >
                <span class="probability-row__label" :title="row.label">{{ row.label }}</span>
                <div class="probability-row__track">
                  <div
                    class="probability-row__fill"
                    :class="index === 0 ? 'probability-row__fill--danger' : ''"
                    :style="{ width: row.probability }"
                  />
                </div>
                <strong>{{ row.probability }}</strong>
              </div>
            </div>
            <el-empty v-else :description="diagnosisCopy.emptyProbability" :image-size="52" />
          </section>
        </div>

      </template>

      <section v-else-if="stage.key === 'error'" class="diagnosis-ready diagnosis-ready--error">
        <div class="diagnosis-ready__header">
          <div>
            <span class="diagnosis-result__eyebrow diagnosis-result__eyebrow--danger">诊断失败</span>
            <h3>离线诊断任务执行失败</h3>
            <p>系统已停止本次诊断任务；请查看错误原因，确认资产与模型可用后重新提交。</p>
          </div>
        </div>

        <div class="diagnosis-error-box">
          <span>错误原因</span>
          <strong>{{ runErrorText || '暂无错误详情' }}</strong>
        </div>

      </section>

      <section v-else class="diagnosis-ready">
        <div class="diagnosis-ready__header">
          <span class="diagnosis-result__eyebrow">待开始</span>
          <h3>等待诊断任务</h3>
          <p>选择日志资产并启动诊断后，这里将展示故障类别、置信度和类别概率。</p>
        </div>
      </section>
    </div>
  </section>
</template>

<script setup>
import ElButton from 'element-plus/es/components/button/index.mjs'
import ElEmpty from 'element-plus/es/components/empty/index.mjs'
import { computed } from 'vue'
import { runtimeCapabilities } from '@/shared/state/runtimeCapabilities'
import { diagnosisCopy, getDiagnosisLabelMeta } from '@/modules/diagnosis/diagnosisView'

const props = defineProps({
  confidenceText: { type: String, required: true },
  latestReportId: { type: [Number, String], default: null },
  probabilityRows: { type: Array, default: () => [] },
  reportCreating: { type: Boolean, default: false },
  resultData: { type: Object, default: null },
  resultMetaItems: { type: Array, default: () => [] },
  runErrorText: { type: String, default: '' },
  riskLevel: {
    type: Object,
    default: () => ({ label: '待确认', tone: 'pending', description: '暂无风险等级。' })
  },
  stage: { type: Object, required: true }
})

defineEmits(['create-report', 'open-reports-center', 'open-ai-assistant'])

const labelMeta = computed(() => getDiagnosisLabelMeta(props.resultData?.prediction?.label))
const riskBadgeClass = computed(() => {
  if (props.stage.key === 'error') return 'status-badge--danger'
  if (props.riskLevel.tone === 'danger') return 'status-badge--danger'
  if (props.riskLevel.tone === 'warn') return 'status-badge--warn'
  if (props.riskLevel.tone === 'ok') return 'status-badge--ok'
  return 'status-badge--active'
})
</script>

<style scoped>
.diagnosis-result {
  overflow: hidden;
}

.diagnosis-result__body {
  padding: var(--space-lg);
}

.diagnosis-result__hero {
  display: grid;
  grid-template-columns: minmax(0, 1fr) 160px;
  gap: var(--space-md);
  padding: 4px 0 18px;
  border-bottom: 1px solid var(--line-soft);
}

.diagnosis-result__eyebrow {
  display: inline-flex;
  padding: 4px 8px;
  border-radius: var(--radius-xs);
  background: var(--brand-soft);
  color: var(--brand);
  font-size: 11px;
  font-weight: 800;
}

.diagnosis-result__eyebrow--danger {
  color: var(--status-danger);
  background: var(--status-danger-soft);
}

.diagnosis-result__headline {
  display: flex;
  align-items: center;
  gap: 10px;
  flex-wrap: wrap;
  margin-top: 10px;
}

.diagnosis-result__headline h3 {
  margin: 0;
  font-size: 28px;
  line-height: 1.2;
  color: var(--text-primary);
}

.diagnosis-risk {
  display: inline-flex;
  padding: 4px 8px;
  border-radius: var(--radius-xs);
  font-size: 12px;
  font-weight: 800;
}

.diagnosis-risk--danger {
  color: var(--status-danger);
  background: var(--status-danger-soft);
}

.diagnosis-risk--warn {
  color: var(--status-warn);
  background: var(--status-warn-soft);
}

.diagnosis-risk--ok {
  color: var(--status-ok);
  background: var(--status-ok-soft);
}

.diagnosis-result__summary p {
  margin: 10px 0 0;
  color: var(--text-secondary);
  line-height: 1.7;
}

.diagnosis-result__confidence {
  min-width: 0;
  padding-left: 16px;
  border-left: 1px solid var(--line-soft);
}

.diagnosis-result__confidence span {
  color: var(--text-tertiary);
}

.diagnosis-result__confidence strong {
  display: block;
  margin-top: 8px;
  font-size: 34px;
  line-height: 1;
  color: var(--status-danger);
  font-family: var(--font-mono);
}

.diagnosis-ready {
  display: flex;
  flex-direction: column;
  gap: var(--space-md);
}

.diagnosis-ready__header {
  padding: 12px 0;
}

.diagnosis-ready__header h3 {
  margin: 10px 0 0;
  color: var(--text-primary);
  font-size: 24px;
  line-height: 1.2;
}

.diagnosis-ready__header p {
  margin: 8px 0 0;
  color: var(--text-secondary);
  line-height: 1.7;
}

.diagnosis-ready--error .diagnosis-ready__header {
  padding: 14px;
  border-left: 3px solid var(--status-danger);
  background: var(--status-danger-soft);
}

.diagnosis-error-box {
  display: grid;
  gap: 8px;
  padding: 14px;
  border: 1px solid rgba(239, 68, 68, 0.24);
  border-radius: var(--radius-md);
  background: var(--bg-panel);
}

.diagnosis-error-box span {
  color: var(--text-tertiary);
  font-size: 12px;
  font-weight: 800;
}

.diagnosis-error-box strong {
  color: var(--status-danger);
  font-size: 12px;
  line-height: 1.7;
  word-break: break-word;
  white-space: pre-wrap;
}

.diagnosis-meta-grid {
  display: flex;
  flex-wrap: wrap;
  gap: 8px 24px;
  margin-top: 12px;
}

.diagnosis-meta-card {
  min-width: 0;
  display: flex;
  align-items: baseline;
  gap: 6px;
}

.diagnosis-meta-card span {
  color: var(--text-tertiary);
  font-size: 12px;
}

.diagnosis-meta-card strong {
  min-width: 0;
  color: var(--text-primary);
  font-size: 12px;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.diagnosis-primary-actions {
  display: flex;
  gap: 10px;
  flex-wrap: wrap;
  justify-content: flex-start;
  margin-top: var(--space-md);
  padding-bottom: var(--space-md);
  border-bottom: 1px solid var(--line-soft);
}

.diagnosis-result__lower {
  margin-top: var(--space-md);
}

.diagnosis-subpanel {
  min-width: 0;
  padding: 0;
}

.diagnosis-subpanel__title {
  color: var(--text-primary);
  font-size: 13px;
  font-weight: 900;
  margin-bottom: 12px;
}

.probability-list {
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.probability-row {
  display: grid;
  grid-template-columns: minmax(100px, 0.8fr) minmax(160px, 1.8fr) 58px;
  gap: 8px;
  align-items: center;
  color: var(--text-secondary);
  font-size: 12px;
}

.probability-row__label {
  min-width: 0;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.probability-row__track {
  height: 7px;
  border-radius: 999px;
  background: var(--bg-panel-strong);
  overflow: hidden;
}

.probability-row__fill {
  height: 100%;
  border-radius: inherit;
  background: var(--brand);
}

.probability-row__fill--danger {
  background: var(--status-danger);
}

.probability-row strong {
  font-family: var(--font-mono);
  color: var(--text-primary);
  text-align: right;
}

@media (max-width: 760px) {
  .diagnosis-result__hero,
  .diagnosis-result__lower {
    grid-template-columns: 1fr;
  }

  .diagnosis-result__confidence {
    padding-left: 0;
    padding-top: 14px;
    border-left: 0;
    border-top: 1px solid var(--line-soft);
  }

  .probability-row {
    grid-template-columns: minmax(88px, 1fr) minmax(80px, 1.4fr) 54px;
  }
}
</style>
