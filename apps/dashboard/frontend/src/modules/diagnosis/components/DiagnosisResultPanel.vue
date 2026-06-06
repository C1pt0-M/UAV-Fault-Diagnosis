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
            <small>主分类置信度</small>
          </div>
        </div>

        <div class="diagnosis-meta-grid">
          <article
            v-for="item in resultMetaItems"
            :key="item.label"
            class="diagnosis-meta-card"
          >
            <span>{{ item.label }}</span>
            <strong :title="String(item.value)">{{ item.value }}</strong>
          </article>
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
            type="success"
            plain
            @click="$emit('open-reports-center')"
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

        <div class="diagnosis-result__lower">
          <section class="diagnosis-subpanel">
            <div class="diagnosis-subpanel__title">{{ diagnosisCopy.probabilityTitle }}</div>
            <div v-if="probabilityRows.length" class="probability-list">
              <div
                v-for="(row, index) in probabilityRows.slice(0, 5)"
                :key="row.label"
                class="probability-row"
              >
                <span class="probability-row__rank">{{ index + 1 }}</span>
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

          <section class="diagnosis-subpanel diagnosis-subpanel--risk">
            <div class="diagnosis-subpanel__title">风险等级</div>
            <div class="risk-meter" :class="`risk-meter--${riskLevel.tone}`">
              <strong>{{ confidenceText }}</strong>
              <span>{{ riskLevel.label }}</span>
            </div>
            <p>{{ riskLevel.description }}</p>
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
          <div>
            <span class="diagnosis-result__eyebrow">待开始</span>
            <h3>诊断准备状态</h3>
            <p>当前尚未提交诊断任务；完成输入配置后，系统会生成故障结论、证据摘要与报告出口。</p>
          </div>
        </div>

        <div class="diagnosis-ready__grid">
          <article
            v-for="item in preparationItems"
            :key="item.label"
            class="diagnosis-ready__item"
          >
            <span class="status-dot" :class="`status-dot--${item.tone}`"></span>
            <div>
              <strong>{{ item.label }}</strong>
              <small>{{ item.value }}</small>
            </div>
          </article>
        </div>

      </section>
    </div>
  </section>
</template>

<script setup>
import ElButton from 'element-plus/es/components/button/index.mjs'
import ElEmpty from 'element-plus/es/components/empty/index.mjs'
import { computed } from 'vue'
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
const preparationItems = [
  { label: '待选择资产', value: '日志资产未锁定', tone: 'offline' },
  { label: '待启动任务', value: '离线诊断未提交', tone: 'active' },
  { label: '报告待生成', value: '结果完成后开放', tone: 'pending' },
  { label: 'AI 解读待解锁', value: '生成报告后可用', tone: 'offline' }
]
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
  padding: 16px;
  border: 1px solid var(--line-soft);
  border-radius: var(--radius-md);
  background: var(--bg-panel-raised);
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

.diagnosis-result__confidence span,
.diagnosis-result__confidence small {
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

.diagnosis-result__confidence small {
  display: block;
  margin-top: 8px;
}

.diagnosis-ready {
  display: flex;
  flex-direction: column;
  gap: var(--space-md);
}

.diagnosis-ready__header {
  padding: 16px;
  border: 1px solid var(--line-soft);
  border-radius: var(--radius-md);
  background: var(--bg-panel-raised);
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

.diagnosis-ready__grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 8px;
}

.diagnosis-ready__item {
  display: flex;
  align-items: flex-start;
  gap: 9px;
  min-width: 0;
  padding: 12px;
  border: 1px solid var(--line-soft);
  border-radius: var(--radius-md);
  background: var(--bg-panel);
}

.diagnosis-ready__item strong {
  display: block;
  color: var(--text-primary);
  font-size: 13px;
}

.diagnosis-ready__item small {
  display: block;
  margin-top: 3px;
  color: var(--text-tertiary);
  font-size: 12px;
}

.diagnosis-ready--error .diagnosis-ready__header {
  border-color: rgba(239, 68, 68, 0.24);
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
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 8px;
  margin-top: var(--space-md);
}

.diagnosis-meta-card {
  min-width: 0;
  padding: 10px;
  border: 1px solid var(--line-soft);
  border-radius: var(--radius-md);
  background: var(--bg-panel);
}

.diagnosis-meta-card span {
  display: block;
  color: var(--text-tertiary);
  font-size: 11px;
  font-weight: 800;
}

.diagnosis-meta-card strong {
  display: block;
  min-width: 0;
  margin-top: 6px;
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
  justify-content: center;
  margin-top: var(--space-md);
  padding-bottom: var(--space-md);
  border-bottom: 1px solid var(--line-soft);
}

.diagnosis-result__lower {
  display: grid;
  grid-template-columns: minmax(0, 1.4fr) minmax(160px, 0.8fr);
  gap: var(--space-md);
  margin-top: var(--space-md);
}

.diagnosis-subpanel {
  min-width: 0;
  padding: 14px;
  border: 1px solid var(--line-soft);
  border-radius: var(--radius-md);
  background: var(--bg-panel);
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
  grid-template-columns: 20px minmax(90px, 1fr) minmax(100px, 1.2fr) 58px;
  gap: 8px;
  align-items: center;
  color: var(--text-secondary);
  font-size: 12px;
}

.probability-row__rank {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 18px;
  height: 18px;
  border-radius: 999px;
  background: var(--brand-soft);
  color: var(--brand-strong);
  font-size: 10px;
  font-weight: 900;
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

.risk-meter {
  display: grid;
  place-items: center;
  width: 112px;
  height: 112px;
  margin: 4px auto 12px;
  border-radius: 999px;
  border: 10px solid var(--brand-soft);
}

.risk-meter--danger {
  border-color: var(--status-danger);
}

.risk-meter--warn {
  border-color: var(--status-warn);
}

.risk-meter strong {
  display: block;
  font-family: var(--font-mono);
  font-size: 22px;
  color: var(--text-primary);
}

.risk-meter span {
  display: block;
  color: var(--text-tertiary);
  font-size: 12px;
}

.diagnosis-subpanel--risk p {
  margin: 0;
  color: var(--text-secondary);
  line-height: 1.7;
  font-size: 12px;
}

@media (max-width: 760px) {
  .diagnosis-result__hero,
  .diagnosis-result__lower,
  .diagnosis-meta-grid {
    grid-template-columns: 1fr;
  }

  .diagnosis-result__confidence {
    padding-left: 0;
    padding-top: 14px;
    border-left: 0;
    border-top: 1px solid var(--line-soft);
  }

  .diagnosis-ready__grid {
    grid-template-columns: 1fr;
  }
}
</style>
