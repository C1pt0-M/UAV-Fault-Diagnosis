<template>
  <aside class="section-card diagnosis-evidence">
    <div class="section-card__header">
      <div>
        <div class="section-card__title">{{ diagnosisCopy.explanationTitle }}</div>
        <div class="section-card__desc">{{ diagnosisCopy.explanationDescription }}</div>
      </div>
    </div>

    <div v-if="!resultData" class="section-card__body diagnosis-evidence__body diagnosis-evidence__body--empty">
      <section class="diagnosis-explain-card diagnosis-evidence-empty">
        <div class="diagnosis-explain-card__title">
          <el-icon><Document /></el-icon>
          完成诊断后生成规则解释
        </div>
        <p>当前尚未形成诊断结果；完成离线诊断后，这里会展示摘要、证据链、规则依据与历史参考报告。</p>
      </section>

      <div class="diagnosis-evidence-empty__grid">
        <article class="diagnosis-evidence-empty__item">
          <span class="status-dot status-dot--offline"></span>
          <div>
            <strong>诊断摘要待生成</strong>
            <small>等待模型输出故障结论</small>
          </div>
        </article>
        <article class="diagnosis-evidence-empty__item">
          <span class="status-dot status-dot--pending"></span>
          <div>
            <strong>证据链待生成</strong>
            <small>完成诊断后汇总可追溯证据</small>
          </div>
        </article>
        <article class="diagnosis-evidence-empty__item">
          <span class="status-dot status-dot--active"></span>
          <div>
            <strong>历史参考待匹配</strong>
            <small>按故障类别优先匹配报告</small>
          </div>
        </article>
      </div>

      <section class="diagnosis-explain-card diagnosis-ai-entry diagnosis-ai-entry--compact">
        <div>
          <div class="diagnosis-explain-card__title">
            <el-icon><MagicStick /></el-icon>
            AI 解读待解锁
          </div>
          <p>生成诊断报告后开放 AI 解读入口。</p>
        </div>
        <el-button type="warning" plain disabled>进入 AI 解读</el-button>
      </section>
    </div>

    <div v-else class="section-card__body diagnosis-evidence__body">
      <section class="diagnosis-explain-card">
        <div class="diagnosis-explain-card__title">
          <el-icon><Document /></el-icon>
          诊断摘要
        </div>
        <p>{{ explanationSummary }}</p>
      </section>

      <section class="diagnosis-explain-card">
        <div class="diagnosis-explain-card__title">
          <el-icon><Connection /></el-icon>
          {{ diagnosisCopy.evidenceTitle }}
        </div>
        <div v-if="evidenceRows.length" class="evidence-table">
          <div class="evidence-table__head">
            <span>证据项</span>
            <span>来源</span>
            <span>时间窗口 / 记录</span>
            <span>说明</span>
          </div>
          <div
            v-for="row in evidenceRows.slice(0, 5)"
            :key="`${row.rank}-${row.name}`"
            class="evidence-table__row"
          >
            <span class="evidence-table__name">
              <b>{{ row.rank }}</b>
              {{ row.name }}
            </span>
            <span>{{ row.source }}</span>
            <span :title="row.window">{{ row.window }}</span>
            <span :title="row.note">{{ row.note }}</span>
          </div>
        </div>
        <el-empty v-else :description="diagnosisCopy.emptyEvidence" :image-size="54" />
      </section>

      <div class="diagnosis-evidence__grid">
        <section class="diagnosis-explain-card">
          <div class="diagnosis-explain-card__title">
            <el-icon><CircleCheck /></el-icon>
            {{ diagnosisCopy.ruleTitle }}
          </div>
          <div class="rule-list">
            <div
              v-for="row in ruleRows"
              :key="row.title"
              class="rule-list__item"
            >
              <span class="status-dot" :class="`status-dot--${statusTone(row.status)}`"></span>
              <div>
                <strong>{{ row.title }}</strong>
                <p>{{ row.detail }}</p>
              </div>
            </div>
          </div>
        </section>

        <section class="diagnosis-explain-card">
          <div class="diagnosis-explain-card__header">
            <div class="diagnosis-explain-card__title">{{ diagnosisCopy.historyTitle }}</div>
            <el-button link type="primary" @click="$emit('open-reports-center')">更多</el-button>
          </div>
          <div v-if="recentReports.length" class="history-list">
            <div
              v-for="report in recentReports.slice(0, 4)"
              :key="report.id"
              class="history-list__row"
            >
              <span class="history-list__id">RPT-{{ report.id }}</span>
              <span>{{ report.prediction_label || report.label || '-' }}</span>
              <strong>{{ formatReportConfidence(report) }}</strong>
            </div>
          </div>
          <el-empty v-else :description="diagnosisCopy.emptyReports" :image-size="48" />
        </section>
      </div>

      <div class="diagnosis-evidence__grid diagnosis-evidence__grid--bottom">
        <section class="diagnosis-explain-card">
          <div class="diagnosis-explain-card__title">{{ diagnosisCopy.recommendationTitle }}</div>
          <ul class="recommendation-list">
            <li v-for="item in recommendationItems" :key="item">{{ item }}</li>
          </ul>
        </section>

        <section class="diagnosis-explain-card diagnosis-ai-entry">
          <div>
            <div class="diagnosis-explain-card__title">
              <el-icon><MagicStick /></el-icon>
              AI 解读入口
            </div>
            <p>生成报告后，可进入 AI 助手对本次诊断结论进行专业解读与建议整理。</p>
          </div>
          <el-button
            type="warning"
            plain
            :disabled="!latestReportId"
            @click="$emit('open-ai-assistant')"
          >
            进入 AI 解读
          </el-button>
        </section>
      </div>
    </div>
  </aside>
</template>

<script setup>
import { CircleCheck, Connection, Document, MagicStick } from '@element-plus/icons-vue'
import ElButton from 'element-plus/es/components/button/index.mjs'
import ElEmpty from 'element-plus/es/components/empty/index.mjs'
import ElIcon from 'element-plus/es/components/icon/index.mjs'
import { diagnosisCopy, formatPercentage } from '@/modules/diagnosis/diagnosisView'

defineProps({
  evidenceRows: { type: Array, default: () => [] },
  explanationSummary: { type: String, default: '' },
  latestReportId: { type: [Number, String], default: null },
  recentReports: { type: Array, default: () => [] },
  recommendationItems: { type: Array, default: () => [] },
  resultData: { type: Object, default: null },
  riskLevel: { type: Object, default: () => ({}) },
  ruleRows: { type: Array, default: () => [] }
})

defineEmits(['open-ai-assistant', 'open-reports-center'])

function statusTone(status) {
  if (status === 'warn') return 'warn'
  if (status === 'active') return 'active'
  if (status === 'offline') return 'offline'
  return 'ok'
}

function formatReportConfidence(report) {
  const value = report.confidence ?? report.prediction_confidence
  if (value === null || value === undefined || value === '') return '-'
  return formatPercentage(value)
}
</script>

<style scoped>
.diagnosis-evidence {
  overflow: hidden;
}

.diagnosis-evidence__body {
  display: flex;
  flex-direction: column;
  gap: var(--space-md);
  padding: var(--space-lg);
}

.diagnosis-evidence__body--empty {
  gap: 10px;
}

.diagnosis-explain-card {
  min-width: 0;
  padding: 14px;
  border: 1px solid var(--line-soft);
  border-radius: var(--radius-md);
  background: var(--bg-panel);
}

.diagnosis-explain-card__header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 10px;
}

.diagnosis-explain-card__title {
  display: flex;
  align-items: center;
  gap: 8px;
  color: var(--text-primary);
  font-size: 13px;
  font-weight: 900;
}

.diagnosis-explain-card p {
  margin: 10px 0 0;
  color: var(--text-secondary);
  line-height: 1.75;
  font-size: 13px;
}

.diagnosis-evidence-empty p {
  margin-top: 8px;
}

.diagnosis-evidence-empty__grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 8px;
}

.diagnosis-evidence-empty__item {
  display: flex;
  gap: 8px;
  align-items: flex-start;
  min-width: 0;
  padding: 12px;
  border: 1px solid var(--line-soft);
  border-radius: var(--radius-md);
  background: var(--bg-panel);
}

.diagnosis-evidence-empty__item strong {
  display: block;
  color: var(--text-primary);
  font-size: 12px;
}

.diagnosis-evidence-empty__item small {
  display: block;
  margin-top: 3px;
  color: var(--text-tertiary);
  font-size: 11px;
  line-height: 1.5;
}

.evidence-table {
  margin-top: 12px;
  overflow-x: auto;
}

.evidence-table__head,
.evidence-table__row {
  display: grid;
  grid-template-columns: minmax(120px, 1.2fr) minmax(80px, 0.8fr) minmax(120px, 1fr) minmax(140px, 1.4fr);
  gap: 10px;
  min-width: 620px;
  align-items: center;
}

.evidence-table__head {
  padding-bottom: 8px;
  color: var(--text-tertiary);
  font-size: 11px;
  font-weight: 900;
  border-bottom: 1px solid var(--line-soft);
}

.evidence-table__row {
  min-height: 38px;
  border-bottom: 1px solid var(--line-soft);
  color: var(--text-secondary);
  font-size: 12px;
}

.evidence-table__row > span {
  min-width: 0;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.evidence-table__name {
  display: flex;
  align-items: center;
  gap: 8px;
  color: var(--text-primary);
  font-weight: 700;
}

.evidence-table__name b {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 18px;
  height: 18px;
  border-radius: 999px;
  background: var(--brand-soft);
  color: var(--brand-strong);
  font-size: 10px;
}

.diagnosis-evidence__grid {
  display: grid;
  grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
  gap: var(--space-md);
}

.rule-list {
  display: flex;
  flex-direction: column;
  gap: 10px;
  margin-top: 12px;
}

.rule-list__item {
  display: flex;
  gap: 8px;
  align-items: flex-start;
}

.rule-list__item strong {
  color: var(--text-primary);
  font-size: 12px;
}

.rule-list__item p {
  margin: 3px 0 0;
  font-size: 12px;
}

.history-list {
  display: flex;
  flex-direction: column;
  gap: 9px;
  margin-top: 12px;
}

.history-list__row {
  display: grid;
  grid-template-columns: minmax(92px, 1fr) minmax(90px, 1fr) 58px;
  gap: 8px;
  align-items: center;
  color: var(--text-secondary);
  font-size: 12px;
}

.history-list__row > span {
  min-width: 0;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.history-list__id,
.history-list__row strong {
  color: var(--text-primary);
  font-family: var(--font-mono);
}

.recommendation-list {
  margin: 10px 0 0;
  padding-left: 18px;
  color: var(--text-secondary);
  line-height: 1.8;
  font-size: 13px;
}

.diagnosis-ai-entry {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
}

.diagnosis-ai-entry--compact p {
  margin-top: 6px;
}

@media (max-width: 760px) {
  .diagnosis-evidence__grid,
  .diagnosis-evidence-empty__grid,
  .diagnosis-ai-entry {
    grid-template-columns: 1fr;
    flex-direction: column;
    align-items: stretch;
  }
}
</style>
