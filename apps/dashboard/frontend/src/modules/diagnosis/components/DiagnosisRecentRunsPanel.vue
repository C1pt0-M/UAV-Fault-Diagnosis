<template>
  <section class="section-card diagnosis-runs">
    <div class="section-card__header">
      <div>
        <div class="section-card__title">{{ diagnosisCopy.recentRunsTitle }}</div>
        <div class="section-card__desc">{{ diagnosisCopy.recentRunsDescription }}</div>
      </div>
      <span class="status-badge status-badge--active">{{ reportStatusText }}</span>
    </div>

    <div class="section-card__body diagnosis-runs__body">
      <div v-if="recentRuns.length" class="diagnosis-run-table">
        <div class="diagnosis-run-table__head">
          <span>任务 ID</span>
          <span>资产</span>
          <span>故障结论</span>
          <span>Confidence</span>
          <span>状态</span>
          <span>报告状态</span>
          <span>创建时间</span>
          <span>更新时间</span>
          <span>操作</span>
        </div>

        <div
          v-for="run in recentRuns"
          :key="run.id"
          class="diagnosis-run-table__row"
        >
          <strong class="diagnosis-run-table__mono">DG-{{ run.id }}</strong>
          <span :title="resolveAssetText(run)">{{ resolveAssetText(run) }}</span>
          <span>{{ resolvePredictionText(run) }}</span>
          <span class="diagnosis-run-table__mono">{{ resolveConfidenceText(run) }}</span>
          <span class="status-badge" :class="runStatusClass(run.status)">
            {{ runStatusText(run.status) }}
          </span>
          <span class="status-badge" :class="run.report_id ? 'status-badge--ok' : 'status-badge--offline'">
            {{ run.report_id ? '已生成' : '待生成' }}
          </span>
          <span>{{ run.created_at || '-' }}</span>
          <span>{{ run.updated_at || run.finished_at || '-' }}</span>
          <span class="diagnosis-run-table__actions">
            <el-button link type="primary" @click="$emit('open-run-result', run.id)">
              查看结果
            </el-button>
            <el-button link type="success" @click="$emit('open-reports-center')">
              查看报告
            </el-button>
            <el-button
              link
              type="warning"
              :disabled="!run.report_id"
              @click="$emit('open-ai-assistant')"
            >
              AI 解读
            </el-button>
          </span>
        </div>
      </div>

      <el-empty v-else :description="diagnosisCopy.emptyRun" :image-size="64" />
    </div>
  </section>
</template>

<script setup>
import ElButton from 'element-plus/es/components/button/index.mjs'
import ElEmpty from 'element-plus/es/components/empty/index.mjs'
import { diagnosisCopy, formatPercentage, getDiagnosisLabelMeta } from '@/modules/diagnosis/diagnosisView'

defineProps({
  recentRuns: { type: Array, default: () => [] },
  reportStatusText: { type: String, default: '待诊断' }
})

defineEmits(['open-run-result', 'open-reports-center', 'open-ai-assistant'])

function resolveAssetText(run) {
  return run.asset_name || run.asset_filename || (run.asset_id ? `资产 #${run.asset_id}` : '-')
}

function resolvePredictionText(run) {
  const label = run.prediction_label || run.result?.prediction?.label || run.prediction?.label
  return label ? getDiagnosisLabelMeta(label).text : '-'
}

function resolveConfidenceText(run) {
  const value = run.confidence ?? run.prediction_confidence ?? run.result?.prediction?.confidence ?? run.prediction?.confidence
  if (value === null || value === undefined || value === '') return '-'
  return formatPercentage(value)
}

function runStatusText(status) {
  const value = String(status || '').toLowerCase()
  if (['success', 'completed', 'done', 'finished'].includes(value)) return '已完成'
  if (['running', 'queued', 'pending'].includes(value)) return value === 'queued' ? '排队中' : '运行中'
  if (['error', 'failed', 'failure'].includes(value)) return '失败'
  return status || '-'
}

function runStatusClass(status) {
  const value = String(status || '').toLowerCase()
  if (['success', 'completed', 'done', 'finished'].includes(value)) return 'status-badge--ok'
  if (['running', 'queued', 'pending'].includes(value)) return 'status-badge--active'
  if (['error', 'failed', 'failure'].includes(value)) return 'status-badge--danger'
  return 'status-badge--offline'
}
</script>

<style scoped>
.diagnosis-runs {
  overflow: hidden;
}

.diagnosis-runs__body {
  padding: var(--space-lg);
}

.diagnosis-run-table {
  width: 100%;
  overflow-x: auto;
}

.diagnosis-run-table__head,
.diagnosis-run-table__row {
  display: grid;
  grid-template-columns: 104px minmax(170px, 1.2fr) minmax(120px, 1fr) 92px 76px 82px 150px 150px minmax(190px, 1.2fr);
  gap: 10px;
  align-items: center;
  min-width: 1180px;
}

.diagnosis-run-table__head {
  padding-bottom: 10px;
  border-bottom: 1px solid var(--line-soft);
  color: var(--text-tertiary);
  font-size: 11px;
  font-weight: 900;
}

.diagnosis-run-table__row {
  min-height: 44px;
  border-bottom: 1px solid var(--line-soft);
  color: var(--text-secondary);
  font-size: 12px;
}

.diagnosis-run-table__row > span,
.diagnosis-run-table__row > strong {
  min-width: 0;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.diagnosis-run-table__mono {
  font-family: var(--font-mono);
  color: var(--text-primary);
}

.diagnosis-run-table__actions {
  display: flex;
  align-items: center;
  gap: 8px;
  overflow: visible !important;
}
</style>
