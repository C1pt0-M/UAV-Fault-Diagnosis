<template>
  <div class="page-stack">

    <!-- 系统状态条：替代 Hero，展示当前实际运行状态 -->
    <div class="system-statusbar">
      <div class="system-statusbar__item">
        <span class="status-dot status-dot--ok"></span>
        <span>系统在线</span>
      </div>
      <div class="system-statusbar__divider"></div>
      <div class="system-statusbar__item">
        <span>资产总量</span>
        <strong class="status-summary__value">{{ overview.assets_total || 0 }}</strong>
      </div>
      <div class="system-statusbar__divider"></div>
      <div class="system-statusbar__item">
        <span>任务总量</span>
        <strong class="status-summary__value">{{ overview.runs_total || 0 }}</strong>
      </div>
      <div class="system-statusbar__divider"></div>
      <div class="system-statusbar__item">
        <span>成功任务</span>
        <strong class="status-summary__value status-summary__value--ok">{{ successRuns }}</strong>
      </div>
      <div class="system-statusbar__divider"></div>
      <div class="system-statusbar__item">
        <span>主故障类型</span>
        <span
          class="status-badge status-summary__badge"
          :class="dominantFaultBadgeClass"
        >{{ dominantFaultLabel }}</span>
      </div>
    </div>

    <!-- KPI 指标行 -->
    <div class="metric-grid">
      <div class="metric-card metric-card--accent">
        <div class="metric-card__label">累计资产数</div>
        <div class="metric-card__value">{{ overview.assets_total || 0 }}</div>
        <div class="metric-card__meta">可用于预处理、诊断与复盘的日志资产</div>
      </div>

      <div class="metric-card metric-card--ok">
        <div class="metric-card__label">累计任务数</div>
        <div class="metric-card__value">{{ overview.runs_total || 0 }}</div>
        <div class="metric-card__meta">已创建的预处理、诊断与相关任务总数</div>
      </div>

      <div class="metric-card metric-card--ok">
        <div class="metric-card__label">成功任务数</div>
        <div class="metric-card__value">{{ successRuns }}</div>
        <div class="metric-card__meta">已稳定完成并返回结果的任务数量</div>
      </div>

      <div class="metric-card" :class="dominantFaultCardClass">
        <div class="metric-card__label">当前主故障</div>
        <div class="metric-card__value metric-card__value--compact">{{ dominantFaultLabel }}</div>
        <div class="metric-card__meta">{{ dominantFaultMeta }}</div>
      </div>
    </div>

    <!-- 主内容：任务流 + 故障分布 -->
    <div class="two-column">

      <!-- 关键任务流 -->
      <div class="section-card section-card--accent">
        <div class="section-card__header">
          <span class="section-card__title">关键任务流</span>
          <span class="section-card__eyebrow">最近 6 条</span>
        </div>
        <div class="section-card__body">
          <div v-if="recentRuns.length" class="run-list">
            <div
              v-for="item in recentRuns.slice(0, 6)"
              :key="item.id"
              class="data-row run-row"
            >
              <!-- 状态指示点 -->
              <span
                class="status-dot"
                :class="statusDotClass(item.status)"
              ></span>

              <!-- 任务信息 -->
              <div class="run-row__info">
                <span class="run-row__id">#{{ item.id }}</span>
                <span class="run-row__task">{{ normalizeTask(item.task) }}</span>
              </div>

              <!-- 状态 badge -->
              <span
                class="status-badge"
                :class="statusBadgeClass(item.status)"
              >{{ normalizeStatusText(item.status) }}</span>

              <!-- 时间 -->
              <span class="run-row__time">{{ formatTime(item.created_at) }}</span>
            </div>
          </div>
          <div v-else class="empty-hint">当前还没有可展示的任务流记录。</div>
        </div>
      </div>

      <!-- 故障分布面板 -->
      <div class="section-card">
        <div class="section-card__header">
          <span class="section-card__title">诊断结果分布</span>
          <span class="section-card__eyebrow">累计 {{ distributionTotal }} 条</span>
        </div>
        <div class="section-card__body">
          <div v-if="distributionTotal > 0" class="fault-dist">
            <!-- 正常 -->
            <div class="fault-dist__item">
              <div class="fault-dist__label-row">
                <span class="status-dot status-dot--ok"></span>
                <span class="fault-dist__name">正常</span>
                <span class="fault-dist__count">{{ faultDistribution.normal || 0 }}</span>
                <span class="fault-dist__pct">{{ distributionPercent.normal }}%</span>
              </div>
              <div class="fault-dist__track">
                <div
                  class="fault-dist__fill fault-dist__fill--ok"
                  :style="{ width: `${distributionPercent.normal}%` }"
                ></div>
              </div>
            </div>

            <!-- 电机故障 -->
            <div class="fault-dist__item">
              <div class="fault-dist__label-row">
                <span class="status-dot status-dot--danger"></span>
                <span class="fault-dist__name">电机故障</span>
                <span class="fault-dist__count">{{ faultDistribution.motor || 0 }}</span>
                <span class="fault-dist__pct">{{ distributionPercent.motor }}%</span>
              </div>
              <div class="fault-dist__track">
                <div
                  class="fault-dist__fill fault-dist__fill--danger"
                  :style="{ width: `${distributionPercent.motor}%` }"
                ></div>
              </div>
            </div>

            <!-- 传感器故障 -->
            <div class="fault-dist__item">
              <div class="fault-dist__label-row">
                <span class="status-dot status-dot--warn"></span>
                <span class="fault-dist__name">传感器故障</span>
                <span class="fault-dist__count">{{ faultDistribution.sensor || 0 }}</span>
                <span class="fault-dist__pct">{{ distributionPercent.sensor }}%</span>
              </div>
              <div class="fault-dist__track">
                <div
                  class="fault-dist__fill fault-dist__fill--warn"
                  :style="{ width: `${distributionPercent.sensor}%` }"
                ></div>
              </div>
            </div>
          </div>

          <div v-else class="empty-hint">当前还没有成功诊断结果可用于统计。</div>

          <!-- 链路概览：五步流程，简化为状态列表 -->
          <div class="pipeline-list">
            <div class="section-card__title pipeline-list__title">诊断链路</div>
            <div
              v-for="step in flowSteps"
              :key="step.key"
              class="pipeline-item"
            >
              <span class="pipeline-item__index">{{ step.index }}</span>
              <span class="pipeline-item__title">{{ step.title }}</span>
              <span class="pipeline-item__desc">{{ step.description }}</span>
            </div>
          </div>
        </div>
      </div>

    </div>
  </div>
</template>

<script setup>
import { computed, onMounted, ref } from 'vue'
import { ElMessage } from 'element-plus'
import { overviewApi } from '@/shared/services'

const overview = ref({})
const recentRuns = ref([])

const flowSteps = [
  { key: 'telemetry', index: '01', title: '遥测采集', description: 'Replay / PX4 数据统一进入链路' },
  { key: 'detection', index: '02', title: '在线预警', description: '实时检测，形成告警与会话历史' },
  { key: 'diagnosis', index: '03', title: '离线诊断', description: '对资产执行故障分类与置信推断' },
  { key: 'report', index: '04', title: '报告生成', description: '结果整理成可交付诊断报告' },
  { key: 'assistant', index: '05', title: 'AI 解读', description: '结合知识库解释故障原因与风险' }
]

const faultDistribution = computed(() => overview.value?.fault_distribution || { normal: 0, motor: 0, sensor: 0 })
const distributionTotal = computed(() => Object.values(faultDistribution.value).reduce((sum, v) => sum + Number(v || 0), 0))
const successRuns = computed(() => Number(overview.value?.runs_by_status?.success || 0))

const dominantFaultKey = computed(() => {
  const entries = [
    ['motor', Number(faultDistribution.value.motor || 0)],
    ['sensor', Number(faultDistribution.value.sensor || 0)],
    ['normal', Number(faultDistribution.value.normal || 0)]
  ]
  entries.sort((a, b) => b[1] - a[1])
  return entries[0]?.[0] || 'normal'
})

const dominantFaultLabel = computed(() => {
  if (distributionTotal.value === 0) return '暂无数据'
  if (dominantFaultKey.value === 'motor') return '电机故障'
  if (dominantFaultKey.value === 'sensor') return '传感器故障'
  return '正常样本'
})

const dominantFaultMeta = computed(() => {
  if (distributionTotal.value === 0) return '当前还没有成功诊断结果可用于统计。'
  const count = Number(faultDistribution.value[dominantFaultKey.value] || 0)
  return `当前分布中数量最多，共 ${count} 条。`
})

const dominantFaultCardClass = computed(() => {
  if (distributionTotal.value === 0) return 'metric-card--accent'
  if (dominantFaultKey.value === 'motor') return 'metric-card--danger'
  if (dominantFaultKey.value === 'sensor') return 'metric-card--warn'
  return 'metric-card--ok'
})

const dominantFaultBadgeClass = computed(() => {
  if (distributionTotal.value === 0) return 'status-badge--offline'
  if (dominantFaultKey.value === 'motor') return 'status-badge--danger'
  if (dominantFaultKey.value === 'sensor') return 'status-badge--warn'
  return 'status-badge--ok'
})

const distributionPercent = computed(() => {
  const total = distributionTotal.value
  if (!total) return { normal: 0, motor: 0, sensor: 0 }
  return {
    normal: Math.round((Number(faultDistribution.value.normal || 0) / total) * 100),
    motor: Math.round((Number(faultDistribution.value.motor || 0) / total) * 100),
    sensor: Math.round((Number(faultDistribution.value.sensor || 0) / total) * 100)
  }
})

function normalizeTask(task) {
  if (task === 'offline_multiclass') return '离线诊断'
  if (task === 'convert_ulg') return '预处理'
  return task || '未知任务'
}

function normalizeStatusText(status) {
  if (status === 'success') return '已完成'
  if (status === 'error' || status === 'failed') return '失败'
  if (status === 'running') return '执行中'
  if (status === 'queued') return '排队中'
  return status || '未知'
}

function statusDotClass(status) {
  if (status === 'success') return 'status-dot--ok'
  if (status === 'error' || status === 'failed') return 'status-dot--danger'
  if (status === 'running') return 'status-dot--active'
  if (status === 'queued') return 'status-dot--pending'
  return 'status-dot--offline'
}

function statusBadgeClass(status) {
  if (status === 'success') return 'status-badge--ok'
  if (status === 'error' || status === 'failed') return 'status-badge--danger'
  if (status === 'running') return 'status-badge--active'
  if (status === 'queued') return 'status-badge--pending'
  return 'status-badge--offline'
}

function formatTime(ts) {
  if (!ts) return '-'
  try {
    return new Date(ts).toLocaleString('zh-CN', {
      month: '2-digit',
      day: '2-digit',
      hour: '2-digit',
      minute: '2-digit'
    })
  } catch {
    return ts
  }
}

async function loadOverview() {
  try {
    const response = await overviewApi.getOverview()
    overview.value = response.data || {}
    recentRuns.value = response.data?.recent_runs || []
  } catch (error) {
    ElMessage.error(`加载总览失败：${error.message || '未知错误'}`)
  }
}

onMounted(loadOverview)
</script>

<style scoped>
.status-summary__value {
  margin-left: var(--space-xs);
  color: var(--text-primary);
  font-family: var(--font-mono);
  font-weight: 700;
}

.status-summary__value--ok {
  color: var(--status-ok);
}

.status-summary__badge {
  margin-left: 6px;
}

.metric-card__value--compact {
  padding-top: var(--space-xs);
  font-size: 18px;
}

.section-card__eyebrow {
  font-size: 11px;
  color: var(--text-tertiary);
}

/* ── 任务流列表 ─────────────────────────────────────────────── */
.run-list {
  display: flex;
  flex-direction: column;
}

.run-row {
  display: grid;
  grid-template-columns: 7px 1fr auto auto;
  gap: 10px;
  align-items: center;
}

.run-row__info {
  display: flex;
  align-items: center;
  gap: 6px;
  min-width: 0;
}

.run-row__id {
  font-size: 11px;
  color: var(--text-tertiary);
  font-family: var(--font-mono);
  flex-shrink: 0;
}

.run-row__task {
  font-size: 13px;
  color: var(--text-primary);
  font-weight: 500;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.run-row__time {
  font-size: 11px;
  color: var(--text-tertiary);
  font-family: var(--font-mono);
  white-space: nowrap;
  flex-shrink: 0;
}

/* ── 故障分布 ────────────────────────────────────────────────── */
.fault-dist {
  display: flex;
  flex-direction: column;
  gap: 14px;
  margin-bottom: 20px;
  padding-bottom: 20px;
  border-bottom: 1px solid var(--line-soft);
}

.fault-dist__item {
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.fault-dist__label-row {
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: 12px;
}

.fault-dist__name {
  flex: 1;
  color: var(--text-secondary);
}

.fault-dist__count {
  font-family: var(--font-mono);
  font-size: 12px;
  color: var(--text-primary);
  font-weight: 600;
}

.fault-dist__pct {
  font-family: var(--font-mono);
  font-size: 11px;
  color: var(--text-tertiary);
  min-width: 32px;
  text-align: right;
}

.fault-dist__track {
  height: 5px;
  border-radius: var(--radius-xs);
  background: var(--bg-panel-strong);
  overflow: hidden;
}

.fault-dist__fill {
  height: 100%;
  border-radius: inherit;
  transition: width 0.4s ease;
}

.fault-dist__fill--ok     { background: var(--status-ok); }
.fault-dist__fill--danger { background: var(--status-danger); }
.fault-dist__fill--warn   { background: var(--status-warn); }

/* ── 诊断链路（简化版五步流） ──────────────────────────────── */
.pipeline-list {
  display: flex;
  flex-direction: column;
  gap: 6px;
  padding-top: 4px;
}

.pipeline-list__title {
  margin-bottom: 10px;
}

.pipeline-item {
  display: grid;
  grid-template-columns: 22px auto 1fr;
  gap: 8px;
  align-items: center;
  padding: 5px 0;
  font-size: 12px;
  border-bottom: 1px solid var(--line-faint);
}

.pipeline-item:last-child {
  border-bottom: none;
}

.pipeline-item__index {
  font-family: var(--font-mono);
  font-size: 10px;
  color: var(--text-tertiary);
}

.pipeline-item__title {
  color: var(--text-primary);
  font-weight: 600;
  white-space: nowrap;
}

.pipeline-item__desc {
  color: var(--text-tertiary);
  text-overflow: ellipsis;
  overflow: hidden;
  white-space: nowrap;
}

/* ── 空状态提示 ─────────────────────────────────────────────── */
.empty-hint {
  padding: 20px 0;
  text-align: center;
  font-size: 13px;
  color: var(--text-tertiary);
}

/* ── metric-card warn 变体（main.css 中未定义） ─────────────── */
:deep(.metric-card--warn) {
  border-left: 3px solid var(--status-warn);
}
</style>
