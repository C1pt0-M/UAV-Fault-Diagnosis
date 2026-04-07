<template>
  <div class="page-stack overview-command">
    <section class="overview-command__hero page-panel">
      <div class="overview-command__hero-main">
        <span class="overview-command__eyebrow">系统总览</span>
        <h2>智诊天眸 - 基于数据驱动的无人机故障检测系统</h2>
        <p>
          以遥测采集、在线预警、离线诊断、报告生成与 AI 解读组成统一闭环，
          将算法能力沉淀为覆盖监测、诊断、报告与解释的统一可视化工作台。
        </p>

        <div class="overview-command__hero-tags">
          <span class="overview-command__tag">实时监测</span>
          <span class="overview-command__tag">离线诊断</span>
          <span class="overview-command__tag">报告生成</span>
          <span class="overview-command__tag">AI 解释</span>
        </div>
      </div>

      <aside class="overview-command__mission-card">
        <div class="mission-card__header">
          <span>核心能力卡</span>
          <strong>System Snapshot</strong>
        </div>

        <div class="mission-card__metrics">
          <div class="mission-card__metric">
            <small>主能力</small>
            <strong>三分类离线诊断</strong>
          </div>
          <div class="mission-card__metric">
            <small>协同能力</small>
            <strong>在线预警联动</strong>
          </div>
        </div>

        <div class="mission-card__footer">
          <span>当前目标</span>
          <p>维持监测、诊断、报告与解释链路的稳定联动。</p>
        </div>
      </aside>
    </section>

    <section class="metric-grid overview-command__metrics">
      <article class="metric-card overview-metric overview-metric--a">
        <div class="overview-metric__label-row">
          <span class="metric-card__label">累计资产数</span>
          <el-tag size="small" round effect="light" type="success">Assets</el-tag>
        </div>
        <div class="metric-card__value">{{ overview.assets_total || 0 }}</div>
        <div class="metric-card__meta">当前可用于预处理、诊断与复盘的日志资产总量。</div>
      </article>

      <article class="metric-card overview-metric overview-metric--b">
        <div class="overview-metric__label-row">
          <span class="metric-card__label">累计任务数</span>
          <el-tag size="small" round effect="light" type="info">Runs</el-tag>
        </div>
        <div class="metric-card__value">{{ overview.runs_total || 0 }}</div>
        <div class="metric-card__meta">系统内已创建的预处理、诊断与相关任务总数。</div>
      </article>

      <article class="metric-card overview-metric overview-metric--c">
        <div class="overview-metric__label-row">
          <span class="metric-card__label">成功任务数</span>
          <el-tag size="small" round effect="light" type="warning">Success</el-tag>
        </div>
        <div class="metric-card__value">{{ successRuns }}</div>
        <div class="metric-card__meta">当前已经稳定完成并返回结果的任务数量。</div>
      </article>

      <article class="metric-card overview-metric overview-metric--d">
        <div class="overview-metric__label-row">
          <span class="metric-card__label">当前主故障</span>
          <el-tag size="small" round effect="light" type="danger">Focus</el-tag>
        </div>
        <div class="metric-card__value">{{ dominantFaultLabel }}</div>
        <div class="metric-card__meta">{{ dominantFaultMeta }}</div>
      </article>
    </section>

    <section class="section-card overview-command__flow">
      <div class="section-card__header">
        <div>
          <div class="section-card__title">五步诊断闭环</div>
        </div>
      </div>
      <div class="section-card__body">
        <div class="flow-lane">
          <article v-for="item in flowSteps" :key="item.key" class="flow-step">
            <span class="flow-step__index">{{ item.index }}</span>
            <strong>{{ item.title }}</strong>
            <p>{{ item.description }}</p>
          </article>
        </div>
      </div>
    </section>

    <section class="two-column overview-command__deck">
      <article class="section-card">
        <div class="section-card__header">
          <div>
            <div class="section-card__title">关键任务流</div>
          </div>
        </div>
        <div class="section-card__body">
          <div v-if="recentRuns.length" class="run-stream">
            <article v-for="item in recentRuns.slice(0, 6)" :key="item.id" class="run-stream__item">
              <div class="run-stream__status" :class="`is-${normalizeStatus(item.status)}`"></div>
              <div class="run-stream__content">
                <div class="run-stream__top">
                  <strong>#{{ item.id }} {{ normalizeTask(item.task) }}</strong>
                  <span>{{ normalizeStatusText(item.status) }}</span>
                </div>
                <p>{{ item.created_at || '-' }}</p>
              </div>
            </article>
          </div>
          <div v-else class="run-stream__empty">当前还没有可展示的任务流记录。</div>
        </div>
      </article>

      <article class="section-card overview-highlights">
        <div class="section-card__header">
          <div>
            <div class="section-card__title">系统亮点</div>
          </div>
        </div>
        <div class="section-card__body">
          <div class="highlight-list">
            <article v-for="item in highlightItems" :key="item.title" class="highlight-card">
              <span>{{ item.kicker }}</span>
              <strong>{{ item.title }}</strong>
              <p>{{ item.description }}</p>
            </article>
          </div>

          <div class="fault-breakdown">
            <div class="fault-breakdown__row">
              <span>正常</span>
              <strong>{{ faultDistribution.normal || 0 }}</strong>
            </div>
            <div class="fault-breakdown__bar"><i :style="{ width: `${distributionPercent.normal}%` }" class="is-normal" /></div>

            <div class="fault-breakdown__row">
              <span>电机故障</span>
              <strong>{{ faultDistribution.motor || 0 }}</strong>
            </div>
            <div class="fault-breakdown__bar"><i :style="{ width: `${distributionPercent.motor}%` }" class="is-motor" /></div>

            <div class="fault-breakdown__row">
              <span>传感器故障</span>
              <strong>{{ faultDistribution.sensor || 0 }}</strong>
            </div>
            <div class="fault-breakdown__bar"><i :style="{ width: `${distributionPercent.sensor}%` }" class="is-sensor" /></div>
          </div>
        </div>
      </article>
    </section>
  </div>
</template>

<script setup>
import { computed, onMounted, ref } from 'vue'
import { ElMessage, ElTag } from 'element-plus'
import { overviewApi } from '@/shared/services'

const overview = ref({})
const recentRuns = ref([])

const flowSteps = [
  { key: 'telemetry', index: '01', title: '遥测采集', description: '从 Replay / PX4 等来源统一进入数据链路。' },
  { key: 'detection', index: '02', title: '在线预警', description: '实时检测关键飞行信号，形成告警提示与会话历史。' },
  { key: 'diagnosis', index: '03', title: '离线诊断', description: '对日志资产执行故障分类与置信度推断。' },
  { key: 'report', index: '04', title: '报告生成', description: '把结果、证据与处置建议整理成可交付报告。' },
  { key: 'assistant', index: '05', title: 'AI 解读', description: '结合知识库与报告上下文解释故障原因与风险。' }
]

const highlightItems = [
  {
    kicker: 'ONLINE + OFFLINE',
    title: '监测与诊断不是割裂模块',
    description: '实时会话可直接承接到离线诊断，让算法展示更像真实系统。'
  },
  {
    kicker: 'REPORT READY',
    title: '从预测结果直接落到报告',
    description: '不是单纯输出标签，而是形成结论、建议与后续动作闭环。'
  },
  {
    kicker: 'AI ASSIST',
    title: '支持面向分析的解释能力',
    description: 'AI 助手围绕报告与知识库补充原因分析与术语解释。'
  }
]

const faultDistribution = computed(() => overview.value?.fault_distribution || { normal: 0, motor: 0, sensor: 0 })
const distributionTotal = computed(() => Object.values(faultDistribution.value).reduce((sum, value) => sum + Number(value || 0), 0))

const successRuns = computed(() => Number(overview.value?.runs_by_status?.success || 0))

const dominantFaultKey = computed(() => {
  const entries = [
    ['motor', Number(faultDistribution.value.motor || 0)],
    ['sensor', Number(faultDistribution.value.sensor || 0)],
    ['normal', Number(faultDistribution.value.normal || 0)]
  ]
  entries.sort((left, right) => right[1] - left[1])
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

function normalizeStatus(status) {
  if (status === 'success') return 'success'
  if (status === 'error' || status === 'failed') return 'error'
  if (status === 'running') return 'running'
  return 'queued'
}

function normalizeStatusText(status) {
  if (status === 'success') return '已完成'
  if (status === 'error' || status === 'failed') return '失败'
  if (status === 'running') return '执行中'
  if (status === 'queued') return '排队中'
  return status || '未知'
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
.overview-command__hero {
  display: grid;
  grid-template-columns: minmax(0, 1.35fr) minmax(320px, 0.9fr);
  gap: 28px;
  padding: 34px;
  overflow: hidden;
  background:
    radial-gradient(circle at right 14% top 18%, rgba(255, 255, 255, 0.56), transparent 24%),
    radial-gradient(circle at left 10% bottom 14%, rgba(15, 118, 110, 0.12), transparent 28%),
    linear-gradient(145deg, #fdfefe 0%, #eff5f7 52%, #e5eef3 100%);
  border: 1px solid rgba(10, 42, 58, 0.08);
  box-shadow: 0 22px 54px rgba(8, 31, 47, 0.08);
}

.overview-command__eyebrow {
  display: inline-flex;
  padding: 7px 14px;
  border-radius: 999px;
  background: rgba(8, 31, 47, 0.06);
  color: #0c5266;
  font-size: 12px;
  font-weight: 800;
  letter-spacing: 0.08em;
}

.overview-command__hero-main h2 {
  margin: 18px 0 0;
  max-width: 760px;
  font-size: 46px;
  line-height: 1.08;
  letter-spacing: -0.03em;
  color: #081f2f;
}

.overview-command__hero-main p {
  margin: 18px 0 0;
  max-width: 720px;
  color: #445e6b;
  font-size: 16px;
  line-height: 1.9;
}

.overview-command__hero-tags {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
  margin-top: 22px;
}

.overview-command__tag {
  padding: 9px 14px;
  border-radius: 999px;
  background: rgba(10, 42, 58, 0.06);
  border: 1px solid rgba(10, 42, 58, 0.08);
  color: #0a4b56;
  font-weight: 700;
}

.overview-command__mission-card {
  display: flex;
  flex-direction: column;
  justify-content: space-between;
  gap: 18px;
  padding: 22px;
  border-radius: 24px;
  background:
    radial-gradient(circle at top right, rgba(255, 255, 255, 0.16), transparent 30%),
    linear-gradient(150deg, #0a2232 0%, #0d4656 56%, #0f766e 100%);
  color: #f4fbfb;
  box-shadow: inset 0 0 0 1px rgba(255, 255, 255, 0.08);
}

.mission-card__header span {
  font-size: 12px;
  letter-spacing: 0.08em;
  color: rgba(244, 251, 251, 0.72);
}

.mission-card__header strong {
  display: block;
  margin-top: 8px;
  font-size: 28px;
  line-height: 1.15;
}

.mission-card__metrics {
  display: grid;
  gap: 12px;
}

.mission-card__metric {
  padding: 14px 16px;
  border-radius: 18px;
  background: rgba(255, 255, 255, 0.08);
  border: 1px solid rgba(255, 255, 255, 0.12);
}

.mission-card__metric small,
.mission-card__footer span {
  color: rgba(244, 251, 251, 0.74);
}

.mission-card__metric strong {
  display: block;
  margin-top: 6px;
  font-size: 22px;
  line-height: 1.35;
}

.mission-card__footer p {
  margin: 8px 0 0;
  line-height: 1.8;
  color: rgba(244, 251, 251, 0.84);
}

.overview-command__metrics {
  gap: 20px;
}

.overview-metric {
  position: relative;
  overflow: hidden;
}

.overview-metric::before {
  content: '';
  position: absolute;
  inset: 0 auto auto 0;
  width: 100%;
  height: 4px;
}

.overview-metric--a::before { background: linear-gradient(90deg, #0f766e, #14b8a6); }
.overview-metric--b::before { background: linear-gradient(90deg, #2563eb, #38bdf8); }
.overview-metric--c::before { background: linear-gradient(90deg, #d97706, #fbbf24); }
.overview-metric--d::before { background: linear-gradient(90deg, #dc2626, #fb7185); }

.overview-metric__label-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
}

.overview-command__flow,
.overview-highlights {
  overflow: hidden;
}

.flow-lane {
  display: grid;
  grid-template-columns: repeat(5, minmax(0, 1fr));
  gap: 16px;
}

.flow-step {
  min-height: 180px;
  padding: 18px;
  border-radius: 20px;
  background: linear-gradient(180deg, #fbfdfd 0%, #f2f7f9 100%);
  border: 1px solid rgba(10, 42, 58, 0.08);
}

.flow-step__index {
  display: inline-flex;
  width: 38px;
  height: 38px;
  align-items: center;
  justify-content: center;
  border-radius: 12px;
  background: #0c3344;
  color: #f4fbfb;
  font-size: 13px;
  font-weight: 800;
}

.flow-step strong {
  display: block;
  margin-top: 18px;
  font-size: 20px;
  line-height: 1.3;
}

.flow-step p {
  margin: 10px 0 0;
  color: var(--text-secondary);
  line-height: 1.8;
}

.overview-command__deck {
  align-items: stretch;
}

.run-stream {
  display: flex;
  flex-direction: column;
  gap: 14px;
}

.run-stream__item {
  display: grid;
  grid-template-columns: 12px minmax(0, 1fr);
  gap: 14px;
  align-items: stretch;
  padding: 14px 16px;
  border-radius: 18px;
  background: #fbfdfd;
  border: 1px solid rgba(10, 42, 58, 0.08);
}

.run-stream__status {
  border-radius: 999px;
  background: #94a3b8;
}

.run-stream__status.is-success { background: #16a34a; }
.run-stream__status.is-error { background: #dc2626; }
.run-stream__status.is-running { background: #2563eb; }
.run-stream__status.is-queued { background: #f59e0b; }

.run-stream__top {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 14px;
}

.run-stream__top strong {
  font-size: 16px;
  line-height: 1.4;
}

.run-stream__top span {
  color: #0c5266;
  font-size: 13px;
  font-weight: 700;
}

.run-stream__content p,
.run-stream__empty {
  margin: 6px 0 0;
  color: var(--text-secondary);
  line-height: 1.7;
}

.highlight-list {
  display: flex;
  flex-direction: column;
  gap: 14px;
}

.highlight-card {
  padding: 16px 18px;
  border-radius: 18px;
  background: linear-gradient(180deg, rgba(15, 118, 110, 0.06) 0%, rgba(255, 255, 255, 0.98) 100%);
  border: 1px solid rgba(15, 118, 110, 0.12);
}

.highlight-card span {
  font-size: 12px;
  font-weight: 800;
  color: #0c5266;
  letter-spacing: 0.08em;
}

.highlight-card strong {
  display: block;
  margin-top: 10px;
  font-size: 20px;
}

.highlight-card p {
  margin: 8px 0 0;
  color: var(--text-secondary);
  line-height: 1.8;
}

.fault-breakdown {
  margin-top: 18px;
  padding-top: 18px;
  border-top: 1px dashed rgba(10, 42, 58, 0.12);
}

.fault-breakdown__row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin: 12px 0 8px;
}

.fault-breakdown__bar {
  height: 10px;
  border-radius: 999px;
  background: #e6edf1;
  overflow: hidden;
}

.fault-breakdown__bar i {
  display: block;
  height: 100%;
  border-radius: inherit;
}

.fault-breakdown__bar .is-normal { background: linear-gradient(90deg, #22c55e, #4ade80); }
.fault-breakdown__bar .is-motor { background: linear-gradient(90deg, #ef4444, #fb7185); }
.fault-breakdown__bar .is-sensor { background: linear-gradient(90deg, #f59e0b, #fbbf24); }

@media (max-width: 1480px) {
  .overview-command__hero,
  .overview-command__deck {
    grid-template-columns: 1fr;
  }

  .flow-lane {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }
}
</style>
